"""Persistent symbolic memory for experience-based move evaluation.

Key design goals:
- Robust msgpack persistence (backup + atomic write, tolerant decoding).
- Lightweight generalization via compact, constant-size state keys.
- Backwards-compatible reuse of older (v2 / legacy) memory files.

Notes on keying:
- v2 keys included the full per-segment age vector (variable length). That was precise but
  harmed generalization and increased memory sparsity.
- v3 introduces a compact key that uses constant-size local features and coarse length buckets.
- For compatibility, lookups fall back to v2-style keys when available; newly learned states
  maintain a small legacy->canonical index to preserve reuse without duplicating entries.
"""

from __future__ import annotations

import errno
import logging
import os
import shutil
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterable, Optional, Sequence, Tuple

import msgpack

from . import config
from .rules import legal_moves
from .utils import segment_age_sequence

logger = logging.getLogger(__name__)

# On-disk format version for the msgpack payload.
MEMORY_FORMAT_VERSION = 3

Move = Tuple[int, int]
Cell = Tuple[int, int]


def _deep_tuple(x: Any) -> Any:
    """Recursively convert lists/tuples into tuples (for hashable keys)."""
    if isinstance(x, (list, tuple)):
        return tuple(_deep_tuple(i) for i in x)
    return x


def _sign_i(x: int) -> int:
    """Return -1, 0, or 1 depending on the sign of x (numpy-free)."""
    return (x > 0) - (x < 0)


def _length_bucket(length: int, cutoffs: Sequence[int]) -> int:
    """Map a snake length to a small integer bucket based on configured cutoffs."""
    for idx, cutoff in enumerate(cutoffs):
        if length <= int(cutoff):
            return idx
    return len(cutoffs)


def _age_bin(age_ratio: float, bins: int) -> int:
    """Convert a normalized age ratio [0,1] to an integer bin [0..bins-1]."""
    if bins <= 1:
        return 0
    # age_ratio may be slightly out of bounds due to numeric drift; clamp defensively.
    x = float(age_ratio)
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    b = int(x * bins)
    if b >= bins:
        b = bins - 1
    return b


class SymbolicMemory:
    """State-action memory with lightweight generalization and recursive scoring."""

    def __init__(self) -> None:
        # Primary store: map from state_key -> entry.
        self.memory: Dict[Tuple, Dict[str, Any]] = {}

        # For v3 learning: map legacy_key -> canonical_key (without duplicating entries).
        self.legacy_index: Dict[Tuple, Tuple] = {}

        self.is_modified = False
        self.total_updates = 0
        self.load_memory()

    # ----------------------------
    # Key construction
    # ----------------------------
    def _food_dir(self, head: Cell, food: Optional[Cell]) -> Tuple[int, int]:
        return (
            (_sign_i(food[0] - head[0]), _sign_i(food[1] - head[1]))
            if food
            else (0, 0)
        )

    def _obstacle_map(self, head: Cell, obstacles: set[Cell]) -> Tuple[bool, ...]:
        """8-neighborhood obstacle occupancy (wall OR body)."""
        obstacle_map: list[bool] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                pos = (head[0] + dr, head[1] + dc)
                is_wall = not (0 <= pos[0] < config.BOARD_SIZE and 0 <= pos[1] < config.BOARD_SIZE)
                is_body = pos in obstacles
                obstacle_map.append(bool(is_wall or is_body))
        return tuple(obstacle_map)

    def _local_age_bins(
        self,
        head: Cell,
        obstacles: set[Cell],
        age_map: Dict[Cell, float],
        *,
        bins: int,
    ) -> Tuple[int, ...]:
        """8-neighborhood local age bins.

        Encoding per neighbor cell:
        -2 : wall
        -1 : empty
        0..bins-1 : occupied by body, binned by normalized age ratio (head=0, tail=1)
        """
        out: list[int] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                pos = (head[0] + dr, head[1] + dc)
                is_wall = not (0 <= pos[0] < config.BOARD_SIZE and 0 <= pos[1] < config.BOARD_SIZE)
                if is_wall:
                    out.append(-2)
                    continue
                if pos in obstacles:
                    out.append(_age_bin(age_map.get(pos, 1.0), bins))
                else:
                    out.append(-1)
        return tuple(out)

    def create_state_key(
        self,
        snake: Sequence[Cell],
        food: Optional[Cell],
        direction: Move,
        segment_ages: Optional[Tuple[float, ...]] = None,
    ) -> Tuple:
        """Create the v3 compact canonical state key (constant size)."""
        head = snake[0]
        ages = segment_ages if segment_ages is not None else segment_age_sequence(snake)

        food_dir = self._food_dir(head, food)
        obstacles = set(snake)
        obstacle_map = self._obstacle_map(head, obstacles)

        bins = int(getattr(config, "MEMORY_AGE_BINS", 4))
        cutoffs = tuple(getattr(config, "MEMORY_LENGTH_BUCKETS", (4, 8, 16, 32, 64)))
        length_bucket = _length_bucket(len(snake), cutoffs)

        age_map: Dict[Cell, float] = {cell: age for cell, age in zip(snake, ages)}
        local_age_bins = self._local_age_bins(head, obstacles, age_map, bins=bins)

        return (food_dir, obstacle_map, direction, int(length_bucket), local_age_bins)

    def create_legacy_state_key(
        self,
        snake: Sequence[Cell],
        food: Optional[Cell],
        direction: Move,
        segment_ages: Optional[Tuple[float, ...]] = None,
    ) -> Tuple:
        """Create the v2-style key (includes the full age sequence)."""
        head = snake[0]
        ages = segment_ages if segment_ages is not None else segment_age_sequence(snake)
        food_dir = self._food_dir(head, food)
        obstacles = set(snake)
        obstacle_map = self._obstacle_map(head, obstacles)
        return (food_dir, obstacle_map, direction, ages)

    def _lookup_entry(self, canonical_key: Tuple, legacy_key: Tuple) -> Optional[Dict[str, Any]]:
        entry = self.memory.get(canonical_key)
        if entry is not None:
            return entry

        if bool(getattr(config, "MEMORY_ENABLE_LEGACY_FALLBACK", True)):
            # Direct legacy-key lookup (for loaded v2/legacy memory files).
            entry = self.memory.get(legacy_key)
            if entry is not None:
                return entry

            # Legacy index for v3-learned states (legacy -> canonical).
            mapped = self.legacy_index.get(legacy_key)
            if mapped is not None:
                return self.memory.get(mapped)

        return None

    # ----------------------------
    # Symbolic state construction
    # ----------------------------
    def create_symbolic_state(
        self,
        snake: Deque[Cell],
        food: Optional[Cell],
        direction: Move,
        *,
        segment_ages: Optional[Tuple[float, ...]] = None,
    ) -> Dict[str, Any]:
        """Return a symbolic representation used by the agent."""
        head = snake[0]
        distance_sq = (
            (head[0] - food[0]) ** 2 + (head[1] - food[1]) ** 2
            if food
            else float("inf")
        )

        ages = segment_ages if segment_ages is not None else segment_age_sequence(snake)
        safe_moves = legal_moves(snake, food, direction, forbid_reverse=True)

        # Cache key features inside the state for downstream use.
        canonical_key = self.create_state_key(tuple(snake), food, direction, ages)
        legacy_key = self.create_legacy_state_key(tuple(snake), food, direction, ages)

        state = {
            "snake_head": head,
            "snake_body": tuple(snake),
            "food": food,
            "direction": direction,
            "distance_sq": distance_sq,
            "safe_moves": safe_moves,
            "segment_ages": ages,
            "snake_len": int(len(snake)),
            "state_key": canonical_key,
            "legacy_state_key": legacy_key,
        }
        return state

    # ----------------------------
    # Recursive lookahead
    # ----------------------------
    def recursive_reasoning(
        self,
        state: Dict[str, Any],
        depth: int,
        lookup_cache: Optional[Dict[Tuple, Optional[Dict[str, Any]]]] = None,
    ) -> Tuple[Dict[Move, float], int, int, Dict[Move, Dict[str, float]]]:
        """Recursive lookahead using stored values as priors."""
        if depth == 0 or not state.get("safe_moves"):
            return ({move: 0.0 for move in state.get("safe_moves", [])}, 0, 0, {})

        scores: Dict[Move, float] = defaultdict(float)
        hits = 0
        lookups = 0
        action_stats: Dict[Move, Dict[str, float]] = {}

        # Prefer precomputed keys from the state when present.
        canonical_key = state.get("state_key")
        legacy_key = state.get("legacy_state_key")
        if not isinstance(canonical_key, tuple) or not isinstance(legacy_key, tuple):
            canonical_key = self.create_state_key(
                state["snake_body"],
                state.get("food"),
                state["direction"],
                state.get("segment_ages"),
            )
            legacy_key = self.create_legacy_state_key(
                state["snake_body"],
                state.get("food"),
                state["direction"],
                state.get("segment_ages"),
            )

        if lookup_cache is None:
            lookup_cache = {}
        cached = lookup_cache.get(canonical_key)
        if cached is None:
            cached = self._lookup_entry(canonical_key, legacy_key) or {}
            lookup_cache[canonical_key] = cached
        state_data = cached
        actions = state_data.get("actions", {}) if isinstance(state_data, dict) else {}

        for move in state["safe_moves"]:
            lookups += 1
            new_head = (state["snake_head"][0] + move[0], state["snake_head"][1] + move[1])

            # Simulate body advance (tail moves unless we ate the food).
            ate = (state.get("food") is not None and new_head == state.get("food"))
            if ate:
                temp_snake = deque([new_head] + list(state["snake_body"]))
                next_food = None  # after eating, new food is unknown to the lookahead
            else:
                temp_snake = deque([new_head] + list(state["snake_body"])[:-1])
                next_food = state.get("food")

            action_str = f"{move[0]},{move[1]}"
            action_info = actions.get(action_str)
            if isinstance(action_info, dict) and "avg_reward" in action_info:
                hits += 1
                avg_reward = float(action_info["avg_reward"])
                scores[move] = avg_reward
                action_stats[move] = {
                    "avg_reward": avg_reward,
                    "count": int(action_info.get("count", 0)),
                }

            new_state = self.create_symbolic_state(temp_snake, next_food, move)
            future_scores, future_hits, future_lookups, _ = self.recursive_reasoning(
                new_state, depth - 1, lookup_cache=lookup_cache
            )
            hits += future_hits
            lookups += future_lookups

            if future_scores:
                scores[move] += float(config.RSM_DECAY_FACTOR) * max(future_scores.values())
            else:
                scores[move] += float(config.PENALTY_DEATH) * float(config.RSM_DECAY_FACTOR)

        return scores, hits, lookups, action_stats

    def action_stats_for_state(
        self,
        state: Dict[str, Any],
    ) -> Dict[Move, Dict[str, float]]:
        canonical_key = state.get("state_key")
        legacy_key = state.get("legacy_state_key")
        if not isinstance(canonical_key, tuple) or not isinstance(legacy_key, tuple):
            canonical_key = self.create_state_key(
                state["snake_body"],
                state.get("food"),
                state["direction"],
                state.get("segment_ages"),
            )
            legacy_key = self.create_legacy_state_key(
                state["snake_body"],
                state.get("food"),
                state["direction"],
                state.get("segment_ages"),
            )
        entry = self._lookup_entry(canonical_key, legacy_key) or {}
        actions = entry.get("actions", {}) if isinstance(entry, dict) else {}
        stats: Dict[Move, Dict[str, float]] = {}
        for action_str, info in actions.items():
            if not isinstance(info, dict):
                continue
            if "," not in action_str:
                continue
            try:
                move = tuple(int(token) for token in action_str.split(","))
            except ValueError:
                continue
            stats[move] = {
                "count": int(info.get("count", 0)),
                "avg_reward": float(info.get("avg_reward", 0.0)),
            }
        return stats

    # ----------------------------
    # Updates / pruning
    # ----------------------------
    def update_memory(
        self,
        state: Dict[str, Any],
        action: Move,
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        done: bool = False,
    ) -> None:
        """Update state-action statistics.

        The original implementation stored a running average of immediate rewards.
        That tends to *reinforce stalling loops* (episodes that end via step_cap)
        because those endings previously carried no negative learning signal.

        This update applies a lightweight TD(0) target so the stored value is
        effectively a Q-value estimate for (state, action):

            Q(s,a) <- (1-alpha) Q(s,a) + alpha * (r + gamma * max_a' Q(s',a'))

        Using a small step penalty and an explicit step-cap terminal penalty,
        this encourages shorter, food-seeking trajectories and reduces looping.
        """

        # Count updates for recency-weighted pruning.
        self.total_updates = int(self.total_updates) + 1

        canonical_key = state.get("state_key")
        legacy_key = state.get("legacy_state_key")

        # Backwards compatibility: some callers/tests provide raw state dicts with
        # "snake_body" (deque/tuple) rather than a precomputed state_key.
        if canonical_key is None or legacy_key is None:
            snake_body = state.get("snake_body") or state.get("snake") or ()
            snake_t = tuple(snake_body)
            food = state.get("food")
            direction = state.get("direction", (0, 1))
            ages = state.get("segment_ages")
            if ages is None:
                try:
                    ages = segment_age_sequence(snake_t)
                except Exception:
                    ages = ()
            if canonical_key is None:
                canonical_key = self.create_state_key(snake_t, food, direction, ages)
            if legacy_key is None:
                legacy_key = self.create_legacy_state_key(snake_t, food, direction, ages)

        action_key = f"{action[0]},{action[1]}"
        entry = self._lookup_entry(canonical_key, legacy_key)

        if not isinstance(entry, dict):
            entry = {
                "last_seen": time.time(),
                "last_visit_step": int(self.total_updates),
                "actions": {},
                "visits": 0,
                "last_visit_step": int(self.total_updates),
            }
            self.memory[canonical_key] = entry

        actions = entry["actions"]
        a = actions.setdefault(
            action_key,
            {
                "avg_reward": 0.0,
                "count": 0,
                "total_reward": 0.0,
                "last_seen": time.time(),
            },
        )

        # TD target
        target = float(reward)
        if (next_state is not None) and (not done):
            next_key = next_state.get("state_key")
            if next_key is None:
                snake_body = next_state.get("snake_body") or next_state.get("snake") or ()
                snake_t = tuple(snake_body)
                food = next_state.get("food")
                direction = next_state.get("direction", (0, 1))
                ages = next_state.get("segment_ages")
                if ages is None:
                    try:
                        ages = segment_age_sequence(snake_t)
                    except Exception:
                        ages = ()
                next_key = self.create_state_key(snake_t, food, direction, ages)
            target += float(config.MEMORY_GAMMA) * self._max_q_for_state_key(next_key)

        a["count"] = int(a.get("count", 0)) + 1
        a["total_reward"] = float(a.get("total_reward", 0.0)) + float(reward)
        a["last_seen"] = time.time()
        a["last_visit_step"] = int(self.total_updates)

        # Slightly decaying alpha, capped by config.MEMORY_ALPHA
        alpha = min(float(config.MEMORY_ALPHA), 1.0 / float(a["count"]))
        old_q = float(a.get("avg_reward", 0.0))
        a["avg_reward"] = (1.0 - alpha) * old_q + alpha * float(target)

        entry["visits"] = int(entry.get("visits", 0)) + 1
        entry["last_seen"] = time.time()
        entry["last_visit_step"] = int(self.total_updates)
        self.is_modified = True

        if len(self.memory) > config.MEMORY_MAX_ENTRIES:
            self.prune_memory()

    def _max_q_for_state_key(self, state_key: Any) -> float:
        entry = self._lookup_entry(state_key, state_key)
        if not isinstance(entry, dict):
            return 0.0
        actions = entry.get("actions", {})
        if not actions:
            return 0.0
        return max(float(v.get("avg_reward", 0.0)) for v in actions.values())

    def prune_memory(self) -> None:
        """Prune memory based on recency-weighted utility."""
        if len(self.memory) <= config.MEMORY_MAX_ENTRIES:
            return

        logger.warning("Pruning memory (%d > %d)", len(self.memory), config.MEMORY_MAX_ENTRIES)

        pruning_scores = {
            k: float(v.get("visits", 0)) - (float(self.total_updates) - float(v.get("last_visit_step", 0))) * float(config.MEMORY_RECENCY_WEIGHT)
            for k, v in self.memory.items()
        }

        sorted_keys = sorted(pruning_scores, key=pruning_scores.get, reverse=True)
        self.memory = {k: self.memory[k] for k in sorted_keys[: int(config.MEMORY_MAX_ENTRIES)]}

        # Also prune legacy_index mappings to keys that remain.
        live = set(self.memory.keys())
        if self.legacy_index:
            self.legacy_index = {lk: ck for lk, ck in self.legacy_index.items() if ck in live}

        self.is_modified = True

    # ----------------------------
    # Persistence and reset
    # ----------------------------
    def reset(self, *, delete_files: bool = False) -> None:
        """Clear in-memory memory; optionally remove on-disk files.

        If SAVE_MEMORY is enabled, the next save will persist the cleared state.
        """
        self.memory.clear()
        self.legacy_index.clear()
        self.total_updates = 0
        self.is_modified = True

        if delete_files:
            self._delete_memory_files()

    def _delete_memory_files(self) -> None:
        for suffix in ("", ".bak", ".tmp"):
            path = config.MEMORY_FILE + suffix
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                logger.warning("Unable to delete %s: %s", path, exc)

    def _decode_payload(self, blob: bytes) -> tuple[Dict[Tuple, Dict[str, Any]], Dict[Tuple, Tuple], bool, int]:
        """Decode legacy dict format, v2 payload, or v3 payload."""
        obj = msgpack.unpackb(
            blob,
            raw=False,
            strict_map_key=False,  # allow non-string keys in legacy files
            use_list=False,        # decode arrays as tuples (hashable)
        )

        if isinstance(obj, dict) and "v" in obj and "states" in obj:
            v = int(obj.get("v", 0) or 0)
            states = obj.get("states", [])
            mem: Dict[Tuple, Dict[str, Any]] = {}
            if isinstance(states, Iterable):
                for pair in states:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    k, val = pair
                    k = _deep_tuple(k)
                    if not isinstance(k, tuple) or not isinstance(val, dict):
                        continue
                    mem[k] = val

            legacy_index: Dict[Tuple, Tuple] = {}
            if v >= 3:
                pairs = obj.get("legacy_index", [])
                if isinstance(pairs, Iterable):
                    for pair in pairs:
                        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                            continue
                        lk, ck = pair
                        lk = _deep_tuple(lk)
                        ck = _deep_tuple(ck)
                        if isinstance(lk, tuple) and isinstance(ck, tuple):
                            legacy_index[lk] = ck

            total_updates = int(obj.get("total_updates", 0) or 0)
            # If v < current version, consider it "legacy" for rewrite on next save.
            legacy_payload = v != MEMORY_FORMAT_VERSION
            return mem, legacy_index, legacy_payload, total_updates

        # legacy dict format: a dict with tuple-ish keys
        if isinstance(obj, dict):
            mem2: Dict[Tuple, Dict[str, Any]] = {}
            for k, val in obj.items():
                k = _deep_tuple(k)
                if isinstance(k, tuple) and isinstance(val, dict):
                    mem2[k] = val
            return mem2, {}, True, 0

        raise ValueError("Unsupported memory payload type")

    def load_memory(self) -> None:
        """Load memory file; if missing or corrupt, start fresh (one-shot backup restore)."""
        if not os.path.exists(config.MEMORY_FILE):
            logger.info("No memory file found; starting fresh")
            self.memory = {}
            self.legacy_index = {}
            self.is_modified = False
            return

        def _try_load(path: str) -> tuple[Dict[Tuple, Dict[str, Any]], Dict[Tuple, Tuple], bool, int]:
            with open(path, "rb") as f:
                return self._decode_payload(f.read())

        try:
            mem, legacy_index, legacy, total_updates = _try_load(config.MEMORY_FILE)
        except Exception as e:
            logger.error("Load failed: %s", e)

            backup_file = config.MEMORY_FILE + ".bak"
            if os.path.exists(backup_file):
                try:
                    shutil.copy(backup_file, config.MEMORY_FILE)
                    logger.warning("Restored memory file from backup")
                    mem, legacy_index, legacy, total_updates = _try_load(config.MEMORY_FILE)
                except Exception as bak_e:
                    logger.error("Backup load failed: %s", bak_e)
                    try:
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        corrupt_name = config.MEMORY_FILE + f".corrupt.{ts}"
                        shutil.move(config.MEMORY_FILE, corrupt_name)
                        logger.warning("Moved corrupt memory file to %s", corrupt_name)
                    except Exception:
                        pass
                    self.memory = {}
                    self.legacy_index = {}
                    return
            else:
                logger.warning("No backup available; starting fresh")
                self.memory = {}
                self.legacy_index = {}
                return

        self.memory = mem
        self.legacy_index = legacy_index
        # Derive total_updates if missing.
        self.total_updates = max(
            int(total_updates),
            max((int(v.get("last_visit_step", 0)) for v in self.memory.values()), default=0),
        )
        logger.info("Loaded %d states from %s", len(self.memory), config.MEMORY_FILE)

        # Only rewrite on next save if we loaded a legacy payload.
        self.is_modified = bool(legacy)

    def save_memory(self) -> None:
        """Save memory with a backup file (v3 format, atomic write)."""
        if not config.SAVE_MEMORY:
            return
        if not self.is_modified:
            return

        os.makedirs(os.path.dirname(config.MEMORY_FILE), exist_ok=True)

        # Backup current file
        if os.path.exists(config.MEMORY_FILE):
            try:
                shutil.copy(config.MEMORY_FILE, config.MEMORY_FILE + ".bak")
            except Exception as e:
                logger.error("Backup failed: %s", e)

        payload: Dict[str, Any] = {
            "v": MEMORY_FORMAT_VERSION,
            "total_updates": int(self.total_updates),
            # list of pairs avoids any map-key restrictions
            "states": [[k, v] for k, v in self.memory.items()],
        }
        if self.legacy_index:
            payload["legacy_index"] = [[lk, ck] for lk, ck in self.legacy_index.items()]

        blob = msgpack.packb(payload, use_bin_type=True)
        tmp_path = config.MEMORY_FILE + ".tmp"

        def _write_blob(path: str) -> None:
            with open(path, "wb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())

        try:
            _write_blob(tmp_path)
        except Exception as exc:
            logger.error("Failed to write temp payload: %s", exc)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return

        saved = False
        try:
            os.replace(tmp_path, config.MEMORY_FILE)
            saved = True
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EPERM}:
                logger.warning("Atomic replace denied (%s); falling back to overwrite", exc)
                try:
                    _write_blob(config.MEMORY_FILE)
                    saved = True
                except Exception as fallback_exc:
                    logger.error("Fallback write failed: %s", fallback_exc)
            else:
                logger.error("Save failed: %s", exc)
        except Exception as exc:  # pragma: no cover
            logger.error("Save failed: %s", exc)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        if saved:
            logger.info("Saved %d states to %s", len(self.memory), config.MEMORY_FILE)
            self.is_modified = False
