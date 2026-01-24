"""Persistent symbolic memory for experience-based move evaluation.

Fixes:
- Robust msgpack loading for tuple-based state keys.
- Avoids strict_map_key failures ("list/tuple is not allowed for map key...").
- One-shot backup restore (no infinite recursion).
- Versioned on-disk format (v2) that avoids non-string map keys entirely.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Tuple, Iterable

import errno
import msgpack

from . import config
from .rules import legal_moves

logger = logging.getLogger(__name__)

MEMORY_FORMAT_VERSION = 2


def _deep_tuple(x: Any) -> Any:
    """Recursively convert lists/tuples into tuples (for hashable keys)."""
    if isinstance(x, (list, tuple)):
        return tuple(_deep_tuple(i) for i in x)
    return x


def _sign_i(x: int) -> int:
    """Return -1, 0, or 1 depending on the sign of x (numpy-free)."""
    return (x > 0) - (x < 0)


class SymbolicMemory:
    """State-action memory with lightweight generalization and recursive scoring."""

    def __init__(self) -> None:
        self.memory: Dict[Tuple, Dict[str, Any]] = {}
        self.is_modified = False
        self.total_updates = 0
        self.load_memory()

    def create_state_key(
        self,
        snake: Deque[Tuple[int, int]],
        food: Optional[Tuple[int, int]],
        direction: Tuple[int, int],
    ) -> Tuple:
        """Create a compact generalized state key."""
        head = snake[0]

        # Food direction vector (sign only)
        food_dir = (
            (_sign_i(food[0] - head[0]), _sign_i(food[1] - head[1]))
            if food
            else (0, 0)
        )

        # Local obstacle map (8 neighbors)
        obstacles = set(snake)
        obstacle_map = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                pos = (head[0] + dr, head[1] + dc)
                is_wall = not (0 <= pos[0] < config.BOARD_SIZE and 0 <= pos[1] < config.BOARD_SIZE)
                is_body = pos in obstacles
                obstacle_map.append(is_wall or is_body)

        return (food_dir, tuple(obstacle_map), direction)

    def create_symbolic_state(
        self,
        snake: Deque[Tuple[int, int]],
        food: Optional[Tuple[int, int]],
        direction: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Return a symbolic representation used by the agent."""
        head = snake[0]
        distance_sq = (
            (head[0] - food[0]) ** 2 + (head[1] - food[1]) ** 2
            if food
            else float("inf")
        )

        safe_moves = legal_moves(snake, food, direction, forbid_reverse=True)

        return {
            "snake_head": head,
            "snake_body": snake,
            "food": food,
            "direction": direction,
            "distance_sq": distance_sq,
            "safe_moves": safe_moves,
        }

    def recursive_reasoning(self, state: Dict[str, Any], depth: int) -> Dict[Tuple[int, int], float]:
        """Recursive lookahead using stored values as priors."""
        if depth == 0 or not state.get("safe_moves"):
            return {move: 0.0 for move in state.get("safe_moves", [])}

        scores: Dict[Tuple[int, int], float] = defaultdict(float)
        for move in state["safe_moves"]:
            new_head = (state["snake_head"][0] + move[0], state["snake_head"][1] + move[1])

            # Simulate body advance (tail moves unless we ate the food).
            ate = (state.get("food") is not None and new_head == state.get("food"))
            if ate:
                temp_snake = deque([new_head] + list(state["snake_body"]))
                next_food = None  # after eating, new food is unknown to the lookahead
            else:
                temp_snake = deque([new_head] + list(state["snake_body"])[:-1])
                next_food = state.get("food")

            key = self.create_state_key(temp_snake, next_food, move)

            state_data = self.memory.get(key, {})
            action_str = f"{move[0]},{move[1]}"
            if action_str in state_data.get("actions", {}):
                scores[move] = float(state_data["actions"][action_str]["avg_reward"])

            new_state = self.create_symbolic_state(temp_snake, next_food, move)
            future_scores = self.recursive_reasoning(new_state, depth - 1)

            if future_scores:
                scores[move] += config.RSM_DECAY_FACTOR * max(future_scores.values())
            else:
                scores[move] += config.PENALTY_DEATH * config.RSM_DECAY_FACTOR

        return scores

    def update_memory(self, state: Dict[str, Any], action: Tuple[int, int], reward: float) -> None:
        """Update state-action statistics."""
        key = self.create_state_key(state["snake_body"], state["food"], state["direction"])
        action_str = f"{action[0]},{action[1]}"
        self.total_updates += 1

        if key not in self.memory:
            compact_state = {k: v for k, v in state.items() if k != "snake_body"}
            self.memory[key] = {
                "state_info": compact_state,
                "actions": {},
                "visits": 0,
                "last_visit_step": 0,
            }

        state_data = self.memory[key]
        state_data["visits"] += 1
        state_data["last_visit_step"] = self.total_updates

        if action_str not in state_data["actions"]:
            state_data["actions"][action_str] = {
                "count": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0,
            }

        action_data = state_data["actions"][action_str]
        action_data["count"] += 1
        action_data["total_reward"] += float(reward)

        alpha = 1.0 / action_data["count"]
        action_data["avg_reward"] = (1 - alpha) * action_data["avg_reward"] + alpha * float(reward)

        self.is_modified = True

        if len(self.memory) > config.MEMORY_MAX_ENTRIES:
            self.prune_memory()

    def prune_memory(self) -> None:
        """Prune memory based on recency-weighted utility."""
        if len(self.memory) <= config.MEMORY_MAX_ENTRIES:
            return

        logger.warning("Pruning memory (%d > %d)", len(self.memory), config.MEMORY_MAX_ENTRIES)

        pruning_scores = {
            k: v["visits"] - (self.total_updates - v["last_visit_step"]) * config.MEMORY_RECENCY_WEIGHT
            for k, v in self.memory.items()
        }

        sorted_keys = sorted(pruning_scores, key=pruning_scores.get, reverse=True)
        self.memory = {k: self.memory[k] for k in sorted_keys[: config.MEMORY_MAX_ENTRIES]}
        self.is_modified = True

    def _decode_payload(self, blob: bytes) -> tuple[Dict[Tuple, Dict[str, Any]], bool]:
        """Decode either legacy dict format or v2 payload format."""
        obj = msgpack.unpackb(
            blob,
            raw=False,
            strict_map_key=False,  # allow non-string keys in legacy files
            use_list=False,        # decode arrays as tuples (hashable)
        )

        # v2 format: {"v": 2, "states": [[key, value], ...], ...}
        if isinstance(obj, dict) and obj.get("v") == MEMORY_FORMAT_VERSION and "states" in obj:
            states = obj.get("states", [])
            mem: Dict[Tuple, Dict[str, Any]] = {}
            if isinstance(states, Iterable):
                for pair in states:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    k, v = pair
                    k = _deep_tuple(k)
                    if not isinstance(k, tuple) or not isinstance(v, dict):
                        continue
                    mem[k] = v
            return mem, False

        # legacy format: a dict with tuple-ish keys
        if isinstance(obj, dict):
            mem2: Dict[Tuple, Dict[str, Any]] = {}
            for k, v in obj.items():
                k = _deep_tuple(k)
                if isinstance(k, tuple) and isinstance(v, dict):
                    mem2[k] = v
            return mem2, True

        raise ValueError("Unsupported memory payload type")

    def load_memory(self) -> None:
        """Load memory file; if missing or corrupt, start fresh (one-shot backup restore)."""
        if not os.path.exists(config.MEMORY_FILE):
            logger.info("No memory file found; starting fresh")
            self.memory = {}
            self.is_modified = False
            return

        def _try_load(path: str) -> tuple[Dict[Tuple, Dict[str, Any]], bool]:
            with open(path, "rb") as f:
                return self._decode_payload(f.read())

        try:
            self.memory, legacy = _try_load(config.MEMORY_FILE)
        except Exception as e:
            logger.error("Load failed: %s", e)

            backup_file = config.MEMORY_FILE + ".bak"
            if os.path.exists(backup_file):
                try:
                    shutil.copy(backup_file, config.MEMORY_FILE)
                    logger.warning("Restored memory file from backup")
                    # One-shot retry only
                    self.memory, legacy = _try_load(config.MEMORY_FILE)
                except Exception as bak_e:
                    logger.error("Backup load failed: %s", bak_e)
                    # Preserve the broken file for inspection
                    try:
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        corrupt_name = config.MEMORY_FILE + f".corrupt.{ts}"
                        shutil.move(config.MEMORY_FILE, corrupt_name)
                        logger.warning("Moved corrupt memory file to %s", corrupt_name)
                    except Exception:
                        pass
                    self.memory = {}
                    return
            else:
                logger.warning("No backup available; starting fresh")
                self.memory = {}
                return

        self.total_updates = max(
            (int(v.get("last_visit_step", 0)) for v in self.memory.values()),
            default=0,
        )
        logger.info("Loaded %d states from %s", len(self.memory), config.MEMORY_FILE)

        # Only rewrite on next save if we loaded a legacy payload.
        self.is_modified = bool(legacy)

    def save_memory(self) -> None:
        """Save memory with a backup file (v2 format, atomic write)."""
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

        payload = {
            "v": MEMORY_FORMAT_VERSION,
            "total_updates": int(self.total_updates),
            # list of pairs avoids any map-key restrictions
            "states": [[k, v] for k, v in self.memory.items()],
        }

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
        except Exception as exc:  # pragma: no cover - fallback already handled by specific cases
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
