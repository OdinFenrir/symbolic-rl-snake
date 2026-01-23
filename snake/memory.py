"""Persistent symbolic memory for experience-based move evaluation."""

from __future__ import annotations

import os
import shutil
import logging
from collections import defaultdict
from typing import Any, Deque, Dict, Optional, Tuple

import msgpack
import numpy as np

from . import config

logger = logging.getLogger(__name__)


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
            (int(np.sign(food[0] - head[0])), int(np.sign(food[1] - head[1])))
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
                is_wall = not (
                    0 <= pos[0] < config.BOARD_SIZE and 0 <= pos[1] < config.BOARD_SIZE
                )
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
            (head[0] - food[0]) ** 2 + (head[1] - food[1]) ** 2 if food else float("inf")
        )

        safe_moves = []
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            new_head = (head[0] + dr, head[1] + dc)
            is_wall = not (0 <= new_head[0] < config.BOARD_SIZE and 0 <= new_head[1] < config.BOARD_SIZE)
            is_self = new_head in snake
            if not is_wall and not is_self:
                safe_moves.append((dr, dc))

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

            # Simulate body advance (tail moves)
            from collections import deque
            temp_snake = deque([new_head] + list(state["snake_body"])[:-1])

            key = self.create_state_key(temp_snake, state["food"], move)

            state_data = self.memory.get(key, {})
            action_str = f"{move[0]},{move[1]}"
            if action_str in state_data.get("actions", {}):
                scores[move] = float(state_data["actions"][action_str]["avg_reward"])

            new_state = self.create_symbolic_state(temp_snake, state["food"], move)
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

    def load_memory(self) -> None:
        """Load memory file; if missing or corrupt, start fresh."""
        if not os.path.exists(config.MEMORY_FILE):
            logger.info("No memory file found; starting fresh")
            self.memory = {}
            return

        try:
            with open(config.MEMORY_FILE, "rb") as f:
                loaded_data = msgpack.load(f, raw=False)

            self.memory = {
                tuple(k) if isinstance(k, list) else k: v for k, v in loaded_data.items()
            }

            self.total_updates = max(
                (int(v.get("last_visit_step", 0)) for v in self.memory.values()),
                default=0,
            )
            logger.info("Loaded %d states from %s", len(self.memory), config.MEMORY_FILE)
        except Exception as e:
            logger.error("Load failed: %s", e)
            backup_file = config.MEMORY_FILE + ".bak"
            if os.path.exists(backup_file):
                try:
                    shutil.copy(backup_file, config.MEMORY_FILE)
                    logger.warning("Restored memory file from backup")
                    self.load_memory()
                except Exception as bak_e:
                    logger.error("Backup restore failed: %s", bak_e)
                    self.memory = {}
            else:
                logger.warning("No backup available; starting fresh")
                self.memory = {}

    def save_memory(self) -> None:
        """Save memory with a backup file."""
        if not self.is_modified:
            return

        os.makedirs(os.path.dirname(config.MEMORY_FILE), exist_ok=True)

        if os.path.exists(config.MEMORY_FILE):
            try:
                shutil.copy(config.MEMORY_FILE, config.MEMORY_FILE + ".bak")
            except Exception as e:
                logger.error("Backup failed: %s", e)

        try:
            with open(config.MEMORY_FILE, "wb") as f:
                msgpack.dump(self.memory, f, use_bin_type=True)
            logger.info("Saved %d states to %s", len(self.memory), config.MEMORY_FILE)
            self.is_modified = False
        except Exception as e:
            logger.error("Save failed: %s", e)
