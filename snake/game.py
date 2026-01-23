"""Snake game engine with optional pygame rendering and state persistence."""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Set, Tuple

from . import config

logger = logging.getLogger(__name__)

# pygame is optional if you run headless
try:
    import pygame  # type: ignore
except Exception:
    pygame = None  # type: ignore


class SnakeGame:
    """Game state manager."""

    def __init__(self, render_enabled: bool = True, seed: Optional[int] = None) -> None:
        self.render_enabled = render_enabled
        self.screen = None
        self.clock = None
        self.font = None
        self.debug_font = None

        if seed is not None:
            random.seed(seed)

        if self.render_enabled:
            if pygame is None:
                raise RuntimeError("Rendering requires pygame. Install it or run with --no-render.")
            pygame.init()
            self.screen = pygame.display.set_mode((config.WINDOW_SIZE, config.WINDOW_SIZE))
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 22, bold=True)
            self.debug_font = pygame.font.SysFont("Arial", 14)

        self.all_coords: Set[Tuple[int, int]] = {
            (r, c) for r in range(config.BOARD_SIZE) for c in range(config.BOARD_SIZE)
        }
        self.reset()

    def reset(self) -> None:
        start_pos = (config.BOARD_SIZE // 2, config.BOARD_SIZE // 2)
        self.snake: Deque[Tuple[int, int]] = deque([start_pos])
        self.direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        self.occupied_cells = {start_pos}
        self.food = self._place_food()
        self.score = 0
        self.life = config.INITIAL_LIFE
        self.game_over = False
        self.consecutive_moves: Dict[str, int] = defaultdict(int)

    def _place_food(self) -> Optional[Tuple[int, int]]:
        available_positions = self.all_coords - self.occupied_cells
        if not available_positions:
            logger.info("No space for food - snake wins")
            return None
        return random.choice(list(available_positions))

    def _update_snake(self, action: Tuple[int, int]) -> Tuple[Tuple[int, int], bool]:
        new_head = (self.snake[0][0] + action[0], self.snake[0][1] + action[1])

        self.snake.appendleft(new_head)
        self.occupied_cells.add(new_head)
        self.direction = action

        ate_food = False
        if new_head == self.food:
            self.score += 1
            self.life = min(self.life + config.LIFE_PER_FOOD, config.MAX_LIFE)
            self.food = self._place_food()
            ate_food = True
        else:
            tail = self.snake.pop()
            self.occupied_cells.remove(tail)

        self.life -= 1
        return new_head, ate_food

    def _check_collision(self, new_head: Tuple[int, int]) -> bool:
        r, c = new_head
        is_wall = not (0 <= r < config.BOARD_SIZE and 0 <= c < config.BOARD_SIZE)
        is_self = new_head in list(self.snake)[1:]
        return is_wall or is_self

    def _compute_reward(self, ate_food: bool, new_head: Tuple[int, int], pre_move_state: Dict[str, Any]) -> float:
        reward = 0.0

        if ate_food:
            reward += config.REWARD_FOOD + (len(self.snake) * config.FOOD_REWARD_LENGTH_MULTIPLIER)

        if self.food:
            old_dist = math.sqrt(pre_move_state["distance_sq"])
            new_dist_sq = (new_head[0] - self.food[0]) ** 2 + (new_head[1] - self.food[1]) ** 2
            reward += (old_dist - math.sqrt(new_dist_sq)) * config.DISTANCE_REWARD_SCALAR

        action_str = f"{self.direction[0]},{self.direction[1]}"
        self.consecutive_moves[action_str] += 1
        for k in list(self.consecutive_moves.keys()):
            if k != action_str:
                self.consecutive_moves[k] = 0
        if self.consecutive_moves[action_str] > 8:
            reward += config.PENALTY_REPEAT * min(self.consecutive_moves[action_str], 20)

        if new_head[0] in (0, config.BOARD_SIZE - 1) or new_head[1] in (0, config.BOARD_SIZE - 1):
            reward += config.PENALTY_WALL_HUG

        return reward

    def move_snake(self, action: Tuple[int, int], pre_move_state: Dict[str, Any]) -> float:
        if self.game_over:
            return 0.0

        new_head = (self.snake[0][0] + action[0], self.snake[0][1] + action[1])

        if self._check_collision(new_head):
            self.game_over = True
            self.life = 0
            return config.PENALTY_DEATH

        _, ate_food = self._update_snake(action)
        return self._compute_reward(ate_food, new_head, pre_move_state)

    def render(self, debug_info: Optional[Dict[str, Any]] = None) -> None:
        if not self.render_enabled or self.screen is None or pygame is None:
            return

        self.screen.fill((20, 20, 30))

        # Grid
        for x in range(0, config.WINDOW_SIZE, config.GRID_SIZE):
            pygame.draw.line(self.screen, (40, 40, 50), (x, 0), (x, config.WINDOW_SIZE))
        for y in range(0, config.WINDOW_SIZE, config.GRID_SIZE):
            pygame.draw.line(self.screen, (40, 40, 50), (0, y), (config.WINDOW_SIZE, y))

        # Debug overlay
        if config.UI_DEBUG_MODE and debug_info:
            if debug_info.get("a_star_path"):
                for r, c in debug_info["a_star_path"]:
                    rect = pygame.Rect(
                        c * config.GRID_SIZE, r * config.GRID_SIZE, config.GRID_SIZE, config.GRID_SIZE
                    )
                    pygame.draw.rect(self.screen, (50, 80, 120), rect, border_radius=3)

            if debug_info.get("final_scores") and self.debug_font:
                head_r, head_c = self.snake[0]
                for move, score in debug_info["final_scores"].items():
                    dr, dc = move
                    text_surf = self.debug_font.render(f"{score:.1f}", True, (200, 200, 220))
                    text_rect = text_surf.get_rect(
                        center=(
                            (head_c + dc) * config.GRID_SIZE + config.GRID_SIZE / 2,
                            (head_r + dr) * config.GRID_SIZE + config.GRID_SIZE / 2,
                        )
                    )
                    self.screen.blit(text_surf, text_rect)

        # Snake
        for i, segment in enumerate(self.snake):
            rect = pygame.Rect(
                segment[1] * config.GRID_SIZE,
                segment[0] * config.GRID_SIZE,
                config.GRID_SIZE,
                config.GRID_SIZE,
            )
            if i == 0:
                pygame.draw.rect(self.screen, (0, 255, 128), rect, border_radius=6)
            else:
                lerp_color = 200 - min(i * 10, 160)
                pygame.draw.rect(self.screen, (0, lerp_color, lerp_color // 2), rect, border_radius=4)

        # Food
        if self.food:
            rect = pygame.Rect(
                self.food[1] * config.GRID_SIZE,
                self.food[0] * config.GRID_SIZE,
                config.GRID_SIZE,
                config.GRID_SIZE,
            )
            pygame.draw.rect(self.screen, (255, 80, 80), rect, border_radius=10)

        if self.font:
            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            life_text = self.font.render(f"Life: {int(self.life)}", True, (255, 255, 255))
            self.screen.blit(score_text, (10, 5))
            self.screen.blit(life_text, (10, 35))

        pygame.display.flip()
        if self.clock:
            self.clock.tick(config.FPS)

    def save_game_state(self) -> None:
        state = {
            "snake": list(self.snake),
            "direction": self.direction,
            "food": self.food,
            "score": self.score,
            "life": self.life,
            "game_over": self.game_over,
            "consecutive_moves": dict(self.consecutive_moves),
        }
        import os
        os.makedirs(os.path.dirname(config.GAME_STATE_FILE), exist_ok=True)
        with open(config.GAME_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_game_state(self) -> bool:
        import os
        if not os.path.exists(config.GAME_STATE_FILE):
            return False

        try:
            with open(config.GAME_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.snake = deque([tuple(x) for x in state["snake"]])
            self.direction = tuple(state["direction"])
            self.food = tuple(state["food"]) if state["food"] is not None else None
            self.score = int(state["score"])
            self.life = float(state["life"])
            self.game_over = bool(state["game_over"])

            self.occupied_cells = set(self.snake)
            self.consecutive_moves = defaultdict(int, state.get("consecutive_moves", {}))

            return True
        except Exception as e:
            logger.error("Load game state failed: %s", e)
            return False
