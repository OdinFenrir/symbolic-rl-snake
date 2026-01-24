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


def _import_pygame():
    import os
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame  # type: ignore
    return pygame


class SnakeGame:
    """Game state manager."""

    def __init__(self, render_enabled: bool = True, seed: Optional[int] = None) -> None:
        self.render_enabled = bool(render_enabled)
        self.seed = seed
        self.rng = random.Random(seed)

        self.pygame = None
        self.screen = None
        self.clock = None
        self.font = None
        self.debug_font = None

        if self.render_enabled:
            self.pygame = _import_pygame()
            self.pygame.init()
            self.screen = self.pygame.display.set_mode((config.WINDOW_SIZE, config.WINDOW_SIZE))
            self.pygame.display.set_caption("Snake")
            self.clock = self.pygame.time.Clock()
            self.font = self.pygame.font.SysFont("Arial", 22, bold=True)
            self.debug_font = self.pygame.font.SysFont("Arial", 14)

        self.all_coords: Set[Tuple[int, int]] = {
            (r, c) for r in range(config.BOARD_SIZE) for c in range(config.BOARD_SIZE)
        }
        self.reset()

    def reset(self) -> None:
        start_pos = (config.BOARD_SIZE // 2, config.BOARD_SIZE // 2)
        self.snake: Deque[Tuple[int, int]] = deque([start_pos])
        self.direction = self.rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
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
        # Deterministic when seeded (set order is not stable).
        return self.rng.choice(sorted(available_positions))

    def _would_eat(self, new_head: Tuple[int, int]) -> bool:
        return self.food is not None and new_head == self.food

    def _check_collision(self, new_head: Tuple[int, int], will_eat: Optional[bool] = None) -> bool:

        # Backward compatible: callers may pass only new_head.

        if will_eat is None:

            will_eat = (self.food is not None and new_head == self.food)


        r, c = new_head

        if not (0 <= r < config.BOARD_SIZE and 0 <= c < config.BOARD_SIZE):

            return True


        # Collision with body. Moving into the current tail is legal if tail moves away (not eating).

        body = list(self.snake)[1:]

        if (not will_eat) and len(body) > 0:

            body = body[:-1]

        return new_head in body

    def _update_snake(self, action: Tuple[int, int]) -> Tuple[Tuple[int, int], bool]:

        new_head = (self.snake[0][0] + action[0], self.snake[0][1] + action[1])


        # Apply movement

        self.snake.appendleft(new_head)

        self.direction = action


        ate_food = (self.food is not None and new_head == self.food)


        if ate_food:

            self.score += 1

            self.life = min(self.life + config.LIFE_PER_FOOD, config.MAX_LIFE)

            # Keep tail (grow)

        else:

            # Normal move: tail advances (this also makes "move into tail" safe when not eating)

            self.snake.pop()


        # Rebuild occupancy to avoid edge-case desync (e.g., head stepping onto tail cell)

        self.occupied_cells = set(self.snake)


        if ate_food:

            self.food = self._place_food()


        self.life -= 1

        return new_head, ate_food

    def _compute_reward(self, ate_food: bool, new_head: Tuple[int, int], pre_move_state: Dict[str, Any]) -> float:
        reward = 0.0

        if ate_food:
            reward += config.REWARD_FOOD + (len(self.snake) * config.FOOD_REWARD_LENGTH_MULTIPLIER)

        # Distance shaping should reference the pre-move food position (not newly spawned food).
        pre_food = pre_move_state.get("food")
        if (pre_food is not None) and (not ate_food):
            old_dist = math.sqrt(float(pre_move_state["distance_sq"]))
            new_dist_sq = (new_head[0] - pre_food[0]) ** 2 + (new_head[1] - pre_food[1]) ** 2
            reward += (old_dist - math.sqrt(float(new_dist_sq))) * config.DISTANCE_REWARD_SCALAR

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
        will_eat = self._would_eat(new_head)

        if self._check_collision(new_head, will_eat=will_eat):
            self.game_over = True
            self.life = 0
            return config.PENALTY_DEATH

        _, ate_food = self._update_snake(action)
        return self._compute_reward(ate_food, new_head, pre_move_state)

    def render(self, debug_info: Optional[Dict[str, Any]] = None) -> None:
        if not self.render_enabled or self.screen is None or self.pygame is None:
            return

        pg = self.pygame
        self.screen.fill((20, 20, 30))

        for x in range(0, config.WINDOW_SIZE, config.GRID_SIZE):
            pg.draw.line(self.screen, (40, 40, 50), (x, 0), (x, config.WINDOW_SIZE))
        for y in range(0, config.WINDOW_SIZE, config.GRID_SIZE):
            pg.draw.line(self.screen, (40, 40, 50), (0, y), (config.WINDOW_SIZE, y))

        if config.UI_DEBUG_MODE and debug_info:
            if debug_info.get("a_star_path"):
                for r, c in debug_info["a_star_path"]:
                    rect = pg.Rect(c * config.GRID_SIZE, r * config.GRID_SIZE, config.GRID_SIZE, config.GRID_SIZE)
                    pg.draw.rect(self.screen, (50, 80, 120), rect, border_radius=3)

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

        for i, segment in enumerate(self.snake):
            rect = pg.Rect(segment[1] * config.GRID_SIZE, segment[0] * config.GRID_SIZE, config.GRID_SIZE, config.GRID_SIZE)
            if i == 0:
                pg.draw.rect(self.screen, (0, 255, 128), rect, border_radius=6)
            else:
                lerp_color = 200 - min(i * 10, 160)
                pg.draw.rect(self.screen, (0, lerp_color, lerp_color // 2), rect, border_radius=4)

        if self.food:
            rect = pg.Rect(self.food[1] * config.GRID_SIZE, self.food[0] * config.GRID_SIZE, config.GRID_SIZE, config.GRID_SIZE)
            pg.draw.rect(self.screen, (255, 80, 80), rect, border_radius=10)

        if self.font:
            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            life_text = self.font.render(f"Life: {int(self.life)}", True, (255, 255, 255))
            self.screen.blit(score_text, (10, 5))
            self.screen.blit(life_text, (10, 35))

        pg.display.flip()
        if self.clock:
            self.clock.tick(config.FPS)

    def save_game_state(self) -> None:
        state = {
            "version": 1,
            "board_size": int(config.BOARD_SIZE),
            "seed": self.seed,
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

            saved_board = int(state.get("board_size", config.BOARD_SIZE))
            if saved_board != int(config.BOARD_SIZE):
                logger.warning(
                    "Saved game uses board_size=%d but config.BOARD_SIZE=%d; loading anyway.",
                    saved_board,
                    int(config.BOARD_SIZE),
                )

            self.seed = state.get("seed", self.seed)
            self.rng = random.Random(self.seed)

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
