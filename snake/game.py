"""Snake game engine with optional pygame rendering and state persistence."""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Optional, Set, Tuple

from . import config
from .rules import legal_moves, is_legal_move, Move

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

        self.menu_visible = bool(self.render_enabled)
        self.menu_state = "main"
        self.menu_index = 0
        self.menu_has_started = False
        self.menu_message = "Use ↑/↓ to navigate, Enter to select."
        self.menu_memory_requested = False
        self.menu_show_metrics = False
        self.metrics_display = False
        self.live_metrics: Dict[str, Any] = {}
        self.menu_show_metrics = False
        self.metrics_info: Dict[str, Any] = {}
        self.metrics_overlay: Optional[list[str]] = None
        self.metrics_overlay_ticks = 0

        self.all_coords: Set[Tuple[int, int]] = {
            (r, c) for r in range(config.BOARD_SIZE) for c in range(config.BOARD_SIZE)
        }
        self.won = False
        self.reset()

    def _lerp_color(self, start: Tuple[int, int, int], end: Tuple[int, int, int], ratio: float) -> Tuple[int, int, int]:
        ratio = max(0.0, min(1.0, ratio))
        return (
            int(start[0] + (end[0] - start[0]) * ratio),
            int(start[1] + (end[1] - start[1]) * ratio),
            int(start[2] + (end[2] - start[2]) * ratio),
        )

    def _fill_vertical_gradient(
        self,
        surface: Any,
        rect: "pygame.Rect",
        start_color: Tuple[int, int, int],
        end_color: Tuple[int, int, int],
    ) -> None:  # type: ignore[name-defined]
        height = max(rect.height, 1)
        denom = height - 1 if height > 1 else 1
        for y in range(rect.height):
            ratio = y / denom
            color = self._lerp_color(start_color, end_color, ratio)
            surface.fill(color, (rect.x, rect.y + y, rect.width, 1))

    def _lerp_color(self, start: Tuple[int, int, int], end: Tuple[int, int, int], ratio: float) -> Tuple[int, int, int]:
        ratio = max(0.0, min(1.0, ratio))
        return (
            int(start[0] + (end[0] - start[0]) * ratio),
            int(start[1] + (end[1] - start[1]) * ratio),
            int(start[2] + (end[2] - start[2]) * ratio),
        )

    def _fill_vertical_gradient(self, surface: Any, rect: "pygame.Rect", start_color: Tuple[int, int, int], end_color: Tuple[int, int, int]) -> None:  # type: ignore[name-defined]
        height = max(rect.height, 1)
        for y in range(rect.height):
            ratio = y / (height - 1)
            color = self._lerp_color(start_color, end_color, ratio)
            surface.fill(color, (rect.x, rect.y + y, rect.width, 1))

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
        self.won = False

    def _place_food(self) -> Optional[Tuple[int, int]]:
        available_positions = self.all_coords - self.occupied_cells
        if not available_positions:
            logger.info("No space for food - snake wins")
            self.game_over = True
            self.won = True
            return None
        # Deterministic when seeded (set order is not stable).
        return self.rng.choice(sorted(available_positions))

    def _would_eat(self, new_head: Tuple[int, int]) -> bool:
        return self.food is not None and new_head == self.food

    def _check_collision(self, new_head: Tuple[int, int]) -> bool:
        move = (new_head[0] - self.snake[0][0], new_head[1] - self.snake[0][1])
        return not is_legal_move(self.snake, self.food, move, self.direction)

    def _update_snake(self, action: Tuple[int, int]) -> Tuple[Tuple[int, int], bool]:

        new_head = (self.snake[0][0] + action[0], self.snake[0][1] + action[1])


        # Apply movement

        self.snake.appendleft(new_head)

        self.direction = action


        ate_food = (self.food is not None and new_head == self.food)


        if ate_food:
            self.score += 1
            self.life = min(self.life + config.LIFE_PER_FOOD, config.MAX_LIFE)
        else:
            self.snake.pop()

        self.occupied_cells = set(self.snake)

        if ate_food:
            next_food = self._place_food()
            if next_food is None:
                self.food = None
                self.life = min(self.life, config.MAX_LIFE)
                return new_head, ate_food
            self.food = next_food
        self.life -= 1
        return new_head, ate_food

    def _compute_reward(self, ate_food: bool, new_head: Tuple[int, int], pre_move_state: Dict[str, Any]) -> float:
        reward = 0.0

        if ate_food:
            reward += config.REWARD_FOOD
            reward += (
                (len(self.snake) ** config.FOOD_REWARD_POWER)
                * config.FOOD_REWARD_LENGTH_MULTIPLIER
            )

        # Distance shaping should reference the pre-move food position (not newly spawned food).
        pre_food = pre_move_state.get("food")
        if (pre_food is not None) and (not ate_food):
            old_head = pre_move_state.get("snake_head", self.snake[0])
            old_dist = abs(old_head[0] - pre_food[0]) + abs(old_head[1] - pre_food[1])
            new_dist = abs(new_head[0] - pre_food[0]) + abs(new_head[1] - pre_food[1])
            delta = old_dist - new_dist
            if delta > 0:
                delta = min(delta, config.MAX_DISTANCE_REWARD_DELTA)
            reward += delta * config.DISTANCE_REWARD_SCALAR

        action_str = f"{self.direction[0]},{self.direction[1]}"
        self.consecutive_moves[action_str] += 1
        for k in list(self.consecutive_moves.keys()):
            if k != action_str:
                self.consecutive_moves[k] = 0
        count = self.consecutive_moves[action_str]
        if count > config.REPEAT_THRESHOLD:
            over = count - config.REPEAT_THRESHOLD
            reward += config.PENALTY_REPEAT * min(over, config.REPEAT_PENALTY_CAP_STEPS)

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
        if self.life <= 0:
            self.game_over = True
            return config.PENALTY_DEATH
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

        head_color = (0, 255, 180)
        body_start = (0, 160, 220)
        body_end = (10, 40, 80)
        total_segments = max(len(self.snake) - 1, 1)
        for i, segment in enumerate(self.snake):
            rect = pg.Rect(segment[1] * config.GRID_SIZE, segment[0] * config.GRID_SIZE, config.GRID_SIZE, config.GRID_SIZE)
            if i == 0:
                pg.draw.rect(self.screen, head_color, rect, border_radius=6)
            else:
                ratio = (i - 1) / total_segments
                color = self._lerp_color(body_start, body_end, ratio)
                gradient_rect = self.pygame.Rect(rect)
                self._fill_vertical_gradient(self.screen, gradient_rect, (color[0], color[1], color[2]), (0, 0, 0))
                pg.draw.rect(self.screen, color, rect, border_radius=5)

        if self.food:
            rect = pg.Rect(self.food[1] * config.GRID_SIZE, self.food[0] * config.GRID_SIZE, config.GRID_SIZE, config.GRID_SIZE)
            pg.draw.rect(self.screen, (255, 80, 80), rect, border_radius=10)

        if self.font:
            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            life_text = self.font.render(f"Life: {int(self.life)}", True, (255, 255, 255))
            self.screen.blit(score_text, (10, 5))
            self.screen.blit(life_text, (10, 35))

        if self.metrics_overlay_ticks > 0 and self.metrics_overlay and self.font:
            overlay_height = len(self.metrics_overlay) * 24 + 16
            overlay_width = 260
            rect = self.pygame.Rect(
                config.WINDOW_SIZE - overlay_width - 10,
                10,
                overlay_width,
                overlay_height,
            )
            surface = pg.Surface((overlay_width, overlay_height))
            surface.set_alpha(200)
            surface.fill((10, 10, 20))
            self.screen.blit(surface, rect.topleft)
            for idx, line in enumerate(self.metrics_overlay):
                text_surf = self.font.render(line, True, (220, 220, 140))
                self.screen.blit(text_surf, (rect.x + 10, rect.y + 8 + idx * 24))
            self.metrics_overlay_ticks -= 1

        if self.metrics_display and self.live_metrics and self.font:
            entries = self._formatted_metrics_lines()
            overlay_width = 300
            overlay_height = len(entries) * 24 + 24
            rect = self.pygame.Rect(10, 80, overlay_width, overlay_height)
            surface = pg.Surface((overlay_width, overlay_height))
            surface.set_alpha(160)
            surface.fill((20, 20, 35))
            self.screen.blit(surface, rect.topleft)
            title = self.font.render("Live metrics", True, (200, 220, 255))
            self.screen.blit(title, (rect.x + 10, rect.y + 5))
            for idx, line in enumerate(entries):
                text_surf = self.font.render(line, True, (210, 210, 220))
                self.screen.blit(text_surf, (rect.x + 10, rect.y + 28 + idx * 24))

        pg.display.flip()
        if self.clock:
            self.clock.tick(config.FPS)

    def _current_menu_items(self) -> list[tuple[str, Callable[[], None]]]:
        if self.menu_state == "main":
            resume_label = "Resume Game" if self.menu_has_started else "Start Game"
            return [
                (resume_label, self._menu_action_resume),
                ("Options", self._menu_action_open_options),
                ("Quit", self._menu_action_quit),
            ]
        return [
            ("Toggle tuning metrics", self._menu_action_toggle_metrics),
            ("Clear Memory", self._menu_action_clear_memory),
            ("Clear Game State", self._menu_action_clear_game_state),
            ("Fresh Start (memory + state)", self._menu_action_clear_all),
            ("Back", self._menu_action_back_to_main),
        ]

    def _menu_action_resume(self) -> None:
        self.menu_visible = False
        self.menu_has_started = True
        self.menu_message = "Resuming game..."

    def _menu_action_open_options(self) -> None:
        self.menu_state = "options"
        self.menu_index = 0
        self.menu_message = "Pick which data to reset."

    def _menu_action_back_to_main(self) -> None:
        self.menu_state = "main"
        self.menu_index = 0
        self.menu_message = "Back to the main menu."

    def _menu_action_clear_memory(self) -> None:
        self._clear_memory_files()
        self.menu_memory_requested = True
        self.menu_message = "Persistent memory cleared; agent will rebuild knowledge."

    def _menu_action_clear_game_state(self) -> None:
        if self._clear_persistent_file(config.GAME_STATE_FILE):
            self.menu_message = "Saved game state deleted."
        else:
            self.menu_message = "No saved state to delete."

    def _menu_action_clear_all(self) -> None:
        self._clear_memory_files()
        self._clear_persistent_file(config.GAME_STATE_FILE)
        self.menu_memory_requested = True
        self.menu_message = "Memory and saved state removed."

    def _menu_action_quit(self) -> None:
        raise KeyboardInterrupt("Quit requested from in-game menu")

    def _menu_action_toggle_metrics(self) -> None:
        self.menu_show_metrics = not self.menu_show_metrics
        status = "showing" if self.menu_show_metrics else "hiding"
        self.menu_message = f"Tuning metrics {status}."

    def _handle_menu_key(self, key: int) -> None:
        if not self.pygame:
            return
        items = self._current_menu_items()
        if not items:
            return
        if key in (self.pygame.K_UP, self.pygame.K_w):
            self.menu_index = (self.menu_index - 1) % len(items)
            return
        if key in (self.pygame.K_DOWN, self.pygame.K_s):
            self.menu_index = (self.menu_index + 1) % len(items)
            return
        if key in (self.pygame.K_RETURN, self.pygame.K_KP_ENTER):
            _, handler = items[self.menu_index]
            handler()

    def _clear_persistent_file(self, path_str: str) -> bool:
        path = Path(path_str)
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError as exc:
            logger.warning("Unable to delete %s: %s", path, exc)
            return False

    def _clear_memory_files(self) -> None:
        for suffix in ("", ".bak", ".tmp"):
            self._clear_persistent_file(config.MEMORY_FILE + suffix)

    def set_metrics_info(self, metrics: Dict[str, Any]) -> None:
        self.metrics_info = dict(metrics)

    def show_metrics_overlay(self, metrics: Dict[str, Any]) -> None:
        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.3f}")
            else:
                lines.append(f"{key}: {value}")
        self.metrics_overlay = lines
        self.metrics_overlay_ticks = int(config.FPS * 2.5)

    def toggle_live_metrics(self) -> None:
        self.metrics_display = not self.metrics_display
        status = "expanded" if self.metrics_display else "hidden"
        self.menu_message = f"Live metrics {status}."

    def update_live_metrics(self, metrics: Dict[str, Any]) -> None:
        if metrics:
            self.live_metrics = dict(metrics)

    def _formatted_metrics_lines(self) -> list[str]:
        lines = []
        for key, value in self.metrics_info.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.3f}")
            else:
                lines.append(f"{key}: {value}")
        return lines

    def handle_pygame_events(self) -> None:
        if not self.render_enabled or self.pygame is None:
            return
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_ESCAPE:
                    if self.menu_visible:
                        self.menu_visible = False
                        self.menu_message = "Menu closed; resuming."
                        continue
                    self.menu_visible = True
                    self.menu_state = "main"
                    self.menu_index = 0
                    self.menu_message = "Menu opened."
                    continue
                if self.menu_visible:
                    self._handle_menu_key(event.key)
                    continue
                if event.key == self.pygame.K_m:
                    self.toggle_live_metrics()

    def render_menu(self) -> None:
        if not self.render_enabled or self.screen is None or self.pygame is None:
            return
        pg = self.pygame
        overlay = pg.Surface((config.WINDOW_SIZE, config.WINDOW_SIZE))
        overlay.set_alpha(230)
        overlay.fill((18, 18, 30))
        self.screen.blit(overlay, (0, 0))
        border = self.pygame.Rect(30, 60, config.WINDOW_SIZE - 60, config.WINDOW_SIZE - 120)
        self.pygame.draw.rect(self.screen, (40, 40, 60), border, border_radius=16, width=2)

        if self.font:
            title = self.font.render("Snake Menu", True, (200, 230, 250))
            title_bg = self.pygame.Rect(40, 50, title.get_width() + 20, 40)
            self.pygame.draw.rect(self.screen, (12, 60, 100), title_bg, border_radius=8)
            self.screen.blit(title, (title_bg.x + 10, title_bg.y + 8))

        items = self._current_menu_items()
        for idx, (label, _) in enumerate(items):
            prefix = "→ " if idx == self.menu_index else "  "
            color = (255, 200, 120) if idx == self.menu_index else (210, 210, 230)
            if idx == self.menu_index:
                highlight = self.pygame.Rect(70, 130 + idx * 34, config.WINDOW_SIZE - 140, 34)
                self.pygame.draw.rect(self.screen, (28, 80, 140), highlight, border_radius=8)
                text_pos = (highlight.x + 10, highlight.y + 4)
            else:
                text_pos = (80, 140 + idx * 32)
            if self.font:
                text_surf = self.font.render(f"{prefix}{label}", True, color)
                self.screen.blit(text_surf, text_pos)

        if self.font:
            msg_surf = self.font.render(self.menu_message, True, (200, 200, 200))
            self.screen.blit(msg_surf, (80, config.WINDOW_SIZE - 70))
            hint = self.font.render("ESC toggles menu, Enter selects.", True, (150, 150, 180))
            self.screen.blit(hint, (80, config.WINDOW_SIZE - 40))
        if self.menu_show_metrics and self.font and self.metrics_info:
            stats = self._formatted_metrics_lines()
            for idx, text in enumerate(stats):
                metric_surf = self.font.render(text, True, (220, 220, 120))
                self.screen.blit(metric_surf, (config.WINDOW_SIZE - 260, 140 + idx * 24))

        pg.display.flip()
        if self.clock:
            self.clock.tick(config.FPS)

    def block_until_menu_closed(self) -> None:
        if not self.render_enabled or self.pygame is None:
            return
        while self.menu_visible:
            self.handle_pygame_events()
            self.render_menu()

    def save_game_state(self) -> None:
        if not config.SAVE_GAME_STATE:
            return

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
