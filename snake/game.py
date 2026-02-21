"""Snake game engine with optional pygame rendering and state persistence."""

from __future__ import annotations

import json
import logging
import math
import random
import subprocess
import sys
import textwrap
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Set, Tuple

from . import config
from .rules import legal_moves, is_legal_move, Move
from .utils import segment_age_ratio

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
        self.metrics_info: Dict[str, Any] = {}
        self.metrics_overlay: Optional[list[str]] = None
        self.metrics_overlay_ticks = 0
        self.summary_tooltip = ""
        self.tooltip_ticks = 0
        self.repo_root = Path(__file__).resolve().parents[1]
        self.runs_dir = self.repo_root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.state_root = self.repo_root / "state"
        self.preset_status: Optional[str] = None
        self.preset_stage_name: Optional[str] = None
        self.preset_start_time: float = 0.0
        self.preset_log_tail: list[str] = []
        self.preset_cancel_requested = False
        self._current_preset_log_path: Optional[Path] = None
        self.preset_seed = 42
        self._preset_queue: list[dict[str, Any]] = []
        self._preset_proc: Optional[subprocess.Popen] = None
        self._preset_log: Optional[Any] = None
        self._preset_active_label: Optional[str] = None
        self.summary_command = [
            sys.executable,
            str(self.repo_root / "scripts" / "compare_runs.py"),
            "--train",
            str(self.runs_dir / "train_preset.jsonl"),
            "--eval",
            str(self.runs_dir / "eval_preset.jsonl"),
        ]
        self.preset_list = [
            (
                "Quick headless 50",
                [{"args": ["--no-render", "--num-games", "50"], "log": "quick50", "state": None}],
            ),
            (
                "Train 300 / Eval 100",
                [
                    {"args": ["--no-render", "--num-games", "300"], "log": "train_preset", "state": "state/train"},
                    {
                        "args": ["--freeze-learning", "--no-render", "--num-games", "100", "--eval"],
                        "log": "eval_preset",
                        "state": "state/train",
                    },
                ],
            ),
            (
                "Frozen eval 100",
                [
                    {
                        "args": ["--freeze-learning", "--no-render", "--num-games", "100", "--eval"],
                        "log": "frozen100",
                        "state": "state/train",
                    }
                ],
            ),
        ]

        self.all_coords: Set[Tuple[int, int]] = {
            (r, c) for r in range(config.BOARD_SIZE) for c in range(config.BOARD_SIZE)
        }
        self.won = False
        self.terminal_reason: Optional[str] = None
        # Optional detail for terminal_reason == "collision".
        # Examples: "wall", "self", "reverse", "tail_entry_eating".
        self.collision_reason: Optional[str] = None
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

    def reset(self) -> None:
        start_pos = (config.BOARD_SIZE // 2, config.BOARD_SIZE // 2)
        self.snake: Deque[Tuple[int, int]] = deque([start_pos])
        self.direction = self.rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        self.occupied_cells = {start_pos}
        self.terminal_reason = None
        self.collision_reason = None
        self.food = self._place_food()
        self.score = 0
        self.steps_since_food = 0
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
            self.terminal_reason = "win"
            self.collision_reason = None
            return None
        # Random placement via RNG.
        return self.rng.choice(sorted(available_positions))

    def _would_eat(self, new_head: Tuple[int, int]) -> bool:
        return self.food is not None and new_head == self.food

    def _check_collision(self, new_head: Tuple[int, int]) -> bool:
        # Reset per-move detail; populated only when we detect an illegal move.
        self.collision_reason = None

        head_r, head_c = new_head
        brd = int(config.BOARD_SIZE)

        # 1) Wall check (fast, independent of other rules)
        if head_r < 0 or head_c < 0 or head_r >= brd or head_c >= brd:
            self.collision_reason = "wall"
            return True

        # 2) Reverse check (engine forbids reversing into yourself when length>1)
        move = (new_head[0] - self.snake[0][0], new_head[1] - self.snake[0][1])
        if len(self.snake) > 1:
            dr, dc = self.direction
            if move == (-dr, -dc) and (dr != 0 or dc != 0):
                self.collision_reason = "reverse"
                return True

        # 3) Self / tail-entry checks. Tail entry is legal only when NOT eating.
        would_eat = self._would_eat(new_head)
        tail = self.snake[-1] if self.snake else None
        if new_head in self.occupied_cells:
            if tail is not None and new_head == tail and not would_eat:
                # Legal tail entry when the tail will move away.
                pass
            else:
                self.collision_reason = "tail_entry_eating" if (tail is not None and new_head == tail and would_eat) else "self"
                return True

        # 4) Fall back to the canonical rules check (keeps us consistent if rules evolve)
        legal = is_legal_move(self.snake, self.food, move, self.direction)
        if not legal and self.collision_reason is None:
            self.collision_reason = "illegal"
        return not legal

    def _update_snake(self, action: Tuple[int, int]) -> Tuple[Tuple[int, int], bool]:

        new_head = (self.snake[0][0] + action[0], self.snake[0][1] + action[1])


        # Apply movement

        self.snake.appendleft(new_head)

        self.direction = action


        ate_food = (self.food is not None and new_head == self.food)


        if ate_food:
            self.score += 1
            self.steps_since_food = 0
            self.life = min(self.life + config.LIFE_PER_FOOD, config.MAX_LIFE)
        else:
            self.snake.pop()
            self.steps_since_food += 1

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
            wall_penalty = config.PENALTY_WALL_HUG
            food_on_border = (
                self.food is not None
                and (
                    self.food[0] in (0, config.BOARD_SIZE - 1)
                    or self.food[1] in (0, config.BOARD_SIZE - 1)
                )
            )
            if food_on_border:
                wall_penalty *= config.WALL_HUG_PENALTY_FOOD_FACTOR
            reward += wall_penalty

        return reward

    def move_snake(self, action: Tuple[int, int], pre_move_state: Dict[str, Any]) -> float:
        if self.game_over:
            return 0.0

        new_head = (self.snake[0][0] + action[0], self.snake[0][1] + action[1])
        if self._check_collision(new_head):
            self.game_over = True
            self.life = 0
            self.terminal_reason = "collision"
            return config.PENALTY_DEATH

        _, ate_food = self._update_snake(action)
        if self.life <= 0:
            self.game_over = True
            if not self.won and not self.terminal_reason:
                self.terminal_reason = "starvation"
            return config.PENALTY_DEATH
        reward = self._compute_reward(ate_food, new_head, pre_move_state)

        # Per-step shaping: discourage endless cycling/stalling
        if not ate_food:
            reward += config.STEP_PENALTY
            if self.steps_since_food >= config.STALL_PENALTY_START:
                reward += config.STALL_PENALTY

        return reward

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
        for i, segment in enumerate(self.snake):
            rect = pg.Rect(segment[1] * config.GRID_SIZE, segment[0] * config.GRID_SIZE, config.GRID_SIZE, config.GRID_SIZE)
            if i == 0:
                pg.draw.rect(self.screen, head_color, rect, border_radius=6)
            else:
                ratio = segment_age_ratio(i, len(self.snake))
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

        if self.font:
            preset_running = self._preset_proc and self._preset_proc.poll() is None
            status_lines: list[str] = []
            if preset_running:
                elapsed = time.time() - self.preset_start_time if self.preset_start_time else 0.0
                stage_label = self.preset_stage_name or "Preset"
                status_lines.append(f"{stage_label} running • {elapsed:.1f}s")
                self._refresh_preset_log_tail()
                status_lines.extend(line for line in self.preset_log_tail[-3:])
                status_lines.append("Press Esc to cancel")
                self.preset_status = f"{stage_label} running"
            elif self.preset_status:
                self._refresh_preset_log_tail()
                status_lines.append(self.preset_status)
                status_lines.extend(line for line in self.preset_log_tail[-3:])
            if status_lines:
                overlay_lines: list[str] = []
                max_overlay_width = config.WINDOW_SIZE - 40
                font_width = self.font.size("W")[0] if self.font else 10
                wrap_chars = max(
                    12,
                    (max_overlay_width - 32) // max(font_width, 1),
                )
                for line in status_lines:
                    wrapped = textwrap.wrap(
                        line,
                        width=wrap_chars,
                        break_long_words=True,
                        replace_whitespace=False,
                    )
                    overlay_lines.extend(wrapped or [""])
                line_width = max(self.font.size(line)[0] for line in overlay_lines)
                overlay_width = min(line_width + 24, max_overlay_width)
                overlay_height = len(overlay_lines) * 22 + 16
                overlay_rect = self.pygame.Rect(
                    (config.WINDOW_SIZE - overlay_width) / 2,
                    20,
                    overlay_width,
                    overlay_height,
                )
                overlay = pg.Surface((overlay_width, overlay_height))
                overlay.set_alpha(220)
                overlay.fill((10, 16, 26))
                self.screen.blit(overlay, overlay_rect.topleft)
                for idx, line in enumerate(overlay_lines):
                    text_surf = self.font.render(line, True, (240, 240, 220))
                    text_rect = text_surf.get_rect()
                    text_rect.midtop = (
                        overlay_rect.x + overlay_width / 2,
                        overlay_rect.y + 8 + idx * 22,
                    )
                    self.screen.blit(text_surf, text_rect.topleft)

        if self.tooltip_ticks > 0 and self.summary_tooltip and self.font:
            tooltip_lines: list[str] = []
            font_width = self.font.size("W")[0] if self.font else 10
            max_chars = max(32, (config.WINDOW_SIZE - 40) // max(font_width, 1))
            for line in self.summary_tooltip.splitlines():
                wrapped = textwrap.wrap(
                    line,
                    width=max_chars,
                    break_long_words=True,
                    replace_whitespace=False,
                )
                tooltip_lines.extend(wrapped or [""])
            if not tooltip_lines:
                tooltip_lines.append("")
            max_width = 0
            for line in tooltip_lines:
                max_width = max(max_width, self.font.size(line)[0])
            tooltip_width = max_width + 22
            tooltip_height = len(tooltip_lines) * 22 + 16
            tooltip_rect = self.pygame.Rect(
                10,
                config.WINDOW_SIZE - tooltip_height - 10,
                tooltip_width,
                tooltip_height,
            )
            tooltip_surface = pg.Surface((tooltip_width, tooltip_height))
            tooltip_surface.set_alpha(210)
            tooltip_surface.fill((12, 12, 20))
            self.screen.blit(tooltip_surface, tooltip_rect.topleft)
            for idx, line in enumerate(tooltip_lines):
                text_surf = self.font.render(line, True, (220, 220, 255))
                self.screen.blit(text_surf, (tooltip_rect.x + 10, tooltip_rect.y + 8 + idx * 22))
            self.tooltip_ticks -= 1

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
        if self.menu_state == "options":
            return [
                ("Toggle tuning metrics", self._menu_action_toggle_metrics),
                ("Presets", self._menu_action_open_presets),
                ("Toggle live metrics overlay", self.toggle_live_metrics),
                ("Benchmark summary", self._menu_action_show_summary),
                ("Clear Memory", self._menu_action_clear_memory),
                ("Clear Game State", self._menu_action_clear_game_state),
                ("Fresh Start (memory + state)", self._menu_action_clear_all),
                ("Back", self._menu_action_back_to_main),
            ]
        if self.menu_state == "presets":
            entries = [
                (label, lambda idx=idx: self._menu_action_launch_preset(idx))
                for idx, (label, _) in enumerate(self.preset_list)
            ]
            entries.append(("Back", self._menu_action_back_to_options))
            return entries
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

    def _menu_action_open_presets(self) -> None:
        self.menu_state = "presets"
        self.menu_index = 0
        self.menu_message = "Select a benchmark preset."

    def _menu_action_back_to_options(self) -> None:
        self.menu_state = "options"
        self.menu_index = 0
        self.menu_message = "Back to options."

    def _menu_action_launch_preset(self, idx: int) -> None:
        if self._preset_proc and self._preset_proc.poll() is None:
            self.menu_message = "A preset is already running."
            return

        label, stages = self.preset_list[idx]
        self._preset_queue = list(stages)
        self._preset_active_label = label
        self.menu_message = f"Starting preset '{label}'..."
        self.preset_status = f"Queued '{label}'"
        self._start_next_preset_stage()

    def _menu_action_show_summary(self) -> None:
        self.menu_message = "Displaying benchmark summary..."
        train = self.runs_dir / "train_preset.jsonl"
        eval_ = self.runs_dir / "eval_preset.jsonl"
        try:
            res = subprocess.run(
                self.summary_command,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            lines = [line for line in out.splitlines() if line.strip()]
            if not lines:
                lines = ["(no output)"]
            self.metrics_overlay = lines[-25:]
            self.metrics_overlay_ticks = int(config.FPS * 6.0)
            self.menu_message = "Benchmark summary displayed."
            self.summary_tooltip = f"Train summary: {train}\nEval summary: {eval_}"
            self.tooltip_ticks = int(config.FPS * 6.0)
            summary_path = self.runs_dir / "latest_summary.txt"
            summary_path.write_text("\n".join(lines), encoding="utf-8")
            self.menu_message += f" (saved to {summary_path.name})"
        except Exception as exc:
            self.menu_message = f"Benchmark summary failed: {exc}"
            self.summary_tooltip = f"Summary failed: {exc}"
            self.tooltip_ticks = int(config.FPS * 4.0)

    def _start_next_preset_stage(self) -> None:
        if not self._preset_queue:
            self._preset_proc = None
            if self._preset_log:
                self._preset_log.close()
                self._preset_log = None
            self.menu_message = f"Preset '{self._preset_active_label}' completed."
            self._preset_active_label = None
            self.preset_status = "Preset completed"
            self.preset_stage_name = None
            self.preset_start_time = 0.0
            self._current_preset_log_path = None
            return

        stage = self._preset_queue.pop(0)
        cmd, log_path = self._build_preset_command(stage)
        if self._preset_log:
            self._preset_log.close()
        self._preset_log = log_path.open("a", encoding="utf-8")
        self._preset_log.write(f"\n\n=== running: {' '.join(cmd)} ===\n")
        self._preset_proc = subprocess.Popen(
            cmd,
            cwd=str(self.repo_root),
            stdout=self._preset_log,
            stderr=subprocess.STDOUT,
        )
        self.menu_message = f"Running preset stage -> {log_path.name}"
        self.preset_status = f"{self._preset_active_label}: {stage.get('log', 'stage')}"
        self.preset_stage_name = stage.get("log", "stage")
        self.preset_start_time = time.time()
        self.preset_log_tail = []
        self._current_preset_log_path = log_path
        self.preset_cancel_requested = False
        self._refresh_preset_log_tail()
        self._current_preset_log_path = log_path
        self.preset_cancel_requested = False

    def _build_preset_command(self, stage: dict[str, Any]) -> tuple[list[str], Path]:
        cmd = [sys.executable, "-m", "snake"]
        cmd.extend(stage.get("args", []))
        if "--seed" not in cmd:
            cmd.extend(["--seed", str(self.preset_seed)])

        state_rel = stage.get("state")
        if state_rel:
            state_path = self.repo_root / state_rel
            state_path.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--state-dir", str(state_path)])

        log_path = self.runs_dir / f"{stage.get('log', 'preset')}.jsonl"
        cmd.extend(["--log-jsonl", str(log_path)])
        return cmd, self.runs_dir / f"preset_{stage.get('log', 'preset')}.log"

    def _clean_preset_log_lines(self, lines: list[str]) -> list[str]:
        cleaned: list[str] = []
        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue
            if " - " in trimmed:
                parts = trimmed.split(" - ")
                trimmed = parts[-1].strip()
            cleaned.append(trimmed)
        return cleaned


    def _refresh_preset_log_tail(self) -> None:
        path = self._current_preset_log_path
        if not path or not path.exists():
            self.preset_log_tail = []
            return
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                lines = [line.rstrip() for line in f.readlines()]
            self.preset_log_tail = self._clean_preset_log_lines(lines[-4:])
        except OSError:
            self.preset_log_tail = []

    def _pump_preset_runner(self) -> None:
        if not self._preset_proc:
            return
        if self._preset_proc.poll() is None:
            return
        returncode = self._preset_proc.returncode
        if returncode != 0:
            if self._preset_log:
                self._preset_log.write(f"\nPreset stage exited with {returncode}\n")
                self._preset_log.close()
                self._preset_log = None
            self._preset_proc = None
            self._preset_queue.clear()
            self.menu_message = (
                f"Preset '{self._preset_active_label}' failed (exit {returncode}); see log."
            )
            self._preset_active_label = None
            self.preset_status = "Preset failed"
            return
        self._start_next_preset_stage()

    def _cancel_preset(self) -> None:
        if not self._preset_proc or self._preset_proc.poll() is not None:
            return
        self.preset_cancel_requested = True
        self.preset_status = "Cancelling preset..."
        self.menu_message = "Cancelling preset..."
        try:
            self._preset_proc.terminate()
        except Exception:
            pass

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
        self._pump_preset_runner()
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_ESCAPE:
                    if self.menu_visible:
                        self.menu_visible = False
                        self.menu_message = "Menu closed; resuming."
                        continue
                    if self._preset_proc and self._preset_proc.poll() is None:
                        self._cancel_preset()
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
        overlay.set_alpha(240)
        overlay.fill((12, 12, 20))
        self.screen.blit(overlay, (0, 0))
        border = self.pygame.Rect(20, 40, config.WINDOW_SIZE - 40, config.WINDOW_SIZE - 80)
        self.pygame.draw.rect(self.screen, (30, 30, 50), border, border_radius=18, width=2)

        if self.font:
            title = self.font.render("Snake Menu", True, (200, 230, 250))
            title_bg = self.pygame.Rect(40, 50, title.get_width() + 20, 40)
            self.pygame.draw.rect(self.screen, (12, 60, 100), title_bg, border_radius=8)
            self.screen.blit(title, (title_bg.x + 10, title_bg.y + 8))

        items = self._current_menu_items()
        for idx, (label, _) in enumerate(items):
            prefix = "→ " if idx == self.menu_index else "  "
            color = (255, 210, 150) if idx == self.menu_index else (230, 230, 240)
            y_offset = 160 + idx * 42
            if idx == self.menu_index:
                highlight = self.pygame.Rect(60, y_offset - 8, config.WINDOW_SIZE - 120, 40)
                self.pygame.draw.rect(self.screen, (32, 90, 150), highlight, border_radius=10)
                text_pos = (highlight.x + 12, highlight.y + 6)
            else:
                text_pos = (80, y_offset)
            if self.font:
                text_surf = self.font.render(f"{prefix}{label}", True, color)
                self.screen.blit(text_surf, text_pos)

        if self.font:
            msg_surf = self.font.render(self.menu_message, True, (220, 220, 220))
            self.screen.blit(msg_surf, (60, config.WINDOW_SIZE - 90))
            hint = self.font.render("ESC=menu  Enter=select  M=metrics presets", True, (180, 180, 200))
            self.screen.blit(hint, (60, config.WINDOW_SIZE - 50))
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
