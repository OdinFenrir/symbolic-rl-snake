"""Autonomous Snake agent."""

from __future__ import annotations

import heapq
import logging
import random
from collections import deque
from functools import lru_cache
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from . import config
from .memory import SymbolicMemory

logger = logging.getLogger(__name__)


class SnakeAgent:
    """Rule-based agent with path guidance and persistent experience memory."""

    def __init__(self) -> None:
        self.actions = {(0, -1): "left", (0, 1): "right", (1, 0): "down", (-1, 0): "up"}
        self.symbolic_memory = SymbolicMemory()

        # Metrics (useful for diagnostics)
        self.total_rsm_depth = 0
        self.decision_count = 0

    @lru_cache(maxsize=1024)
    def _count_reachable_space(
        self,
        snake_body_tuple: Tuple[Tuple[int, int], ...],
        start_pos: Tuple[int, int],
    ) -> int:
        r0, c0 = start_pos
        if not (0 <= r0 < config.BOARD_SIZE and 0 <= c0 < config.BOARD_SIZE):
            return 0

        blocked = set(snake_body_tuple)
        if start_pos in blocked:
            return 0

        q: deque[Tuple[int, int]] = deque([start_pos])
        seen = {start_pos}
        count = 0

        while q:
            r, c = q.popleft()
            count += 1
            for dr, dc in self.actions.keys():
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if (
                    0 <= nr < config.BOARD_SIZE
                    and 0 <= nc < config.BOARD_SIZE
                    and nxt not in blocked
                    and nxt not in seen
                ):
                    seen.add(nxt)
                    q.append(nxt)

        return count

    def _a_star_pathfinding(
        self,
        start: Tuple[int, int],
        goal: Optional[Tuple[int, int]],
        obstacles: Set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        if goal is None:
            return []

        def h(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set: List[Tuple[int, Tuple[int, int]]] = [(0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path: List[Tuple[int, int]] = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dr, dc in self.actions.keys():
                neighbor = (current[0] + dr, current[1] + dc)

                if not (0 <= neighbor[0] < config.BOARD_SIZE and 0 <= neighbor[1] < config.BOARD_SIZE):
                    continue
                if neighbor in obstacles:
                    continue

                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, 10**9):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g + h(neighbor, goal), neighbor))

        return []

    def _get_heuristic_scores(
        self,
        snake: Deque[Tuple[int, int]],
        safe_moves: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], float]:
        scores: Dict[Tuple[int, int], float] = {}
        snake_body_tuple = tuple(snake)
        tail_segments = list(snake)[-5:]

        for move in safe_moves:
            score = 0.0
            next_head = (snake[0][0] + move[0], snake[0][1] + move[1])

            # Reachable space
            space = float(self._count_reachable_space(snake_body_tuple, next_head))
            score += space

            # Dead-end detection (1 step ahead)
            temp_snake = deque([next_head])
            temp_snake.extend(list(snake)[:-1])
            future_safe = 0
            for dr, dc in self.actions.keys():
                future_head = (next_head[0] + dr, next_head[1] + dc)
                if (
                    0 <= future_head[0] < config.BOARD_SIZE
                    and 0 <= future_head[1] < config.BOARD_SIZE
                    and future_head not in temp_snake
                ):
                    future_safe += 1
            if future_safe <= 1:
                score *= 0.1

            # Tail proximity (discourage squeezing into tight spaces)
            min_dist_to_tail = min(
                (abs(next_head[0] - sx) + abs(next_head[1] - sy) for sx, sy in tail_segments),
                default=100,
            )
            if min_dist_to_tail <= 2 and space < len(snake) * 2:
                score += config.PENALTY_TAIL_PROXIMITY

            scores[move] = score

        return scores

    def _combine_scores(
        self,
        heuristic_scores: Dict[Tuple[int, int], float],
        rsm_scores: Dict[Tuple[int, int], float],
        a_star_path: List[Tuple[int, int]],
        safe_moves: List[Tuple[int, int]],
        snake_head: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Dict[Tuple[int, int], float]]:
        final_scores: Dict[Tuple[int, int], float] = {}
        max_h = max(heuristic_scores.values()) if heuristic_scores else 1.0

        a_star_move = None
        if a_star_path:
            dr = a_star_path[0][0] - snake_head[0]
            dc = a_star_path[0][1] - snake_head[1]
            if (dr, dc) in self.actions:
                a_star_move = (dr, dc)

        for move in safe_moves:
            h_score = (heuristic_scores.get(move, 0.0) / max_h if max_h > 0 else 0.0) * config.HEURISTIC_WEIGHT
            r_score = float(rsm_scores.get(move, 0.0)) * config.RSM_WEIGHT
            bonus = config.A_STAR_BONUS if (a_star_move and move == a_star_move) else 0.0
            final_scores[move] = h_score + r_score + bonus

        best = max(final_scores, key=final_scores.get)
        return best, final_scores

    def choose_action(
        self,
        snake: Deque[Tuple[int, int]],
        food: Optional[Tuple[int, int]],
        direction: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Dict[str, Any]]:
        self.decision_count += 1
        pre_state = self.symbolic_memory.create_symbolic_state(snake, food, direction)
        safe_moves: List[Tuple[int, int]] = pre_state["safe_moves"]
        debug_info: Dict[str, Any] = {}

        if not safe_moves:
            return random.choice(list(self.actions.keys())), debug_info
        if len(safe_moves) == 1:
            return safe_moves[0], debug_info

        heuristic_scores = self._get_heuristic_scores(snake, safe_moves)

        rsm_depth = max(config.RSM_MIN_DEPTH, min(config.RSM_MAX_DEPTH, len(snake) // 4))
        self.total_rsm_depth += rsm_depth
        rsm_scores = self.symbolic_memory.recursive_reasoning(pre_state, depth=rsm_depth)

        a_star_path = self._a_star_pathfinding(snake[0], food, set(snake))
        debug_info["a_star_path"] = a_star_path

        best, final_scores = self._combine_scores(
            heuristic_scores, rsm_scores, a_star_path, safe_moves, snake[0]
        )
        debug_info["final_scores"] = final_scores
        return best, debug_info
