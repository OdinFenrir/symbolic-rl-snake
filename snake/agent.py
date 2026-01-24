"""Autonomous Snake agent with trap-avoidance, anti-pocketing, and instrumentation."""

from __future__ import annotations

import heapq
import logging
import random
from collections import deque
from functools import lru_cache
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from . import config
from .memory import SymbolicMemory
from .rules import legal_moves

logger = logging.getLogger(__name__)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class AdaptiveTuner:
    """Keep a short history of safety/score metrics and gently rebalance heuristics."""

    def __init__(self) -> None:
        self.forced_history: Deque[float] = deque(maxlen=config.ADAPTIVE_HISTORY_SIZE)
        self.score_history: Deque[float] = deque(maxlen=config.ADAPTIVE_HISTORY_SIZE)
        self.safety_bias = 0.0
        self.reward_bias = 0.0
        self.best_score = 0

    def penalty_scale(self) -> float:
        return max(0.2, 1.0 + self.safety_bias)

    def reward_scale(self) -> float:
        return max(0.2, 1.0 + self.reward_bias)

    def observe_episode(self, score: int, forced_rate: float) -> None:
        forced = max(0.0, forced_rate)
        self.forced_history.append(forced)
        self.score_history.append(score)

        if not self.forced_history:
            return

        avg_forced = sum(self.forced_history) / len(self.forced_history)
        delta_forced = avg_forced - config.ADAPTIVE_TARGET_FORCED_RATE
        self.safety_bias = _clamp(
            self.safety_bias + delta_forced * config.ADAPTIVE_SAFETY_ADJUST_SCALE,
            -config.ADAPTIVE_SAFETY_BIAS_CAP,
            config.ADAPTIVE_SAFETY_BIAS_CAP,
        )
        self.reward_bias = _clamp(
            self.reward_bias - delta_forced * config.ADAPTIVE_REWARD_ADJUST_SCALE,
            -config.ADAPTIVE_REWARD_BIAS_CAP,
            config.ADAPTIVE_REWARD_BIAS_CAP,
        )

        if score > self.best_score:
            self.best_score = score
            self.reward_bias = _clamp(
                self.reward_bias + config.ADAPTIVE_SCORE_BOOST,
                -config.ADAPTIVE_REWARD_BIAS_CAP,
                config.ADAPTIVE_REWARD_BIAS_CAP,
            )
        else:
            self.reward_bias = _clamp(
                self.reward_bias - config.ADAPTIVE_SCORE_DECAY,
                -config.ADAPTIVE_REWARD_BIAS_CAP,
                config.ADAPTIVE_REWARD_BIAS_CAP,
            )

Move = Tuple[int, int]
Cell = Tuple[int, int]


class SnakeAgent:
    """Rule-based agent with path guidance and persistent experience memory.

    Key behaviors:
    - Avoid hard traps (especially on the step you eat, when tail doesn't move).
    - Reduce "pocketing" by preferring moves that do not increase unreachable open space.
    - Instrumentation counters per episode: rejects/forced.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.actions: Dict[Move, str] = {
            (0, -1): "left",
            (0, 1): "right",
            (1, 0): "down",
            (-1, 0): "up",
        }
        self.rng = random.Random(seed)
        self.symbolic_memory = SymbolicMemory()

        self.total_rsm_depth = 0
        self.decision_count = 0

        # Safety tuning (local; no config churn)
        self.escape_fail_penalty = -1e6

        # Pocket policy
        self.pocket_increase_allow_short = 2   # allow small increases early
        self.pocket_increase_allow_long = 1    # be stricter later
        self.long_snake_len = 16               # threshold
        self.pocket_penalty = 3.0              # penalty per unreachable cell (pocket size)
        self.pocket_delta_penalty = 8.0        # extra penalty per NEW unreachable cell
        self.eat_reach_slack = 2            # require reachable >= len(snake)+slack on eat step

        self._board_cells = int(config.BOARD_SIZE) * int(config.BOARD_SIZE)

        # Safety instrumentation
        self.safety_rejects = 0
        self.safety_forced = 0
        self._ep_safety_rejects = 0
        self._ep_safety_forced = 0

        # Short-term loop breaking (local-only; resets each episode)
        self._recent_heads: Deque[Cell] = deque(maxlen=32)
        self._recent_moves: Deque[Move] = deque(maxlen=16)

        # Adaptive heuristics tuner
        self.tuner = AdaptiveTuner()

    def begin_episode(self) -> None:
        """Reset per-episode instrumentation counters."""
        self._ep_safety_rejects = 0
        self._ep_safety_forced = 0

        self._recent_heads.clear()
        self._recent_moves.clear()

    def end_episode_stats(self) -> dict:
        """Return per-episode instrumentation stats."""
        return {
            "safety_rejects": int(self._ep_safety_rejects),
            "safety_forced": int(self._ep_safety_forced),
        }

    def record_episode_stats(self, score: int, steps: int) -> dict:
        stats = self.end_episode_stats()
        forced = stats.get("safety_forced", 0)
        forced_rate = float(forced) / max(1, steps) if steps else 0.0
        stats.update(
            {
                "score": score,
                "steps": steps,
                "forced_rate": forced_rate,
            }
        )
        self.tuner.observe_episode(score, forced_rate)
        return stats

    # ----------------------------
    # Utilities
    # ----------------------------
    def _in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < config.BOARD_SIZE and 0 <= c < config.BOARD_SIZE

    def _neighbors4(self, cell: Cell) -> List[Cell]:
        r, c = cell
        return [(r + dr, c + dc) for dr, dc in self.actions.keys()]

    def _simulate_step(self, snake: Deque[Cell], move: Move, food: Optional[Cell]) -> Tuple[Deque[Cell], bool]:
        head = snake[0]
        new_head = (head[0] + move[0], head[1] + move[1])
        ate = (food is not None and new_head == food)

        new_snake: Deque[Cell] = deque([new_head])
        new_snake.extend(snake)

        if not ate:
            new_snake.pop()  # tail moves only if not eating
        return new_snake, ate

    def _bfs_reachable_count(self, start: Cell, blocked: Set[Cell]) -> int:
        if start in blocked or not self._in_bounds(start):
            return 0
        q = deque([start])
        seen = {start}
        count = 0
        while q:
            cur = q.popleft()
            count += 1
            for nxt in self._neighbors4(cur):
                if nxt in seen or nxt in blocked or not self._in_bounds(nxt):
                    continue
                seen.add(nxt)
                q.append(nxt)
        return count

    def _bfs_path_exists(self, start: Cell, goal: Cell, blocked: Set[Cell]) -> bool:
        if start == goal:
            return True
        if start in blocked or not self._in_bounds(start):
            return False
        q = deque([start])
        seen = {start}
        while q:
            cur = q.popleft()
            for nxt in self._neighbors4(cur):
                if not self._in_bounds(nxt) or nxt in blocked or nxt in seen:
                    continue
                if nxt == goal:
                    return True
                seen.add(nxt)
                q.append(nxt)
        return False

    def _escape_metrics(self, snake_after: Deque[Cell], ate_food: bool) -> Tuple[int, int, bool, int]:
        """Return (reachable, total_open, tail_reachable, pocket_size).

        - If ate_food, tail did NOT move this tick -> do NOT treat tail as free.
        - If not ate_food, we treat tail as potentially freeing space next tick (heuristic).
        """
        head = snake_after[0]
        tail = snake_after[-1]

        blocked = set(snake_after)
        blocked.discard(head)  # allow BFS start
        if not ate_food:
            blocked.discard(tail)

        reachable = self._bfs_reachable_count(head, blocked)
        total_open = self._board_cells - len(blocked)

        pocket = total_open - reachable
        if pocket < 0:
            pocket = 0

        tail_reachable = True
        if not ate_food:
            tail_reachable = self._bfs_path_exists(head, tail, blocked)

        return reachable, total_open, tail_reachable, pocket

    def _safe_moves_with_tail_rule(
        self, snake: Deque[Cell], food: Optional[Cell], direction: Move
    ) -> List[Move]:
        return legal_moves(snake, food, direction, forbid_reverse=True)

    # ----------------------------
    # Existing scoring helpers
    # ----------------------------
    @lru_cache(maxsize=1024)
    def _count_reachable_space(self, snake_body_tuple: Tuple[Cell, ...], start_pos: Cell) -> int:
        r0, c0 = start_pos
        if not (0 <= r0 < config.BOARD_SIZE and 0 <= c0 < config.BOARD_SIZE):
            return 0

        blocked = set(snake_body_tuple)
        if start_pos in blocked:
            return 0

        q: deque[Cell] = deque([start_pos])
        seen = {start_pos}
        count = 0
        while q:
            r, c = q.popleft()
            count += 1
            for dr, dc in self.actions.keys():
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if 0 <= nr < config.BOARD_SIZE and 0 <= nc < config.BOARD_SIZE and nxt not in blocked and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return count

    def _a_star_pathfinding(self, start: Cell, goal: Optional[Cell], obstacles: Set[Cell]) -> List[Cell]:
        if goal is None:
            return []

        def h(a: Cell, b: Cell) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set: List[Tuple[int, Cell]] = [(0, start)]
        came_from: Dict[Cell, Cell] = {}
        g_score: Dict[Cell, int] = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path: List[Cell] = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dr, dc in self.actions.keys():
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._in_bounds(neighbor):
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
        snake: Deque[Cell],
        safe_moves: List[Move],
        food: Optional[Cell],
        life: Optional[float],
    ) -> Dict[Move, float]:
        scores: Dict[Move, float] = {}
        ok_map: Dict[Move, bool] = {}

        # Baseline pocket size from the current position (connectivity reference).
        _, _, _, baseline_pocket = self._escape_metrics(snake, ate_food=False)

        rejected_this_step = 0
        ok_any = False

        # Treat tail as likely free for coarse space estimate.
        snake_body_tuple = tuple(list(snake)[:-1]) if len(snake) > 1 else tuple(snake)

        # Allow small pocket increases early; be stricter later.
        allow = self.pocket_increase_allow_long if len(snake) >= self.long_snake_len else self.pocket_increase_allow_short

        penalty_scale = self.tuner.penalty_scale()
        reward_scale = self.tuner.reward_scale()

        # Starvation urgency: when life is low, bias slightly toward food.
        urgency = 0.0
        if life is not None:
            life_f = float(life)
            thresh = 0.35 * float(config.MAX_LIFE)
            if thresh > 0:
                urgency = max(0.0, min(1.0, (thresh - life_f) / thresh))

        base_food_dist = 0
        if food is not None:
            base_food_dist = abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1])

        for move in safe_moves:
            next_head = (snake[0][0] + move[0], snake[0][1] + move[1])

            # Coarse open space score.
            coarse_space = float(self._count_reachable_space(snake_body_tuple, next_head))
            score = coarse_space * reward_scale

            # Gentle urgency-to-food shaping when life is low.
            if food is not None and urgency > 0.0:
                d_after = abs(next_head[0] - food[0]) + abs(next_head[1] - food[1])
                score += reward_scale * 4.0 * urgency * float(base_food_dist - d_after)

            # Simulate and evaluate topology.
            sim_snake, ate = self._simulate_step(snake, move, food)
            reachable, _total_open, tail_ok, pocket = self._escape_metrics(sim_snake, ate_food=ate)
            pocket_delta_raw = pocket - baseline_pocket

            # Safety gate:
            # - Eating is high risk: require sufficient reachable space.
            # - Non-eat moves: allow temporary disconnection if head component remains comfortably large,
            #   but avoid creating large new pockets while disconnected.
            if ate:
                ok = (reachable >= (len(sim_snake) + int(self.eat_reach_slack)))
            else:
                ok = bool(tail_ok) or (reachable >= (len(sim_snake) + 2))

                if (not tail_ok) and (pocket_delta_raw > 0):
                    # Dynamic cutoff: be slightly more tolerant when short / when starving.
                    cutoff = 7 + (2 if len(sim_snake) < self.long_snake_len else 0) + int(round(urgency * 3.0))
                    if pocket_delta_raw > cutoff:
                        ok = False

            ok_map[move] = ok
            if not ok:
                rejected_this_step += 1
            else:
                ok_any = True

            # SOFT pocket control.
            pocket_delta = pocket_delta_raw
            if pocket_delta < 0:
                # Reward "healing" pockets.
                score += 2.0 * float(-pocket_delta)
                pocket_delta = 0

            # Allow small increases without harsh delta penalty.
            pocket_delta_eff = max(0, int(pocket_delta) - int(allow))

            score -= self.pocket_penalty * penalty_scale * float(pocket)
            score -= self.pocket_delta_penalty * penalty_scale * float(pocket_delta_eff) * (2.0 if ate else 1.0)

            if ok and pocket == 0:
                score += 2.0

            # Avoid low-degree positions (future mobility).
            temp_snake = deque(sim_snake)
            future_safe = 0
            for dr, dc in self.actions.keys():
                fh = (next_head[0] + dr, next_head[1] + dc)
                if self._in_bounds(fh) and fh not in temp_snake:
                    future_safe += 1
            if future_safe <= 1:
                score *= 0.1

            scores[move] = score

        # Only hard-penalize unsafe moves if there exists at least one OK alternative.
        # If everything is unsafe, keep relative ordering so we pick the least-bad move.
        if ok_any:
            for mv, ok in ok_map.items():
                if not ok:
                    scores[mv] += self.escape_fail_penalty * penalty_scale

        # Instrumentation aggregation (per decision).
        if rejected_this_step:
            self._ep_safety_rejects += rejected_this_step
            self.safety_rejects += rejected_this_step
        if not ok_any:
            self._ep_safety_forced += 1
            self.safety_forced += 1

        return scores

    def _cycle_penalty(self, move: Move, snake_head: Cell, direction: Move, life: Optional[float]) -> float:
        """Soft penalty to discourage tight cycles / oscillations.

        This does not veto moves; it only nudges tie-breaks and reduces dithering.
        When life is low, the penalty is reduced so the agent can prioritize reaching food.
        """
        new_head = (snake_head[0] + move[0], snake_head[1] + move[1])

        penalty = 0.0

        # Penalize revisiting recent head positions (cycle breaking).
        if new_head in self._recent_heads:
            recent = list(self._recent_heads)
            idx = recent.index(new_head)  # 0 = most recent
            penalty += 0.35 * (1.0 - min(1.0, float(idx) / 10.0))

        # Mild penalty for U-turns (often correlates with oscillation).
        if move == (-direction[0], -direction[1]):
            penalty += 0.10

        # Scale down penalties when starving.
        if life is not None:
            thresh = 0.30 * float(config.MAX_LIFE)
            if thresh > 0:
                urgency = max(0.0, min(1.0, (thresh - float(life)) / thresh))
                penalty *= (1.0 - 0.85 * urgency)

        return penalty


    def _combine_scores(
        self,
        heuristic_scores: Dict[Move, float],
        rsm_scores: Dict[Move, float],
        a_star_path: List[Cell],
        safe_moves: List[Move],
        snake_head: Cell,
    ) -> Tuple[Move, Dict[Move, float]]:
        final_scores: Dict[Move, float] = {}

        h_min = min(heuristic_scores.values()) if heuristic_scores else 0.0
        h_max = max(heuristic_scores.values()) if heuristic_scores else 0.0
        denom = (h_max - h_min)

        a_star_move: Optional[Move] = None
        if a_star_path:
            dr = a_star_path[0][0] - snake_head[0]
            dc = a_star_path[0][1] - snake_head[1]
            if (dr, dc) in self.actions:
                a_star_move = (dr, dc)

        for move in safe_moves:
            # Normalize heuristics robustly even if all values are negative.
            h_raw = float(heuristic_scores.get(move, h_min))
            if denom > 1e-9:
                h_norm = (h_raw - float(h_min)) / float(denom)
            else:
                h_norm = 0.0
            h_score = h_norm * float(config.HEURISTIC_WEIGHT)

            r_score = float(rsm_scores.get(move, 0.0)) * float(config.RSM_WEIGHT)
            bonus = float(config.A_STAR_BONUS) * h_norm if (a_star_move and move == a_star_move) else 0.0
            final_scores[move] = h_score + r_score + bonus

        best_score = max(final_scores.values())
        best_moves = [m for m, s in final_scores.items() if s == best_score]
        best = self.rng.choice(best_moves)
        return best, final_scores

    # ----------------------------
    # Main decision
    # ----------------------------
    def choose_action(self, snake: Deque[Cell], food: Optional[Cell], direction: Move, life: Optional[float] = None) -> Tuple[Move, Dict[str, Any]]:
        self.decision_count += 1

        # Use tail-rule safe moves (more accurate than "new_head in snake" checks).
        safe_moves = self._safe_moves_with_tail_rule(snake, food, direction)

        # Track recent head positions for loop-breaking (soft).
        if not self._recent_heads or self._recent_heads[0] != snake[0]:
            self._recent_heads.appendleft(snake[0])

        # Build symbolic state for memory/RSM, but override safe_moves to match our real rule.
        pre_state = self.symbolic_memory.create_symbolic_state(snake, food, direction)
        pre_state["safe_moves"] = safe_moves

        debug_info: Dict[str, Any] = {}
        if not safe_moves:
            return self.rng.choice(list(self.actions.keys())), debug_info
        if len(safe_moves) == 1:
            return safe_moves[0], debug_info

        heuristic_scores = self._get_heuristic_scores(snake, safe_moves, food, life)

        rsm_depth = max(config.RSM_MIN_DEPTH, min(config.RSM_MAX_DEPTH, len(snake) // 4))
        self.total_rsm_depth += rsm_depth
        rsm_scores = self.symbolic_memory.recursive_reasoning(pre_state, depth=rsm_depth)

        obstacles = set(snake)
        if len(snake) > 1:
            obstacles.discard(snake[-1])  # allow routes through moving tail (planning heuristic)
        a_star_path = self._a_star_pathfinding(snake[0], food, obstacles)
        debug_info["a_star_path"] = a_star_path

        best, final_scores = self._combine_scores(heuristic_scores, rsm_scores, a_star_path, safe_moves, snake[0])

        # Apply soft loop/oscillation penalties and re-select.
        for mv in list(final_scores.keys()):
            final_scores[mv] -= self._cycle_penalty(mv, snake[0], direction, life)

        best_score = max(final_scores.values())
        best_moves = [m for m, s in final_scores.items() if s == best_score]
        best = self.rng.choice(best_moves)

        # Update short-term history with the chosen transition.
        self._recent_moves.appendleft(best)
        self._recent_heads.appendleft((snake[0][0] + best[0], snake[0][1] + best[1]))

        debug_info["final_scores"] = final_scores
        return best, debug_info
