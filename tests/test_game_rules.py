from __future__ import annotations

import unittest
from collections import deque

from snake import config
from snake.game import SnakeGame


def _make_pre_move_state(food, head):
    if food is None:
        distance_sq = float("inf")
    else:
        dx = head[0] - food[0]
        dy = head[1] - food[1]
        distance_sq = dx * dx + dy * dy
    return {"food": food, "distance_sq": distance_sq}


class TailRuleBehaviorTest(unittest.TestCase):
    def test_tail_entry_allowed_when_not_eating(self):
        game = SnakeGame(render_enabled=False, seed=0)
        game.snake = deque([(1, 0), (0, 0)])
        game.direction = (0, 1)
        game.food = (5, 5)

        pre_move = _make_pre_move_state(game.food, game.snake[0])
        reward = game.move_snake((-1, 0), pre_move)

        self.assertNotEqual(reward, config.PENALTY_DEATH)
        self.assertFalse(game.game_over)

    def test_tail_entry_rejected_when_eating(self):
        game = SnakeGame(render_enabled=False, seed=0)
        game.snake = deque([(1, 0), (0, 0)])
        game.direction = (0, 1)
        game.food = (0, 0)

        pre_move = _make_pre_move_state(game.food, game.snake[0])
        reward = game.move_snake((-1, 0), pre_move)

        self.assertEqual(reward, config.PENALTY_DEATH)
        self.assertTrue(game.game_over)
