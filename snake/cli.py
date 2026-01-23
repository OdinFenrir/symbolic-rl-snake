"""Command-line interface and run loop."""

from __future__ import annotations

import argparse
import cProfile
import logging
import pstats
import sys
from typing import Optional

from . import config
from .agent import SnakeAgent
from .game import SnakeGame

logger = logging.getLogger(__name__)

try:
    import pygame  # type: ignore
except Exception:
    pygame = None  # type: ignore


def run(num_games: int, render: bool, load_state: bool, debug: bool, seed: Optional[int], max_steps: Optional[int]) -> int:
    config.UI_DEBUG_MODE = bool(debug)
    if max_steps is not None:
        config.MAX_STEPS_PER_GAME = int(max_steps)

    config.validate_config()

    agent = SnakeAgent()
    game = SnakeGame(render_enabled=render, seed=seed)

    if load_state:
        if game.load_game_state():
            logger.info("Resumed from %s", config.GAME_STATE_FILE)
        else:
            logger.info("No saved game state found; starting new")

    scores = []
    total_steps = 0

    try:
        for i in range(num_games):
            if not (load_state and i == 0 and not game.game_over and len(game.snake) > 0):
                game.reset()

            steps_this_game = 0

            while not game.game_over:
                if steps_this_game >= config.MAX_STEPS_PER_GAME:
                    logger.info("Reached per-game step cap (%d); ending game", config.MAX_STEPS_PER_GAME)
                    game.game_over = True
                    break

                if render:
                    if pygame is None:
                        raise RuntimeError("Rendering requires pygame. Install it or run with --no-render.")
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt

                pre_move_state = agent.symbolic_memory.create_symbolic_state(game.snake, game.food, game.direction)
                action, debug_info = agent.choose_action(game.snake, game.food, game.direction)

                reward = game.move_snake(action, pre_move_state)
                agent.symbolic_memory.update_memory(pre_move_state, action, reward)

                if render:
                    game.render(debug_info if debug else None)

                steps_this_game += 1
                total_steps += 1

                if not render and (steps_this_game % config.PROGRESS_LOG_INTERVAL == 0):
                    logger.info("Game %d | Step %d | Score %d | Life %d", i + 1, steps_this_game, game.score, int(game.life))

                if game.life <= 0:
                    game.game_over = True

            scores.append(game.score)
            logger.info("Game %d/%d: Score=%d Steps=%d", i + 1, num_games, game.score, steps_this_game)

            if (i + 1) % config.MEMORY_SAVE_INTERVAL == 0:
                agent.symbolic_memory.save_memory()

    except KeyboardInterrupt:
        logger.info("Interrupted; saving state...")
        game.save_game_state()
        agent.symbolic_memory.save_memory()
        return 130

    # Save at the end
    game.save_game_state()
    agent.symbolic_memory.save_memory()

    if scores:
        logger.info("Session: avg=%.2f max=%d total_steps=%d", sum(scores) / len(scores), max(scores), total_steps)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Snake Agent")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to run")
    parser.add_argument("--no-render", action="store_true", help="Disable window rendering (headless)")
    parser.add_argument("--load-state", action="store_true", help="Resume from saved state in state/game_state.json")
    parser.add_argument("--debug", action="store_true", help="Enable debug overlay (windowed mode)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-steps", type=int, default=None, help="Per-game step cap (safety)")
    parser.add_argument("--profile", action="store_true", help="Enable profiling output")
    args = parser.parse_args(argv)

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        rc = run(
            num_games=args.num_games,
            render=not args.no_render,
            load_state=args.load_state,
            debug=args.debug,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        print("\n=== Profiling Results ===")
        stats.print_stats(30)
        return rc

    return run(
        num_games=args.num_games,
        render=not args.no_render,
        load_state=args.load_state,
        debug=args.debug,
        seed=args.seed,
        max_steps=args.max_steps,
    )
