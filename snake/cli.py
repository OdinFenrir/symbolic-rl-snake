"""Command-line interface and run loop."""

from __future__ import annotations

import argparse
import cProfile
import json
import logging
import os
import pstats
import time
from pathlib import Path
from typing import Optional

from . import config
from .agent import SnakeAgent
from .game import SnakeGame

logger = logging.getLogger(__name__)


def _open_jsonl(path: Optional[str]):
    if not path:
        return None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.open("a", encoding="utf-8")


def _clear_path(path_str: str) -> bool:
    path = Path(path_str)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except OSError as exc:
        logger.warning("Unable to delete %s: %s", path, exc)
        return False


def _reset_memory_files():
    for suffix in ("", ".bak", ".tmp"):
        _clear_path(config.MEMORY_FILE + suffix)


def _reset_state_files():
    _clear_path(config.GAME_STATE_FILE)


def run(
    num_games: int,
    render: bool,
    load_state: bool,
    debug: bool,
    seed: Optional[int],
    max_steps: Optional[int],
    log_jsonl: Optional[str],
    save_every: Optional[int],
    state_dir: Optional[str],
    no_save: bool,
    reset_memory: bool = False,
    reset_state: bool = False,
    reset_all: bool = False,
) -> int:
    config.UI_DEBUG_MODE = bool(debug)
    config_snapshot = {
        "SAVE_GAME_STATE": config.SAVE_GAME_STATE,
        "SAVE_MEMORY": config.SAVE_MEMORY,
        "MAX_STEPS_PER_GAME": config.MAX_STEPS_PER_GAME,
        "STATE_DIR": config.STATE_DIR,
        "MEMORY_FILE": config.MEMORY_FILE,
        "GAME_STATE_FILE": config.GAME_STATE_FILE,
    }
    if state_dir:
        config.set_state_dir(state_dir)

    if reset_all:
        reset_memory = True
        reset_state = True

    if reset_memory:
        _reset_memory_files()
        logger.info("Memory reset via CLI")
    if reset_state:
        _reset_state_files()
        logger.info("Game state reset via CLI")

    # Evaluation mode: allow loading, but disable writes.
    if no_save:
        config.SAVE_GAME_STATE = False
        config.SAVE_MEMORY = False

    if max_steps is not None:
        config.MAX_STEPS_PER_GAME = int(max_steps)

    config.validate_config()

    agent = SnakeAgent(seed=seed)
    game = SnakeGame(render_enabled=render, seed=seed)

    if load_state:
        if game.load_game_state():
            logger.info("Resumed from %s", config.GAME_STATE_FILE)
        else:
            logger.info("No saved game state found; starting new")

    # If we're rendering, pull pygame from the game (single import/init path).
    pygame = game.pygame
    if render and game.menu_visible:
        game.block_until_menu_closed()

    scores: list[int] = []

    session_safety_rejects = 0
    session_safety_forced = 0
    total_steps = 0
    t0 = time.time()

    jsonl_f = _open_jsonl(log_jsonl)
    save_interval = int(save_every) if save_every is not None else int(config.MEMORY_SAVE_INTERVAL)

    try:
        for i in range(num_games):
            if not (load_state and i == 0 and not game.game_over and len(game.snake) > 0):
                game.reset()

                agent.begin_episode()
            steps_this_game = 0
            start_game_time = time.time()

            while not game.game_over:
                if steps_this_game >= config.MAX_STEPS_PER_GAME:
                    logger.info(
                        "Reached per-game step cap (%d); ending game",
                        config.MAX_STEPS_PER_GAME,
                    )
                    game.game_over = True
                    break

                if render:
                    if pygame is None:
                        raise RuntimeError("Rendering requires pygame. Install it or run with --no-render.")
                    game.handle_pygame_events()
                    if game.menu_memory_requested:
                        agent.symbolic_memory.memory.clear()
                        agent.symbolic_memory.total_updates = 0
                        agent.symbolic_memory.is_modified = False
                        game.menu_memory_requested = False
                        logger.info("Menu cleared persistent memory; fresh learning will start.")
                    if game.menu_visible:
                        game.render_menu()
                        continue

                pre_move_state = agent.symbolic_memory.create_symbolic_state(
                    game.snake, game.food, game.direction
                )
                action, debug_info = agent.choose_action(game.snake, game.food, game.direction, game.life)
                game.update_live_metrics(debug_info.get("metrics", {}))

                reward = game.move_snake(action, pre_move_state)
                agent.symbolic_memory.update_memory(pre_move_state, action, reward)

                if render:
                    game.render(debug_info if debug else None)

                steps_this_game += 1
                total_steps += 1

                if not render and (steps_this_game % config.PROGRESS_LOG_INTERVAL == 0):
                    logger.info(
                        "Game %d | Step %d | Score %d | Life %d",
                        i + 1,
                        steps_this_game,
                        game.score,
                        int(game.life),
                    )

                if game.life <= 0:
                    game.game_over = True

            scores.append(game.score)
            elapsed_game = time.time() - start_game_time
            stats = agent.record_episode_stats(game.score, steps_this_game)
            session_safety_rejects += int(stats.get('safety_rejects', 0))
            session_safety_forced += int(stats.get('safety_forced', 0))
            tuning_metrics = agent.tuner_metrics()
            game.set_metrics_info(tuning_metrics)
            logger.info(
                "Game %d/%d: Score=%d Steps=%d (%.2fs) | safety rejects=%d forced=%d",
                i + 1,
                num_games,
                game.score,
                steps_this_game,
                elapsed_game,
                stats["safety_rejects"],
                stats["safety_forced"],
            )
            if jsonl_f is not None:
                row = {
                    "ts": time.time(),
                    "episode": i + 1,
                    "score": game.score,
                    "steps": steps_this_game,
                    "seed": seed,
                    "render": bool(render),
                    "board": int(config.BOARD_SIZE),
                }
                row.update(
                    {
                        "forced_rate": float(stats.get("forced_rate", 0.0)),
                        "safety_rejects": int(stats.get("safety_rejects", 0)),
                        "safety_forced": int(stats.get("safety_forced", 0)),
                        "tuner_safety": tuning_metrics["safety_bias"],
                        "tuner_reward": tuning_metrics["reward_bias"],
                        "memory_size": len(agent.symbolic_memory.memory),
                        "won": bool(game.won),
                    }
                )
                jsonl_f.write(json.dumps(row) + "\n")
                jsonl_f.flush()

            if save_interval > 0 and ((i + 1) % save_interval == 0):
                agent.symbolic_memory.save_memory()
                game.show_metrics_overlay(tuning_metrics)

        game.save_game_state()
        agent.symbolic_memory.save_memory()
        game.show_metrics_overlay(agent.tuner_metrics())
        if jsonl_f is not None:
            jsonl_f.close()

        elapsed = time.time() - t0
        if scores:
            logger.info(
                "Session: avg=%.2f max=%d games=%d total_steps=%d (%.2fs)",
                sum(scores) / len(scores),
                max(scores),
                len(scores),
                total_steps,
                elapsed,
            )
            try:
                logger.info("Safety totals: rejects=%d forced=%d", session_safety_rejects, session_safety_forced)
                if total_steps > 0:
                    logger.info("forced_rate=%.2f%%", 100.0 * session_safety_forced / float(total_steps))
            except NameError:
                pass
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted; saving state...")
        game.save_game_state()
        agent.symbolic_memory.save_memory()
        if jsonl_f is not None:
            jsonl_f.close()
        return 130
    finally:
        for attr, value in config_snapshot.items():
            setattr(config, attr, value)

def main(argv: Optional[list[str]] = None) -> int:
    # Ensure pygame banner stays hidden even when importing via `snake.cli`.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    parser = argparse.ArgumentParser(description="Snake Agent")
    parser.add_argument("--num-games", "--games", type=int, default=10, help="Number of games to run")
    parser.add_argument("--no-render", action="store_true", help="Disable window rendering (headless)")
    parser.add_argument("--load-state", action="store_true", help="Resume from saved state in state/game_state.json")
    parser.add_argument("--debug", action="store_true", help="Enable debug overlay (windowed mode)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-steps", type=int, default=None, help="Per-game step cap (safety)")
    parser.add_argument("--profile", action="store_true", help="Enable profiling output")

    # “Modern touch” options (optional)
    parser.add_argument(
        "--log-jsonl",
        type=str,
        default=None,
        help="Append per-episode metrics to a JSONL file (e.g. runs/session.jsonl)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Override memory save interval (episodes). Defaults to config.MEMORY_SAVE_INTERVAL.",
    )

    parser.add_argument(
        "--state-dir",
        type=str,
        default=None,
        help="Override state directory (default: state/). Useful for isolated runs.",
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Delete persisted symbolic memory (state/symbolic_memory.msgpack) before running.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete saved game state (state/game_state.json) before running.",
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Delete both memory and saved game state before running.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write game_state or memory to disk (evaluation mode).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Alias for --no-save (kept for convenience).",
    )

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
            log_jsonl=args.log_jsonl,
            save_every=args.save_every,
            state_dir=args.state_dir,
            no_save=args.no_save or args.eval,
            reset_memory=args.reset_memory,
            reset_state=args.reset_state,
            reset_all=args.reset_all,
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
        log_jsonl=args.log_jsonl,
        save_every=args.save_every,
        state_dir=args.state_dir,
        no_save=args.no_save or args.eval,
        reset_memory=args.reset_memory,
        reset_state=args.reset_state,
        reset_all=args.reset_all,
    )
