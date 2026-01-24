# Snake Agent

A compact Snake implementation where a hybrid agent blends:
- local safety and space heuristics
- shortest-path guidance toward food (A*)
- persistent symbolic experience memory that rewrites itself after each move

Supports both a windowed pygame mode and a fast headless evaluation mode. The runtime adapts penalties and rewards over time and includes an ESC-driven menu for clearing state or quitting cleanly.

## Quickstart

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Run locally

Windowed mode (default):

```bash
python main.py
```

Headless evaluation:

```bash
python main.py --no-render --num-games 200
```

Show the debug overlay (score + path):

```bash
python main.py --debug
```

If you prefer the package entry point, `python -m snake` works the same, and after installing via `pip install -e .` you can run `snake-agent` directly.

## State and persistence

Runtime state lives under state/:

- state/symbolic_memory.msgpack - adaptive experience values
- state/game_state.json - last saved board, direction, and snake

Use --load-state or python main.py --load-state to resume, or delete those files for a fresh start. Useful scripts live under scripts/:

- scripts/migrate_state.ps1 / .sh: move legacy root-level state files into state/.
- scripts/reset_state.ps1 / .sh: delete state files for a fresh start.

## CLI options

| Flag | Effect |
| --- | --- |
| --num-games N / --games N | Number of games to run (default: 10) |
| --no-render | Headless mode |
| --load-state | Resume from state/game_state.json |
| --debug | Enable the debug overlay text and path overlays |
| --seed SEED | Random seed for reproducibility |
| --max-steps N | Per-game safety cap (overrides config) |
| --state-dir DIR | Override the state directory (isolated runs) |
| --no-save / --eval | Skip writing game or memory state (evaluation mode) |
| --reset-memory | Delete persisted symbolic memory before running |
| --reset-state | Delete saved game state before running |
| --reset-all | Delete both memory and saved game state before running |

## Adaptive tuning

The agent monitors forced / reject counts and keeps a short history of forced rates and scores. After each episode it nudges pocket and reward scaling so the forced rate drifts toward the configured target (about 2%) while still rewarding longer runs. No manual penalty tweaks are required.

## Interactive menu

Windowed runs now support an Esc-driven menu. Press Esc to pause, navigate with the up/down arrow keys and Enter, clear state/symbolic_memory.msgpack or state/game_state.json, or quit cleanly. The menu synchronizes with the CLI so memory clears happen before the next game starts.

The options screen exposes a “Toggle tuning metrics” entry that reveals the adaptive tuner’s current `safety_bias`, `reward_bias`, and best score in the menu overlay and briefly flashes them after every memory save. Press `M` during play to open the extended live metrics panel (left side) and keep watching the heuristics, final-score distribution, life, and tuner biases in real time without opening the menu.

## Testing & automation

- `python -m pytest` now exercises the full suite (smoke + determinism + memory/tail-rule coverage) before each push.
- `python -m unittest discover -s tests -p "test_*.py" -v` mirrors the GitHub Actions smoke test entry so CLI regressions stay visible.
- The GitHub Actions workflow runs the headless smoke test and captures JSONL telemetry for easier correlation with forced rate and tuner bias trends.

## Logging & telemetry

`--log-jsonl PATH` now records extra columns per episode: `forced_rate`, `safety_rejects`, `safety_forced`, `tuner_safety`, `tuner_reward`, `memory_size`, and `won`. These give you actionable benchmarking data without rendering and let you track how the adaptive tuner drifts.

## Visual + planning parity

The renderer now draws the snake with a teal-to-navy body gradient whose ratio to the head/tail colors is shared with the agent. Each decision exposes a `segment_ages` vector (head=0, tail=1) in the symbolic state so `SymbolicMemory` can distinguish freshly traveled paths from aging tails, and the same age data drives the live metrics overlay and menu palette. This keeps rendering and planning in sync, making it obvious where the agent is relying on “young” segments versus “stale” ones when escaping pockets.

## Key project files

- `main.py` / `snake/__main__.py`: package entry point that forwards to `snake.cli.main`.
- `snake/cli.py`: orchestrates rendering vs. headless runs, adapts `config`, and manages persistence/no-save evaluation.
- `snake/game.py`: engine rules, reward shaping, rendering, and the ESC-driven menu that can wipe memory/state.
- `snake/agent.py`: agent heuristics, safety filters, A* guidance, loop-breaking penalties, and the adaptive tuner.
- `snake/memory.py`: versioned symbolic memory, msgpack persistence, legacy migration, and pruning hooks.
- `snake/config.py`: central defaults, `STATE_DIR` derivation, and validation logic for rewards/penalties.
- `pyproject.toml` + `requirements.txt`: packaging metadata (including the new pytest ignore list) and runtime dependencies.
- `tests/`: unit/behavioral tests (smoke, tail rule, memory, determinism) that guard safety and determinism guarantees.
- `state/`: runtime persistence (symbolic memory + saved boards); keep it around for analysis, but wipe via `scripts/` or the in-game menu when needed.

## Repository structure

- snake/ : library code (agent, game, memory, config, CLI)
- main.py : convenience entrypoint (calls python -m snake)
- state/ : persisted runtime state (commit for snapshots)
- scripts/ : helper scripts to migrate or reset state
- 	ests/ : minimal smoke tests

## License

MIT (see LICENSE).
