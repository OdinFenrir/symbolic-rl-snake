# Snake Agent

A compact Snake implementation where a hybrid agent blends:
- local safety and space heuristics
- shortest-path guidance toward food (A*)
- persistent symbolic experience memory that rewrites itself after each move

Supports both a windowed pygame mode and a fast headless evaluation mode. The runtime adapts penalties and rewards over time and includes an ESC-driven menu for clearing state or quitting cleanly.

## Quickstart

### 1. Install dependencies

`
python -m pip install -r requirements.txt
`

### 2. Run locally

Windowed mode (default):

`
python main.py
`

Headless evaluation:

`
python main.py --no-render --num-games 200
`

Show the debug overlay (score + path):

`
python main.py --debug
`

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

## Adaptive tuning

The agent monitors forced / reject counts and keeps a short history of forced rates and scores. After each episode it nudges pocket and reward scaling so the forced rate drifts toward the configured target (about 2%) while still rewarding longer runs. No manual penalty tweaks are required.

## Interactive menu

Windowed runs now support an Esc-driven menu. Press Esc to pause, navigate with the up/down arrow keys and Enter, clear state/symbolic_memory.msgpack or state/game_state.json, or quit cleanly. The menu synchronizes with the CLI so memory clears happen before the next game starts.

## Testing & automation

- python -m pytest tests/test_smoke.py ensures the CLI loop runs headless without crashing (GitHub Actions runs this on every push).
- The GitHub Actions workflow (.github/workflows/ci.yml) runs the headless smoke test plus linting.

## Repository structure

- snake/ : library code (agent, game, memory, config, CLI)
- main.py : convenience entrypoint (calls python -m snake)
- state/ : persisted runtime state (commit for snapshots)
- scripts/ : helper scripts to migrate or reset state
- 	ests/ : minimal smoke tests

## License

MIT (see LICENSE).
