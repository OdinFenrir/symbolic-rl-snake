# Snake Agent

A compact Snake implementation with an autonomous agent. The agent chooses moves using a blend of:
- local safety/space heuristics
- shortest-path guidance toward food (A*)
- a persistent experience memory that updates after each move

The project supports both a pygame windowed mode and a fast headless mode.

## Quickstart

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run (windowed):

```bash
python main.py
```

Run headless (no window):

```bash
python main.py --no-render --num-games 200
```

Show debug overlay (scores + path):

```bash
python main.py --debug
```

## State and persistence

Runtime state is stored under `state/`:

- `state/symbolic_memory.msgpack` — learned experience values
- `state/game_state.json` — last saved game state (used for resume)

Resume from the last saved game state:

```bash
python main.py --load-state
```

Snapshot your current state into git (recommended workflow):
1. Run the game until you are happy with the learned behavior.
2. Commit `state/` and tag the commit, e.g. `snapshot-YYYYMMDD`.

Reset to a fresh start (delete state files):

```bash
del state\symbolic_memory.msgpack
del state\game_state.json
```



### Useful scripts

- `scripts/migrate_state.ps1` / `.sh`: move legacy root-level state files into `state/`.
- `scripts/reset_state.ps1` / `.sh`: delete state files for a fresh start.

## CLI options

- `--num-games N` / `--games N` : number of games to run (default: 10)
- `--no-render`   : headless mode
- `--load-state`  : resume from `state/game_state.json` if present
- `--debug`       : enable debug overlay in windowed mode
- `--seed SEED`   : set RNG seed for reproducibility
- `--max-steps N` : per-game safety cap (overrides config)
- `--state-dir DIR` : override the state directory (isolated evaluations / experiments)
- `--no-save` : do not write game_state or memory to disk (evaluation mode)
- `--eval` : alias for `--no-save`

### Adaptive tuning

Runs automatically shape their own reward and safety balance. After each episode the agent monitors forced/reject counts and nudges its pocket penalties, heuristic rewards, and repeat penalties toward the settings that keep forced rates near the configured target (currently ~2%) while still rewarding longer runs, so you can just run `python -m main` and focus on evaluating the behavior instead of tweaking knobs manually.

### Interactive menu

When rendering with `python main.py` (or `python -m snake` without `--no-render`), press `Esc` to pause and open the menu. From there you can resume, access the options submenu, clear the persisted memory/`game_state.json`, or quit cleanly. The menu manages the persistence files for you, and any memory reset triggers a rebuild on the next games.

## Repository structure

- `snake/` : library code (agent, game, memory, config, CLI)
- `main.py` : convenience entrypoint (calls `python -m snake`)
- `state/` : persisted runtime state (commit when you want a snapshot)
- `tests/` : minimal smoke tests

## License

MIT (see `LICENSE`).
