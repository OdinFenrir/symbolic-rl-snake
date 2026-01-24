"""Central configuration for the Snake game and agent."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)-12s - %(levelname)-8s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("snake")


# ----------------------------
# Paths / persistence
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_state_dir() -> Path:
    """Resolve the state directory (supports env override)."""
    raw = os.environ.get("SNAKE_STATE_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return REPO_ROOT / "state"


STATE_DIR = _default_state_dir()
STATE_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_FILE = str(STATE_DIR / "symbolic_memory.msgpack")
GAME_STATE_FILE = str(STATE_DIR / "game_state.json")

# If False, the program will not write state/memory to disk.
SAVE_MEMORY = True
SAVE_GAME_STATE = True


def set_state_dir(state_dir: str | Path) -> None:
    """Update the state directory and derived file paths at runtime.

    This is primarily used by the CLI for isolated evaluations.
    """
    global STATE_DIR, MEMORY_FILE, GAME_STATE_FILE
    STATE_DIR = Path(state_dir).expanduser().resolve()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE = str(STATE_DIR / "symbolic_memory.msgpack")
    GAME_STATE_FILE = str(STATE_DIR / "game_state.json")


# ----------------------------
# Game & rendering
# ----------------------------
BOARD_SIZE = 15
GRID_SIZE = 40
WINDOW_SIZE = BOARD_SIZE * GRID_SIZE
FPS = 15
UI_DEBUG_MODE = False

# Safety caps / logging
MAX_STEPS_PER_GAME = 5000
PROGRESS_LOG_INTERVAL = 200

# Rewards
INITIAL_LIFE = 1000
LIFE_PER_FOOD = 100
MAX_LIFE = 1000

REWARD_FOOD = 25.0
FOOD_REWARD_LENGTH_MULTIPLIER = 0.1

PENALTY_DEATH = -40.0

# Repeat penalty is applied per step *beyond* REPEAT_THRESHOLD.
REPEAT_THRESHOLD = 8
REPEAT_PENALTY_CAP_STEPS = 12
PENALTY_REPEAT = -0.35

# Adaptive tuning defaults (autorebalances rewards/safety over episodes).
ADAPTIVE_HISTORY_SIZE = 10
ADAPTIVE_TARGET_FORCED_RATE = 0.02
ADAPTIVE_SAFETY_ADJUST_SCALE = 0.35
ADAPTIVE_REWARD_ADJUST_SCALE = 0.4
ADAPTIVE_SCORE_BOOST = 0.045
ADAPTIVE_SCORE_DECAY = 0.015
ADAPTIVE_SAFETY_BIAS_CAP = 1.0
ADAPTIVE_REWARD_BIAS_CAP = 0.5

PENALTY_WALL_HUG = -0.5
PENALTY_TAIL_PROXIMITY = -2.0
DISTANCE_REWARD_SCALAR = 1.0

CENTER_BIAS_WEIGHT = 0.05
CENTER_BIAS_MAX_LENGTH = 8
CENTER_BIAS_MIN_DISTANCE = 7
EDGE_GAP_PENALTY = -0.9
EDGE_GAP_THRESHOLD = 2
EDGE_PROXIMITY = 3
EDGE_GAP_MIN_LENGTH = 20
TRAP_PENALTY = 3.5
SAFE_PATH_BONUS = 1.0
AGE_PROXIMITY_WEIGHT = 0.45

MEMORY_MAX_ENTRIES = 90000
MEMORY_RECENCY_WEIGHT = 0.001
MEMORY_SAVE_INTERVAL = 1

# Decision weights
HEURISTIC_WEIGHT = 1.0
RSM_WEIGHT = 0.1
A_STAR_BONUS = 2.0

# Recursive lookahead
RSM_MIN_DEPTH = 2
RSM_MAX_DEPTH = 4
RSM_DECAY_FACTOR = 0.6


def validate_config() -> None:
    """Basic sanity checks."""
    ok = True
    if BOARD_SIZE <= 5:
        logger.error("BOARD_SIZE must be > 5")
        ok = False
    if GRID_SIZE <= 4:
        logger.error("GRID_SIZE must be > 4")
        ok = False
    if MAX_STEPS_PER_GAME < 100:
        logger.error("MAX_STEPS_PER_GAME must be >= 100")
        ok = False
    if REPEAT_THRESHOLD < 0:
        logger.error("REPEAT_THRESHOLD must be >= 0")
        ok = False
    if REPEAT_PENALTY_CAP_STEPS < 0:
        logger.error("REPEAT_PENALTY_CAP_STEPS must be >= 0")
        ok = False
    if not ok:
        raise SystemExit(1)
