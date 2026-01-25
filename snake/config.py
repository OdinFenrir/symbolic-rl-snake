"""Central configuration for the Snake game and agent."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Logging
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)-12s - %(levelname)-8s - %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging if the host application has not done so already.

    This keeps the package library-friendly (it will not override an existing logging setup),
    while preserving CLI ergonomics.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT, stream=sys.stdout)


configure_logging()

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
GRID_SIZE = 64
BOARD_SIZE = 16
WINDOW_SIZE = BOARD_SIZE * GRID_SIZE
FPS = 15
UI_DEBUG_MODE = False

# Safety caps / logging
MAX_STEPS_PER_GAME = 20000
PROGRESS_LOG_INTERVAL = 200

# Rewards
INITIAL_LIFE = 2000
LIFE_PER_FOOD = 100
MAX_LIFE = 2000

REWARD_FOOD = 25.0
FOOD_REWARD_LENGTH_MULTIPLIER = 0.1
FOOD_REWARD_POWER = 1.15

PENALTY_DEATH = -40.0

# Repeat penalty is applied per step *beyond* REPEAT_THRESHOLD.
REPEAT_THRESHOLD = 8
REPEAT_PENALTY_CAP_STEPS = 12
PENALTY_REPEAT = -0.35

# Adaptive tuning defaults (autorebalances rewards/safety over episodes).
ADAPTIVE_HISTORY_SIZE = 10
ADAPTIVE_TARGET_FORCED_RATE = 0.03
ADAPTIVE_SAFETY_ADJUST_SCALE = 0.35
ADAPTIVE_REWARD_ADJUST_SCALE = 0.4
ADAPTIVE_SCORE_BOOST = 0.045
ADAPTIVE_SCORE_DECAY = 0.015
ADAPTIVE_SAFETY_BIAS_CAP = 1.0
ADAPTIVE_REWARD_BIAS_CAP = 0.5
ADAPTIVE_SCORE_MARGIN = 3.0
ADAPTIVE_FORCE_DEADBAND = 0.005

PENALTY_WALL_HUG = -0.5
WALL_HUG_PENALTY_FOOD_FACTOR = 0.0  # no border penalty when chasing border food

# Memory guardrails
MIN_MEMORY_COUNT = 5
MEMORY_CLIP = 1.0
MEMORY_WEIGHT = 0.15
ABS_HEURISTIC_EPS = 0.02
PENALTY_TAIL_PROXIMITY = -2.0
DISTANCE_REWARD_SCALAR = 1.0
MAX_DISTANCE_REWARD_DELTA = 4.0

CENTER_BIAS_WEIGHT = 0.0
CENTER_BIAS_MAX_LENGTH = 8
CENTER_BIAS_MIN_DISTANCE = 7
EDGE_GAP_PENALTY = -0.9
EDGE_GAP_THRESHOLD = 2
EDGE_PROXIMITY = 3
EDGE_GAP_MIN_LENGTH = 20
TRAP_PENALTY = 3.5

MEMORY_MAX_ENTRIES = 90000
MEMORY_RECENCY_WEIGHT = 0.001
MEMORY_SAVE_INTERVAL = 1


# Symbolic-memory keying (v3 hybrid keys)
# - "compact keys" generalize across snake lengths by using constant-size local features.
# - legacy fallback allows continued reuse of older v2 memory files.
MEMORY_ENABLE_COMPACT_KEYS = True
MEMORY_ENABLE_LEGACY_FALLBACK = True
MEMORY_AGE_BINS = 4
# Length bucket cutoffs (<= cutoff -> bucket index). Values should be monotonic.
MEMORY_LENGTH_BUCKETS = (4, 8, 16, 32, 64)
# Decision weights
HEURISTIC_WEIGHT = 1.0
RSM_WEIGHT = 0.1
SAFE_PATH_BONUS = 1.5
AGE_PROXIMITY_WEIGHT = 0.75
A_STAR_BONUS = 3.0

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
