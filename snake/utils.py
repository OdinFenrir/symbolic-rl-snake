from __future__ import annotations

from collections import deque
from typing import Deque, Tuple


def segment_age_ratio(index: int, length: int) -> float:
    """Return a normalized "age" for a body segment: head=0, tail=1."""
    if length <= 1:
        return 0.0
    return float(index) / float(length - 1)


def segment_age_sequence(snake: Deque[Tuple[int, int]]) -> Tuple[float, ...]:
    """Return normalized ages for every segment in the snake (head first)."""
    length = len(snake)
    return tuple(segment_age_ratio(idx, length) for idx in range(length))
