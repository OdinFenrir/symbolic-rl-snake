from __future__ import annotations

from typing import Sequence, Tuple

from . import config

Move = Tuple[int, int]
Cell = Tuple[int, int]

MOVES_4: Sequence[Move] = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _in_bounds(cell: Cell) -> bool:
    r, c = cell
    return 0 <= r < config.BOARD_SIZE and 0 <= c < config.BOARD_SIZE


def _would_eat(cell: Cell, food: Cell | None) -> bool:
    return food is not None and cell == food


def reverse_move(move: Move, direction: Move) -> bool:
    return move == (-direction[0], -direction[1])


def _is_legal_tail_entry(
    *,
    new_head: Cell,
    tail: Cell,
    food: Cell | None,
) -> bool:
    """Return True if entering the current tail cell is legal.

    The tail cell is only enterable if the snake is NOT going to eat on this step.
    If the snake eats, the tail does not move, so entering the tail would collide.
    """
    return (not _would_eat(new_head, food)) and (new_head == tail)


def legal_moves(
    snake: Sequence[Cell],
    food: Cell | None,
    direction: Move,
    *,
    forbid_reverse: bool = True,
) -> list[Move]:
    """Return legal 4-way moves under the engine's collision rules.

    This implementation avoids repeated O(n) membership checks by using a set.
    """
    if not snake:
        return []

    head = snake[0]
    tail = snake[-1]
    occupied = set(snake)

    moves: list[Move] = []
    for move in MOVES_4:
        new_head = (head[0] + move[0], head[1] + move[1])
        if not _in_bounds(new_head):
            continue
        if forbid_reverse and len(snake) > 1 and reverse_move(move, direction):
            continue

        if new_head in occupied:
            if _is_legal_tail_entry(new_head=new_head, tail=tail, food=food):
                moves.append(move)
            continue

        moves.append(move)
    return moves


def is_legal_move(
    snake: Sequence[Cell],
    food: Cell | None,
    move: Move,
    direction: Move,
    *,
    forbid_reverse: bool = True,
) -> bool:
    """Return True if the move is legal under the engine's collision rules."""
    if not snake:
        return False

    head = snake[0]
    tail = snake[-1]
    new_head = (head[0] + move[0], head[1] + move[1])

    if not _in_bounds(new_head):
        return False
    if forbid_reverse and len(snake) > 1 and reverse_move(move, direction):
        return False

    occupied = set(snake)
    if new_head in occupied:
        return _is_legal_tail_entry(new_head=new_head, tail=tail, food=food)

    return True
