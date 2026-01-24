from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from . import config

Move = Tuple[int, int]
Cell = Tuple[int, int]

MOVES_4: Sequence[Move] = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _in_bounds(cell: Cell) -> bool:
    r, c = cell
    return 0 <= r < config.BOARD_SIZE and 0 <= c < config.BOARD_SIZE


def _would_eat(cell: Cell, food: Cell | None) -> bool:
    return food is not None and cell == food


def _is_legal_tail_entry(
    snake: Iterable[Cell],
    new_head: Cell,
    food: Cell | None,
) -> bool:
    snake_list = list(snake)
    if not snake_list:
        return True
    tail = snake_list[-1]
    will_eat = _would_eat(new_head, food)
    if not will_eat and new_head == tail:
        return True
    return False


def reverse_move(move: Move, direction: Move) -> bool:
    return move == (-direction[0], -direction[1])


def legal_moves(
    snake: Sequence[Cell],
    food: Cell | None,
    direction: Move,
    *,
    forbid_reverse: bool = True,
) -> list[Move]:
    moves: list[Move] = []
    head = snake[0]
    for move in MOVES_4:
        new_head = (head[0] + move[0], head[1] + move[1])
        if not _in_bounds(new_head):
            continue
        if forbid_reverse and len(snake) > 1 and reverse_move(move, direction):
            continue
        if new_head in snake:
            if _is_legal_tail_entry(snake, new_head, food):
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
    head = snake[0]
    new_head = (head[0] + move[0], head[1] + move[1])
    if not _in_bounds(new_head):
        return False
    if forbid_reverse and len(snake) > 1 and reverse_move(move, direction):
        return False
    if new_head in snake:
        return _is_legal_tail_entry(snake, new_head, food)
    return True
