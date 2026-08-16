"""Engine-free structural features computed per ply.

Two independent groups. Board features measure the position: how much room each
side has, how locked the pawns are, how much heat is on each king. Time-pressure
features measure the players instead, from clock state — a scramble and a
comfortable position can share a board and sound nothing alike.

Every board function is pure: it takes a `chess.Board`, returns a dict, and
leaves the caller's board unmodified.
"""

from __future__ import annotations

import logging

import chess

from .ingest import increment_for_move, parse_time_control, period_thresholds

log = logging.getLogger(__name__)

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

CENTRE_SQUARES = (chess.D4, chess.D5, chess.E4, chess.E5)

# Moves assumed remaining in an open-ended final period, where there is no
# threshold left to divide the clock by.
NOMINAL_FINAL_PERIOD_MOVES = 20

# Seconds per move at which time pressure reads as zero. 6s/move gives ~0.9.
PRESSURE_SATURATION_S = 60.0


def _mobility(board: chess.Board) -> tuple[int, int]:
    """Legal move count for White and Black in `board`.

    The side not to move is counted behind a null move, which python-chess
    permits even when the mover is in check. The board is restored afterwards.
    """
    mover = board.legal_moves.count()
    board.push(chess.Move.null())
    try:
        waiter = board.legal_moves.count()
    finally:
        board.pop()
    return (mover, waiter) if board.turn == chess.WHITE else (waiter, mover)


def _locked_pawns(board: chess.Board) -> int:
    """Pawns directly blocked by an enemy pawn one square ahead, both colours."""
    white = board.pieces(chess.PAWN, chess.WHITE)
    black = board.pieces(chess.PAWN, chess.BLACK)
    blocked = sum(1 for square in white if square + 8 in black)
    blocked += sum(1 for square in black if square - 8 in white)
    return blocked


def _pawn_tension(board: chess.Board) -> int:
    """Pawn pairs that attack each other.

    Pawn attacks are symmetric, so counting the black pawns attacked by each
    white pawn already counts each mutually attacking pair exactly once.
    """
    black = board.pieces(chess.PAWN, chess.BLACK)
    return sum(
        len(board.attacks(square) & black)
        for square in board.pieces(chess.PAWN, chess.WHITE)
    )


def _king_pressure(board: chess.Board, king_color: chess.Color) -> int:
    """Enemy attackers on the king square and its neighbours."""
    king_square = board.king(king_color)
    if king_square is None:
        return 0
    zone = board.attacks(king_square)
    zone.add(king_square)
    return sum(len(board.attackers(not king_color, square)) for square in zone)


def _material_balance(board: chess.Board) -> int:
    """Standard 1/3/3/5/9 weights, White POV."""
    return sum(
        value
        * (
            len(board.pieces(piece_type, chess.WHITE))
            - len(board.pieces(piece_type, chess.BLACK))
        )
        for piece_type, value in PIECE_VALUES.items()
    )


def _centre_occupancy(board: chess.Board, color: chess.Color) -> int:
    """Pieces of `color` standing on d4, d5, e4 or e5."""
    return sum(
        1
        for square in CENTRE_SQUARES
        if (piece := board.piece_at(square)) is not None and piece.color == color
    )


def board_features(board: chess.Board) -> dict:
    """Structural measurements for a position."""
    mobility_w, mobility_b = _mobility(board)
    return {
        "mobility_w": mobility_w,
        "mobility_b": mobility_b,
        "locked_pawns": _locked_pawns(board),
        "pawn_tension": _pawn_tension(board),
        "king_pressure_w": _king_pressure(board, chess.WHITE),
        "king_pressure_b": _king_pressure(board, chess.BLACK),
        "material_balance": _material_balance(board),
        "centre_occupancy_w": _centre_occupancy(board, chess.WHITE),
        "centre_occupancy_b": _centre_occupancy(board, chess.BLACK),
        "is_check": board.is_check(),
    }


def move_features(board_before: chess.Board, move: chess.Move) -> dict:
    """Measurements that need the move and the position it was played from."""
    legal = list(board_before.legal_moves)
    return {
        "is_capture": board_before.is_capture(move),
        "is_promotion": move.promotion is not None,
        "forced": len(legal) == 1,
    }


def moves_to_threshold(thresholds: list[int], move_number: int) -> int:
    """Moves this side still has before the next period threshold."""
    for threshold in thresholds:
        if threshold >= move_number:
            return threshold - move_number
    return NOMINAL_FINAL_PERIOD_MOVES


def time_pressure(budget: float) -> float:
    """Seconds-per-move budget mapped to [0, 1], saturating."""
    return round(1.0 - min(1.0, budget / PRESSURE_SATURATION_S), 4)


def annotate(frames: list[dict], periods: list | None = None) -> list[dict]:
    """Append board and time-pressure features to each frame, in place."""
    if not frames:
        return frames
    if periods is None:
        periods = parse_time_control(frames[0].get("time_control"))
    thresholds = period_thresholds(periods)

    board_before = chess.Board()
    # Pressure persists between a side's own moves, so the last computed value
    # for each colour is carried forward onto every intervening ply.
    carried: dict[str, tuple] = {"w": (None, None, None), "b": (None, None, None)}
    desynced = False

    for frame in frames:
        move = chess.Move.from_uci(frame["uci"])
        if not desynced and move not in board_before.legal_moves:
            log.warning(
                "ply %d (%s) is not legal in the reconstructed position; the game "
                "may not start from the standard position",
                frame["ply"],
                frame["san"],
            )
            desynced = True

        frame.update(move_features(board_before, move))
        board_after = chess.Board(frame["fen_after"])
        frame.update(board_features(board_after))

        color = frame["color"]
        move_number = (frame["ply"] + 1) // 2
        clock = frame["clock_remaining"]
        if clock is None:
            carried[color] = (None, None, None)
        else:
            remaining = moves_to_threshold(thresholds, move_number)
            budget = clock / max(1, remaining) + increment_for_move(
                periods, move_number
            )
            carried[color] = (remaining, round(budget, 3), time_pressure(budget))

        for side in ("w", "b"):
            remaining, budget, pressure = carried[side]
            frame[f"moves_to_threshold_{side}"] = remaining
            frame[f"budget_{side}"] = budget
            frame[f"time_pressure_{side}"] = pressure

        board_before = board_after

    return frames
