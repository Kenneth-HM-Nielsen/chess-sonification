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
import math
import statistics
from collections import deque

import chess

from .ingest import (
    increment_for_move,
    move_number_for_ply,
    parse_time_control,
    period_thresholds,
)

log = logging.getLogger(__name__)

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

CENTRE_SQUARES = (chess.D4, chess.D5, chess.E4, chess.E5)

# Moves assumed to lie ahead where no threshold bounds the period. Used for both
# the opening allowance and the running budget, so pressure is zero at move one
# by construction and measures how far behind that starting pace a player has
# fallen -- which makes bullet and classical comparable without flattening them.
NOMINAL_HORIZON = 40

# Below this, a reply was executed before the position arose. Not a fast
# decision; not a decision at all. One constant, from human reaction time.
PREMOVE_FLOOR_S = 0.15

# Think times are compared against a trailing median of this many of the same
# player's own decisions.
ROLLING_WINDOW_MOVES = 12

# log-ratio clamp, so a single outlier cannot dominate the reverb mapping.
THINK_RELATIVE_CLAMP = 3.0


def _move_count(board: chess.Board) -> int:
    """Legal moves, discounting any that capture a king.

    Behind a null move the side that has just given check can "capture" the
    enemy king, which python-chess reports as legal. Left in, it would inflate
    mobility by one on every checking ply -- a systematic error landing on
    precisely the sharpest moves in the game.
    """
    return sum(
        1
        for move in board.legal_moves
        if board.piece_type_at(move.to_square) != chess.KING
    )


def _mobility(board: chess.Board) -> tuple[int, int]:
    """Legal move count for White and Black in `board`.

    The side not to move is counted behind a null move, which python-chess
    permits even when the mover is in check. The board is restored afterwards.
    """
    mover = _move_count(board)
    board.push(chess.Move.null())
    try:
        waiter = _move_count(board)
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
    return {
        "is_capture": board_before.is_capture(move),
        "is_promotion": move.promotion is not None,
        "forced": board_before.legal_moves.count() == 1,
    }


def moves_to_threshold(thresholds: list[int], move_number: int) -> int:
    """Moves this side must still make after completing `move_number`.

    A clock reading is taken after its move, so the horizon paired with it counts
    the moves still ahead: on move 39 with the control at 40, that is one. On the
    control move itself the threshold has just been passed and the horizon runs
    to the next one, which is what keeps a credited clock from being read as a
    single move's budget.
    """
    for threshold in thresholds:
        if threshold > move_number:
            return threshold - move_number
    return NOMINAL_HORIZON


def initial_budget(periods: list) -> float | None:
    """Seconds per move a player starts the game with, increment included."""
    moves, base, increment = periods[0]
    if base is None:
        return None
    divisor = moves if moves is not None else NOMINAL_HORIZON
    return base / max(1, divisor) + increment


def time_pressure(budget: float, initial: float | None) -> float | None:
    """Budget as a shortfall against the player's own opening pace, in [0, 1].

    Relative rather than absolute: half your starting allowance reads 0.5 whether
    that is ninety seconds or two.
    """
    if not initial or initial <= 0:
        return None
    return round(1.0 - min(1.0, budget / initial), 4)


def think_medians(frames: list[dict]) -> tuple[float | None, dict[str, float | None]]:
    """Whole-game median think time, overall and per player.

    Median rather than mean throughout: think-time distributions are heavily
    right-tailed and one long think would otherwise flatten everything else.
    Premoves are excluded -- they are not thinking.
    """
    per_player: dict[str, list[float]] = {"w": [], "b": []}
    for frame in frames:
        if frame["decision"]:
            per_player[frame["color"]].append(frame["think_time"])
    combined = per_player["w"] + per_player["b"]
    return (
        round(statistics.median(combined), 3) if combined else None,
        {
            side: (statistics.median(values) if values else None)
            for side, values in per_player.items()
        },
    )


def annotate(frames: list[dict]) -> list[dict]:
    """Append board, progress, time-pressure and think-shape features, in place."""
    if not frames:
        return frames
    periods = parse_time_control(frames[0].get("time_control"))
    thresholds = period_thresholds(periods)
    opening_pace = initial_budget(periods)

    # A ply is a decision only when it is known to have taken longer than human
    # reaction time. Unknown think times are not decisions either; downstream can
    # tell the two apart because a premove has a think time and a gap does not.
    for frame in frames:
        think = frame["think_time"]
        frame["decision"] = think is not None and think >= PREMOVE_FLOOR_S
    game_tempo_scale, player_median = think_medians(frames)

    # Repetition is a property of the move stack, not of a position in isolation,
    # so the board is carried forward and pushed rather than rebuilt from FENs.
    # The game does not always start from the standard position.
    board = chess.Board(frames[0].get("start_fen") or chess.STARTING_FEN)
    carried: dict[str, tuple] = {"w": (None, None, None), "b": (None, None, None)}
    recent: dict[str, deque] = {
        "w": deque(maxlen=ROLLING_WINDOW_MOVES),
        "b": deque(maxlen=ROLLING_WINDOW_MOVES),
    }
    since_pawn_move = 0
    since_capture = 0

    for frame in frames:
        move = chess.Move.from_uci(frame["uci"])
        # Continuing past a desync would silently fabricate every board and
        # progress feature for the rest of the game, so stop instead.
        if move not in board.legal_moves:
            raise ValueError(
                f"ply {frame['ply']} ({frame['san']}) is not legal in the replayed "
                f"position {board.fen()!r}; frames are inconsistent with start_fen"
            )

        frame.update(move_features(board, move))
        was_pawn_move = board.piece_type_at(move.from_square) == chess.PAWN
        was_capture = board.is_capture(move)

        board.push(move)
        frame.update(board_features(board))

        since_pawn_move = 0 if was_pawn_move else since_pawn_move + 1
        since_capture = 0 if was_capture else since_capture + 1
        frame.update(
            {
                "halfmove_clock": board.halfmove_clock,
                "repetition_2": board.is_repetition(2),
                "plies_since_pawn_move": since_pawn_move,
                "plies_since_capture": since_capture,
            }
        )

        color = frame["color"]
        move_no = move_number_for_ply(frame["ply"])
        clock = frame["clock_remaining"]
        if clock is None:
            carried[color] = (None, None, None)
        else:
            # Everything here describes the state the move leaves behind, since
            # that is what the clock reading describes: the moves still to make,
            # and the increment that will be paid for them. On the control move
            # that means the new period on both counts, which is why the credited
            # clock is not mistaken for one move's budget.
            remaining = moves_to_threshold(thresholds, move_no)
            budget = clock / remaining + increment_for_move(periods, move_no + 1)
            carried[color] = (
                remaining,
                round(budget, 3),
                time_pressure(budget, opening_pace),
            )

        # Pressure persists between a side's own moves, so each colour's last
        # value is carried forward onto every intervening ply.
        for side in ("w", "b"):
            remaining, budget, pressure = carried[side]
            frame[f"moves_to_threshold_{side}"] = remaining
            frame[f"budget_{side}"] = budget
            frame[f"time_pressure_{side}"] = pressure

        frame["think_relative"] = None
        if frame["decision"]:
            # Strictly trailing: the baseline is taken before this ply joins the
            # window, so a long think is measured against what came before it
            # rather than partly against itself.
            window = recent[color]
            baseline = (
                statistics.median(window)
                if len(window) == window.maxlen
                else player_median[color]
            )
            window.append(frame["think_time"])
            if baseline:
                ratio = math.log(frame["think_time"] / baseline)
                frame["think_relative"] = round(
                    max(-THINK_RELATIVE_CLAMP, min(THINK_RELATIVE_CLAMP, ratio)), 4
                )
        frame["game_tempo_scale"] = game_tempo_scale

    return frames
