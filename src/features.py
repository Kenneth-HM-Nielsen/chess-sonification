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
    Period,
    increment_for_move,
    move_number_for_ply,
    parse_time_control,
    period_bounds,
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

# A king can capture but can never be traded, so as an attacker it is never the
# cheap one. Only the first test -- outnumbered -- can catch a piece a king takes.
ATTACKER_VALUES = {**PIECE_VALUES, chess.KING: 1000}

# Moves assumed to lie ahead where the rules bound nothing: a prior on game
# length, not a rule. It sets both the opening allowance and the running budget,
# so pressure is zero at move one by construction and measures how far behind
# that starting pace a player has fallen -- which is what makes bullet and
# classical comparable without flattening them.
NOMINAL_HORIZON = 40

# Floor for the counting-down horizon, used where no increment sets a sustainable
# pace -- an open-ended period without one, or the run-on past a control that the
# rules do not extend. It takes over NOMINAL_HORIZON - MIN_HORIZON moves in, so
# move 30 of a sudden-death game rather than move 40, and from there pressure
# becomes a function of the absolute clock, because no defined pace remains to
# measure against. Must stay above zero: it is the divisor of the budget.
MIN_HORIZON = 10

# Plies a side's pressure survives without a fresh clock reading. Stale pressure
# is worse than absent pressure: absence switches the layer off honestly, while
# a value carried from twenty plies ago keeps it playing a lie.
FORWARD_FILL_PLIES = 4

# Below this, a reply was executed before the position arose. Not a fast
# decision; not a decision at all. One constant, from human reaction time.
PREMOVE_FLOOR_S = 0.15

# How a ply's think time came about, reported on every frame as `decision_state`.
# A premove and a measurement artifact both look instant, and must not sound
# alike: only a genuinely measured sub-floor value is a premove.
#
# Three values, and the naming matters. `DECIDED` was `DECISION = "decision"`
# while the field was still boolean, and a reader meeting a bare `decision` on a
# three-valued field reads it as the yes half of a yes/no -- which is exactly the
# conflation the third value exists to remove. Adjective, not noun, so the three
# read as alternatives.
DECIDED = "decided"
PREMOVE = "premove"
UNKNOWN = "unknown"


# Think times are compared against a trailing median of this many of the same
# player's own decisions.
ROLLING_WINDOW_MOVES = 12

# log-ratio clamp, so a single outlier cannot dominate the reverb mapping.
THINK_RELATIVE_CLAMP = 3.0


def classify_think(think_time: float | None, clamped: bool) -> str:
    """Whether a ply's think time is a decision, a premove, or unknown."""
    if think_time is None or clamped:
        return UNKNOWN
    return PREMOVE if think_time < PREMOVE_FLOOR_S else DECIDED


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


def _hanging_material(board: chess.Board) -> int:
    """Value of every loose piece on the board, both colours.

    A piece is loose when more enemies attack its square than friends defend it,
    or when the cheapest attacker is worth less than the piece — the two ways a
    tactic gets started. Deliberately transient rather than a material count: a
    gambit is a pawn down permanently, but what makes it sharp is that pieces are
    hanging *now*. A settled queen-versus-rook endgame reads near zero, where an
    imbalance measure would read maximal for ninety plies on a decided verdict.

    Loose pieces are what tactics operate on, which makes this the honest
    engine-free proxy for something being about to happen.
    """
    loose = 0
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(not piece.color, square)
        if not attackers:
            continue
        defenders = board.attackers(piece.color, square)
        value = PIECE_VALUES[piece.piece_type]
        cheapest = min(
            ATTACKER_VALUES[board.piece_type_at(attacker)] for attacker in attackers
        )
        if len(attackers) > len(defenders) or cheapest < value:
            loose += value
    return loose


def _material_total(board: chess.Board) -> int:
    """All material still on the board, both colours, same weights.

    Distinct from the balance: a queen-for-queen trade leaves the balance
    untouched while taking eighteen points off the board, and it is the removal
    that simplifies a position.
    """
    return sum(
        value
        * (
            len(board.pieces(piece_type, chess.WHITE))
            + len(board.pieces(piece_type, chess.BLACK))
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
        "material_total": _material_total(board),
        "hanging_material": _hanging_material(board),
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


def moves_to_threshold(periods: list[Period], move_number: int) -> int:
    """Moves this side must still make after completing `move_number`.

    A clock reading is taken after its move, so the horizon paired with it counts
    the moves still ahead: on move 39 with the control at 40, that is one. On the
    control move itself the threshold has just been passed and the horizon runs
    to the next one, which is what keeps a credited clock from being read as a
    single move's budget.

    Whether an open-ended period counts down turns on the increment, because the
    two situations are not alike. Without one, the clock has to cover every
    remaining move: a finite demand on a depleting resource, so the horizon is a
    real countdown. With one, the increment is a sustainable pace by itself and
    the clock covers only the excess above it, so the situation is memoryless and
    there is no endpoint to count towards. Counting down anyway would make
    pressure fall on a flat clock, which is the move-number dependence this whole
    horizon exists to avoid.
    """
    if not periods:
        return NOMINAL_HORIZON
    bounds = period_bounds(periods)
    for bound in bounds:
        if bound > move_number:
            return bound - move_number

    # Asked of the period the move leaves the game in, by the same reckoning the
    # budget uses for its increment -- one definition of what is in force, not two.
    if increment_for_move(periods, move_number + 1) > 0:
        # NOMINAL_HORIZON here is a prior on game length, not a rule.
        return NOMINAL_HORIZON
    period_start = bounds[-1] if bounds else 0
    return max(NOMINAL_HORIZON - (move_number - period_start), MIN_HORIZON)


def initial_budget(periods: list) -> float | None:
    """Seconds per move a player starts the game with, increment included."""
    moves, base, increment = periods[0]
    if base is None:
        return None
    # Where the rules bound nothing, the nominal stands in as a prior on length.
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
        if frame["decision_state"] == DECIDED:
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
    opening_pace = initial_budget(periods)

    for frame in frames:
        frame["decision_state"] = classify_think(
            frame["think_time"], frame.get("think_time_clamped", False)
        )
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
    last_reading: dict[str, int | None] = {"w": None, "b": None}

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
        board.push(move)
        frame.update(board_features(board))
        frame.update(
            {
                "halfmove_clock": board.halfmove_clock,
                "repetition_2": board.is_repetition(2),
            }
        )

        color = frame["color"]
        move_no = move_number_for_ply(frame["ply"])
        clock = frame["clock_remaining"]
        # A missing reading leaves the last good value in place rather than
        # discarding it; how long it stays usable is decided below.
        if clock is not None:
            last_reading[color] = frame["ply"]
            # Everything here describes the state the move leaves behind, since
            # that is what the clock reading describes: the moves still to make,
            # and the increment that will be paid for them. On the control move
            # that means the new period on both counts, which is why the credited
            # clock is not mistaken for one move's budget.
            remaining = moves_to_threshold(periods, move_no)
            budget = clock / remaining + increment_for_move(periods, move_no + 1)
            carried[color] = (
                remaining,
                round(budget, 3),
                time_pressure(budget, opening_pace),
            )

        # Pressure persists between a side's own moves, so each colour's last
        # value is carried forward onto every intervening ply -- but only so far.
        # Past that the reading is too old to stand for the player's state, and
        # absence is reported instead.
        for side in ("w", "b"):
            reading = last_reading[side]
            if reading is None or frame["ply"] - reading > FORWARD_FILL_PLIES:
                remaining, budget, pressure = (None, None, None)
            else:
                remaining, budget, pressure = carried[side]
            frame[f"moves_to_threshold_{side}"] = remaining
            frame[f"budget_{side}"] = budget
            frame[f"time_pressure_{side}"] = pressure

        frame["think_relative"] = None
        if frame["decision_state"] == DECIDED:
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
