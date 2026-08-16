"""PGN parsing into a per-ply frame list.

Reads a PGN with python-chess built-ins, deriving clock and evaluation data from
`[%clk]` / `[%eval]` annotations where the source provides them. Games with no
clocks, no evals, or neither are all supported; missing data stays `None` rather
than being guessed at.

Multi-period classical time controls are handled per period, so the increment
used for a think time is the one actually in force at that move, and the bulk
time credited at a period boundary is subtracted back out instead of being read
as a negative think time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chess
import chess.pgn

log = logging.getLogger(__name__)

MATE_CP = 10000


# (moves_in_period, base_seconds, increment). moves_in_period None means "to the
# end of the game"; base_seconds None means the header gave no usable allocation.
Period = tuple[int | None, int | None, int]

UNKNOWN_PERIODS: list[Period] = [(None, None, 0)]


def parse_time_control(time_control: str | None) -> list[Period]:
    """Ordered periods from a PGN TimeControl header.

    '40/7200:20/3600:900+30' -> [(40, 7200, 0), (20, 3600, 0), (None, 900, 30)]
    '180+2'                  -> [(None, 180, 2)]
    '600'                    -> [(None, 600, 0)]
    '-' / '?' / '*180' / absent -> [(None, None, 0)]
    """
    if not time_control:
        return list(UNKNOWN_PERIODS)
    tc = time_control.strip()
    if tc in {"-", "?", ""} or tc.startswith("*"):
        return list(UNKNOWN_PERIODS)

    periods: list[Period] = []
    for segment in tc.split(":"):
        segment = segment.strip()
        if not segment:
            continue
        moves: int | None = None
        if "/" in segment:
            moves_text, segment = segment.split("/", 1)
            try:
                moves = int(moves_text)
            except ValueError:
                log.warning("unparseable TimeControl %r, treating as unknown", time_control)
                return list(UNKNOWN_PERIODS)
        increment = 0
        if "+" in segment:
            segment, increment_text = segment.split("+", 1)
            try:
                increment = int(float(increment_text))
            except ValueError:
                log.warning("unparseable increment in TimeControl %r, assuming 0", time_control)
        try:
            base = int(float(segment))
        except ValueError:
            log.warning("unparseable TimeControl %r, treating as unknown", time_control)
            return list(UNKNOWN_PERIODS)
        periods.append((moves, base, increment))

    return periods or list(UNKNOWN_PERIODS)


def increment_for_move(periods: list[Period], move_number: int) -> int:
    """Increment of the period that `move_number` falls in."""
    first_move = 1
    for moves, _base, increment in periods:
        if moves is None or move_number < first_move + moves:
            return increment
        first_move += moves
    return periods[-1][2]


def period_thresholds(periods: list[Period]) -> list[int]:
    """Move numbers whose completion credits the next period's allocation.

    For '40/7200:20/3600:900+30' this is [40, 60]: a player who completes move 40
    is credited the 60 minutes of the second period, and completing move 60 is
    credited the 15 minutes of the third. The final period is open-ended and
    credits nothing further, so it never contributes a threshold.
    """
    thresholds: list[int] = []
    completed = 0
    for moves, _base, _increment in periods[:-1]:
        if moves is None:
            break
        completed += moves
        thresholds.append(completed)
    return thresholds


def is_period_boundary(periods: list[Period], move_number: int) -> bool:
    """True on the move whose completion credits a new period's time.

    This is the move *at* the cumulative threshold rather than the one after it:
    FIDE adds the next period's allocation when a player completes the period's
    final move, and the `[%clk]` recorded for that move already includes it.
    """
    return move_number in period_thresholds(periods)


def credited_base(periods: list[Period], move_number: int) -> int | None:
    """Base seconds credited on completing `move_number`, if it is a boundary."""
    completed = 0
    for index, (moves, _base, _increment) in enumerate(periods[:-1]):
        if moves is None:
            break
        completed += moves
        if completed == move_number:
            return periods[index + 1][1]
    return None


def _eval_cp(node: chess.pgn.ChildNode) -> int | None:
    """White-POV centipawns from a `[%eval]` annotation; mate as +/-10000."""
    povscore = node.eval()
    if povscore is None:
        return None
    score = povscore.white()
    if score.is_mate():
        return MATE_CP if (score.mate() or 0) > 0 else -MATE_CP
    return score.score()


def ingest(pgn_path: Path) -> list[dict]:
    """Parse the first game in `pgn_path` into a list of per-ply frame dicts."""
    pgn_path = Path(pgn_path)
    with pgn_path.open(encoding="utf-8-sig", errors="replace") as fh:
        game = chess.pgn.read_game(fh)
        if game is None:
            raise ValueError(f"no game found in {pgn_path}")
        if chess.pgn.read_game(fh) is not None:
            log.warning("%s contains more than one game; using the first", pgn_path.name)

    for err in game.errors:
        log.warning("%s: parser error: %s", pgn_path.name, err)

    time_control = game.headers.get("TimeControl")
    periods = parse_time_control(time_control)
    start_fen = game.board().fen()
    # Without a usable allocation there is no way to tell an instant reply from a
    # bulk credit, so unexplained clock rises are reported as unknown, not zero.
    allocation_known = any(base is not None for _moves, base, _inc in periods)
    board = game.board()
    # Baseline clock per colour. Reset to None whenever a ply lacks a clock, so a
    # think time is never measured across a gap in the annotations.
    last_clock: dict[str, float | None] = {"w": None, "b": None}
    frames: list[dict] = []

    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        color = "w" if board.turn == chess.WHITE else "b"
        moved = board.piece_at(move.from_square)
        san = board.san(move)

        move_number = (ply + 1) // 2
        increment = increment_for_move(periods, move_number)
        boundary = is_period_boundary(periods, move_number)

        clock = node.clock()
        baseline = last_clock[color]
        if clock is None or baseline is None:
            think_time = None
        elif boundary:
            # The clock rises here: the new period's allocation has already been
            # added to the reading. Subtract it back out rather than clamping,
            # which would report 0 at the tensest moment of the time scramble.
            base = credited_base(periods, move_number)
            if base is None:
                think_time = None
            else:
                think_time = baseline - (clock - base) + increment
                if think_time < 0:
                    log.warning(
                        "%s ply %d (%s): boundary reconstruction gave %.1fs, "
                        "reporting unknown",
                        pgn_path.name,
                        ply,
                        san,
                        think_time,
                    )
                    think_time = None
                else:
                    think_time = round(think_time, 3)
        else:
            think_time = baseline - clock + increment
            if think_time < 0 and not allocation_known:
                log.warning(
                    "%s ply %d (%s): clock rose by %.1fs with no TimeControl to "
                    "explain it, reporting unknown",
                    pgn_path.name,
                    ply,
                    san,
                    -think_time,
                )
                think_time = None
            elif think_time < 0:
                log.warning(
                    "%s ply %d (%s): negative think time %.1fs, clamped to 0",
                    pgn_path.name,
                    ply,
                    san,
                    think_time,
                )
                think_time = 0.0
            else:
                think_time = round(think_time, 3)
        last_clock[color] = clock

        board.push(move)
        frames.append(
            {
                "ply": ply,
                "san": san,
                "uci": move.uci(),
                "color": color,
                "piece": moved.symbol().upper() if moved else "P",
                "from_sq": chess.square_name(move.from_square),
                "to_sq": chess.square_name(move.to_square),
                "clock_remaining": clock,
                "think_time": think_time,
                "period_boundary": boundary,
                # Carried on every frame so the record stays self-describing once
                # it has been written out and reloaded from JSON. The starting
                # position is needed to replay the game, and is not always the
                # standard one.
                "time_control": time_control,
                "start_fen": start_fen,
                "eval_cp": _eval_cp(node),
                "fen_after": board.fen(),
            }
        )

    return frames


def write_frames(frames: list[dict], game_id: str, frames_dir: Path) -> Path:
    """Write frames to `{frames_dir}/{game_id}.json` and return the path."""
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"{game_id}.json"
    path.write_text(json.dumps(frames, indent=1), encoding="utf-8")
    return path
