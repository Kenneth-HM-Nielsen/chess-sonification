"""PGN parsing into a per-ply frame list.

Reads a PGN with python-chess built-ins, deriving clock and evaluation data from
`[%clk]` / `[%eval]` annotations where the source provides them. Games with no
clocks, no evals, or neither are all supported; missing data stays `None` rather
than being guessed at.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chess
import chess.pgn

log = logging.getLogger(__name__)

MATE_CP = 10000


def parse_increment(time_control: str | None) -> int:
    """Seconds of increment from a PGN TimeControl header.

    '600+5' -> 5, '600' -> 0, '-' or missing -> 0. Multi-period controls such as
    '40/9000+30:1800+30' take the increment of the first period.
    """
    if not time_control:
        return 0
    tc = time_control.strip()
    if tc in {"-", "?", ""}:
        return 0
    first_period = tc.split(":")[0]
    if "+" not in first_period:
        return 0
    try:
        return int(float(first_period.split("+", 1)[1]))
    except ValueError:
        log.warning("unparseable increment in TimeControl %r, assuming 0", time_control)
        return 0


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

    increment = parse_increment(game.headers.get("TimeControl"))
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

        clock = node.clock()
        baseline = last_clock[color]
        if clock is None or baseline is None:
            think_time = None
        else:
            think_time = baseline - clock + increment
            if think_time < 0:
                log.warning(
                    "%s ply %d (%s): negative think time %.1fs, clamped to 0",
                    pgn_path.name,
                    ply,
                    san,
                    think_time,
                )
                think_time = 0.0
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
                "eval_cp": _eval_cp(node),
                "fen_after": board.fen(),
            }
        )

    return frames


def coverage(frames: list[dict]) -> dict:
    """Count how many plies carry each optional signal."""
    return {
        "plies": len(frames),
        "clock": sum(f["clock_remaining"] is not None for f in frames),
        "think_time": sum(f["think_time"] is not None for f in frames),
        "eval": sum(f["eval_cp"] is not None for f in frames),
    }


def write_frames(frames: list[dict], game_id: str, frames_dir: Path) -> Path:
    """Write frames to `{frames_dir}/{game_id}.json` and return the path."""
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"{game_id}.json"
    path.write_text(json.dumps(frames, indent=1), encoding="utf-8")
    return path
