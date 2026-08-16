"""PGN construction for tests.

Tests build their own games rather than reading `data/pgn/`, which is gitignored
and therefore absent from a fresh clone.
"""

from __future__ import annotations

import pathlib
import tempfile

SHUFFLE = ("Nf3", "Nf6", "Ng1", "Ng8")


def clock_text(seconds: float) -> str:
    """Seconds as a `[%clk]` value, to hundredths.

    Rounding is done first, in integer hundredths, so that a value such as 599.95
    cannot be decomposed into an impossible '0:09:60.0' -- which python-chess
    reads back as 600, silently destroying sub-second precision.
    """
    hundredths = int(round(seconds * 100))
    whole, fraction = divmod(hundredths, 100)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    stem = f"{hours}:{minutes:02d}:{secs:02d}"
    if fraction == 0:
        return stem
    if fraction % 10 == 0:
        return f"{stem}.{fraction // 10}"
    return f"{stem}.{fraction:02d}"


def shuffle_moves(plies: int) -> list[str]:
    """A legal, endlessly repeatable knight shuffle."""
    return [SHUFFLE[i % 4] for i in range(plies)]


def build_pgn(
    sans: list[str],
    clocks: list[float | None] | None = None,
    evals: list[str | None] | None = None,
    time_control: str | None = None,
    fen: str | None = None,
    result: str = "*",
) -> str:
    lines = ['[Event "Test"]']
    if time_control is not None:
        lines.append(f'[TimeControl "{time_control}"]')
    if fen is not None:
        lines.append('[SetUp "1"]')
        lines.append(f'[FEN "{fen}"]')
    lines.append("")

    tokens: list[str] = []
    for index, san in enumerate(sans):
        if index % 2 == 0:
            tokens.append(f"{index // 2 + 1}.")
        annotations = []
        if evals is not None and evals[index] is not None:
            annotations.append(f"[%eval {evals[index]}]")
        if clocks is not None and clocks[index] is not None:
            annotations.append(f"[%clk {clock_text(clocks[index])}]")
        tokens.append(san + (" { " + " ".join(annotations) + " }" if annotations else ""))
    tokens.append(result)
    lines.append(" ".join(tokens))
    return "\n".join(lines) + "\n"


def write_pgn(text: str) -> pathlib.Path:
    path = pathlib.Path(tempfile.mkdtemp()) / "game.pgn"
    path.write_text(text, encoding="utf-8")
    return path


def clocks_from_think(
    think_times: list[float], start: float, increment: float = 0.0
) -> list[float]:
    """Per-ply clock readings that produce the given think times for both sides."""
    remaining = {0: start, 1: start}
    readings = []
    for index, spent in enumerate(think_times):
        side = index % 2
        remaining[side] = remaining[side] - spent + increment
        readings.append(remaining[side])
    return readings
