"""PGN parsing into a per-ply frame list.

Phase 1. Reads a PGN with python-chess built-ins, deriving clock and evaluation
data from `[%clk]` / `[%eval]` annotations where the source provides them.
"""

from __future__ import annotations

from pathlib import Path


def parse_increment(time_control: str | None) -> int:
    """Seconds of increment from a PGN TimeControl header ('600+5' -> 5)."""
    raise NotImplementedError("Phase 1")


def ingest(pgn_path: Path) -> list[dict]:
    """Parse the first game in `pgn_path` into a list of per-ply frame dicts."""
    raise NotImplementedError("Phase 1")


def write_frames(frames: list[dict], game_id: str, frames_dir: Path) -> Path:
    """Write frames to `{frames_dir}/{game_id}.json` and return the path."""
    raise NotImplementedError("Phase 1")
