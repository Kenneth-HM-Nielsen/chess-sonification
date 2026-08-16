"""Engine-free structural features computed per ply from board state.

Phase 2. Every function here is pure: it takes a `chess.Board`, returns a dict,
and leaves the caller's board unmodified.
"""

from __future__ import annotations


def board_features(board) -> dict:
    """Structural measurements for a position: mobility, pawn structure,
    king pressure, material balance, centre occupancy."""
    raise NotImplementedError("Phase 2")


def annotate(frames: list[dict]) -> list[dict]:
    """Append board features to each frame, in place, using `fen_after`."""
    raise NotImplementedError("Phase 2")
