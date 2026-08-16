"""Stateful tension model turning board features into a musical parameter track.

Phase 3. Tension is a scalar in [0, 1] carried across plies, accumulating from
cramped mobility, locked and mutually-attacking pawns, king pressure and (where
present) evaluation volatility. It releases only on genuine board events — a pawn
break, material simplification, a king reaching shelter, or the end of the game.
"""

from __future__ import annotations


def tension_track(frames: list[dict]) -> list[dict]:
    """Single stateful pass over frames, emitting per-ply musical parameters."""
    raise NotImplementedError("Phase 3")


def plot_track(track: list[dict], out_path) -> None:
    """Plot the tension track and save it."""
    raise NotImplementedError("Phase 3")
