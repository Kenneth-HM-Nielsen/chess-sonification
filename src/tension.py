"""Stateful tension model turning board features into a musical parameter track.

Phase 3. Tension is a scalar in [0, 1] carried across plies, accumulating from
locked and mutually-attacking pawns, king pressure, a stalling fifty-move counter
or repeated position, and (where present) evaluation volatility. It releases only
on genuine board events -- a pawn break, material simplification, a king reaching
shelter, a pawn move or capture resetting the grind, or the end of the game.

Mobility is deliberately absent here: measured over a locked King's Indian
against an open gambit it separates them by only 8.5%, because a closed centre
pushes play to the wings without reducing the legal move count. It drives density
instead. Time pressure is likewise a separate input rather than a tension term --
tension is a property of the position, pressure a property of the players.
"""

from __future__ import annotations


def tension_track(frames: list[dict]) -> list[dict]:
    """Single stateful pass over frames, emitting per-ply musical parameters."""
    raise NotImplementedError("Phase 3")


def plot_track(track: list[dict], out_path) -> None:
    """Plot the tension track and save it."""
    raise NotImplementedError("Phase 3")
