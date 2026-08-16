"""Stateful reading of how a game is going, as a per-ply state track.

Phase 3. Tension is a scalar in [0, 1] carried across plies, accumulating from
locked and mutually-attacking pawns, king pressure, a stalling fifty-move counter
or repeated position, and (where present) evaluation volatility. It releases only
on genuine board events -- a pawn break, material simplification, a king reaching
shelter, a pawn move or capture resetting the grind, or the end of the game.

This module emits *state*, not sound. It has no opinion on pitch, scale, timbre
or register, imports nothing audio-related, and does not know that music is the
consumer -- adding a musical layer must not require editing anything here.
Mapping state onto sound belongs to `render`.

Mobility is deliberately absent from tension: measured over a locked King's
Indian against an open gambit it separates them by only 8.5%, because a closed
centre pushes play to the wings without reducing the legal move count. It feeds
density instead. Time pressure is likewise carried through as its own axis --
tension is a property of the position, pressure a property of the players.
"""

from __future__ import annotations


def tension_track(frames: list[dict]) -> list[dict]:
    """Single stateful pass over frames, emitting per-ply state.

    Each entry carries `tension`, `pressure_w`, `pressure_b`, `density`, `meter`,
    `cadence_strength`, `stall` and `active_layers`.
    """
    raise NotImplementedError("Phase 3")


def plot_track(track: list[dict], out_path) -> None:
    """Plot the tension track and save it."""
    raise NotImplementedError("Phase 3")
