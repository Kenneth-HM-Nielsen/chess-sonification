"""Additive synthesis and convolution reverb, rendered to a mono WAV.

Phase 4. Piece type selects a harmonic spectrum and envelope, colour selects a
lowpass cutoff, destination square selects pitch, and think time sets the length
of a generated impulse response.

This module owns the whole state-to-sound mapping: register, dissonance, scale
selection and timbre are decided here from the state track that `tension` emits,
so that changing how the piece sounds never means editing the analysis layers.
"""

from __future__ import annotations

SAMPLE_RATE = 44100


def render(frames: list[dict], track: list[dict], out_path) -> None:
    """Render frames plus their tension track to a 44.1 kHz mono WAV."""
    raise NotImplementedError("Phase 4")
