"""Additive synthesis and convolution reverb, rendered to a mono WAV.

Phase 4. Piece type selects a harmonic spectrum and envelope, colour selects a
lowpass cutoff, destination square selects pitch against a scale chosen by the
tension track's dissonance value, and think time sets the length of a generated
impulse response.
"""

from __future__ import annotations

SAMPLE_RATE = 44100


def render(frames: list[dict], track: list[dict], out_path) -> None:
    """Render frames plus their tension track to a 44.1 kHz mono WAV."""
    raise NotImplementedError("Phase 4")
