"""The musical mapping from the state track to audio.

Everything about how a game sounds is decided in this module: pitch, scale,
register, timbre, envelopes, dissonance, timing. Nothing upstream knows that
music is the consumer, so adding a musical layer never means editing an analysis
layer.

Sub-phase 4a's palette has been deleted. It was six additive sine stacks and it
sounded like six additive sine stacks, where the target is orchestral -- each
voice aimed at an instrument family rather than at an abstract spectrum. Nothing
of the harmonic stacks is kept "in case", because a palette held in reserve is a
palette that gets reached for.

What survived is the machinery that was right independently of the spectra:

- the lowpass cutoff expressed as a multiple of the note's own fundamental,
  since a fixed frequency sits above the whole spectrum of a low note and removes
  nothing from it
- the declick fade applied *after* filtering, since the filter rings past the end
  of the envelope and fading first leaves a step that clicks once notes are summed
- the mixer, now summing stereo blocks, and the square-to-pitch mapping

Until Part B fills it the timeline is silent, and silent by construction rather
than by accident: there is no voice left to put in it.

Output is stereo, 44.1 kHz, float32.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100

# Provisional grid. Onsets come from density in B5; until then the timeline is
# one step per ply, which is what makes a longer game a longer render.
STEP_SECONDS = 0.50
FADE_SECONDS = 0.005
PEAK = 0.89

# Major scale, one degree per file. Eight files need eight degrees, so the last
# one is the octave. Tension selects the scale from B4.
MAJOR_DEGREES = (0, 2, 4, 5, 7, 9, 11, 12)

# Two ranks per octave. Eight full octaves from a usable bass would put the back
# rank above 4 kHz, where timbre stops being legible, so the span is compressed
# and the ordering of ranks preserved.
BASE_MIDI = 50
RANKS_PER_OCTAVE = 2

# Bright against dark: the colour cue, expressed as a multiple of the note's own
# fundamental rather than a fixed frequency. Measured at 1.00x separation for
# four of the six 4a voices when it was fixed, because the cutoff sat above their
# whole spectrum. Relative, the same harmonics are removed at every pitch.
CUTOFF_HARMONIC = {"w": 9.0, "b": 2.6}


def midi_for_square(square: str) -> int:
    """File to scale degree, rank to octave."""
    file_index = ord(square[0]) - ord("a")
    rank_index = int(square[1]) - 1
    octave = rank_index // RANKS_PER_OCTAVE
    return BASE_MIDI + 12 * octave + MAJOR_DEGREES[file_index]


def _frequency(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _envelope(length: int, attack: float, decay: float, sustain: float) -> np.ndarray:
    """Attack ramp into an exponential fall toward a sustain floor."""
    time = np.arange(length) / SAMPLE_RATE
    envelope = sustain + (1.0 - sustain) * np.exp(-time / max(decay, 1e-4))
    attack_samples = max(1, int(attack * SAMPLE_RATE))
    if attack_samples < length:
        envelope[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples)
    return envelope


def _declick(samples: np.ndarray) -> np.ndarray:
    """Fade both ends to zero.

    Applied after filtering, not before: the lowpass rings past the end of the
    envelope, so fading first leaves a step at the note boundary that reappears
    as a click once notes are summed.
    """
    fade = min(int(FADE_SECONDS * SAMPLE_RATE), len(samples) // 2)
    if fade > 0:
        samples[:fade] *= np.linspace(0.0, 1.0, fade)
        samples[-fade:] *= np.linspace(1.0, 0.0, fade)
    return samples


def _colour(samples: np.ndarray, colour: str, fundamental: float) -> np.ndarray:
    """Bright against dark, at a cutoff relative to the note's fundamental."""
    cutoff = CUTOFF_HARMONIC[colour] * fundamental / (SAMPLE_RATE / 2)
    sections = signal.butter(4, min(max(cutoff, 1e-4), 0.99), btype="low",
                             output="sos")
    return signal.sosfilt(sections, samples)


def _mix(blocks: list[tuple[int, np.ndarray]], total: int) -> np.ndarray:
    """Sum stereo blocks at their onsets, then normalise once to a fixed peak."""
    span = max(1, total, *(onset + len(block) for onset, block in blocks)) \
        if blocks else max(1, total)
    buffer = np.zeros((span, 2))
    for onset, block in blocks:
        buffer[onset:onset + len(block)] += block
    loudest = np.abs(buffer).max()
    if loudest > 0:
        buffer *= PEAK / loudest
    return buffer.astype(np.float32)


def render(frames: list[dict], track: dict, out_path) -> None:
    """Render frames plus their state track to a 44.1 kHz stereo WAV.

    `track` is the two-scope record from `tension.tension_track`: per-game fields
    alongside a `plies` list.

    The timeline is the right length and carries nothing, because the 4a palette
    that used to fill it has been deleted and Part B has not yet replaced it.
    """
    import soundfile

    log.info("rendering %d plies, layers available: %s",
             len(frames), ", ".join(track["active_layers"]) or "none")

    blocks: list[tuple[int, np.ndarray]] = []
    audio = _mix(blocks, int(len(frames) * STEP_SECONDS * SAMPLE_RATE))
    soundfile.write(str(out_path), audio, SAMPLE_RATE, subtype="FLOAT")
    log.info("wrote %s (%.1fs)", out_path, len(audio) / SAMPLE_RATE)
