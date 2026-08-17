"""Additive synthesis of the state track into a mono WAV.

This module owns the whole state-to-sound mapping. Piece type picks a harmonic
spectrum and an envelope, colour picks a lowpass cutoff, the destination square
picks a pitch, and later stages will take reverb, harmony and rhythm from the
track. Register, dissonance, scale selection and timbre are all decided here, so
that changing how the piece sounds never means editing an analysis layer.

Sub-phase 4a: voices only. Fixed note duration, notes in sequence, one fixed
major scale, no reverb and no tension. The point of stopping here is that six
timbres either separate by ear or they do not, and no later layer rescues them.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100

# Sub-phase 4a places notes on a fixed grid; density drives this from 4d.
STEP_SECONDS = 0.50
FADE_SECONDS = 0.005
PEAK = 0.89

# Major scale, one degree per file. Eight files need eight degrees, so the last
# one is the octave. The scale becomes tension-dependent in 4c.
MAJOR_DEGREES = (0, 2, 4, 5, 7, 9, 11, 12)

# Two ranks per octave. Eight full octaves from a usable bass would put the back
# rank above 4 kHz, where timbre stops being legible, so the span is compressed
# and the ordering of ranks preserved.
BASE_MIDI = 50
RANKS_PER_OCTAVE = 2

# Bright against dark on the same spectra: the only difference between colours.
# Expressed as a multiple of the note's own fundamental rather than a fixed
# frequency, because a fixed one sits above the whole spectrum of a low note and
# removes nothing from it -- measured at 1.00x separation for four of the six
# voices. Relative, the same harmonics are removed at every pitch.
CUTOFF_HARMONIC = {"w": 9.0, "b": 2.6}

# Per voice: harmonic (multiple, amplitude) pairs, an envelope, a register
# offset in semitones, and a level. Attack and decay are seconds; sustain is the
# fraction of peak the note settles to, so 0.0 is a pluck and 0.8 is a pad.
VOICES = {
    # Short percussive pluck, fast decay, a little upper grit.
    "P": {"harmonics": ((1, 1.0), (2, 0.45), (3, 0.28), (5, 0.10)),
          "attack": 0.002, "decay": 0.055, "sustain": 0.0,
          "duration": 0.26, "register": 12, "level": 0.55},
    # Detuned pair with a bend into pitch on the attack.
    "N": {"harmonics": ((1, 1.0), (2, 0.55), (3, 0.35), (4, 0.18)),
          "attack": 0.006, "decay": 0.20, "sustain": 0.18,
          "duration": 0.40, "register": 5, "level": 0.62,
          "detune_cents": 17.0, "bend_semitones": 1.6, "bend_seconds": 0.055},
    # Near-pure, soft attack, sustained. Carries the most upper partials it can
    # while remaining the purest of the six, because a lowpass has nothing to
    # take from a bare sine and this is the only voice colour struggles to reach.
    "B": {"harmonics": ((1, 1.0), (2, 0.32), (3, 0.20), (4, 0.11)),
          "attack": 0.075, "decay": 0.40, "sustain": 0.72,
          "duration": 0.62, "register": 0, "level": 0.58},
    # Low and blunt, odd-heavy, squarish.
    "R": {"harmonics": ((1, 1.0), (3, 0.62), (5, 0.40), (7, 0.26), (9, 0.15)),
          "attack": 0.010, "decay": 0.30, "sustain": 0.42,
          "duration": 0.52, "register": -17, "level": 0.66},
    # Full stack, every harmonic present.
    "Q": {"harmonics": ((1, 1.0), (2, 0.70), (3, 0.55), (4, 0.42), (5, 0.33),
                        (6, 0.25), (7, 0.19), (8, 0.14)),
          "attack": 0.020, "decay": 0.45, "sustain": 0.55,
          "duration": 0.70, "register": -5, "level": 0.70},
    # Hollow: odd harmonics only, third above the fundamental, quiet.
    "K": {"harmonics": ((1, 0.55), (3, 1.0), (5, 0.42), (7, 0.16)),
          "attack": 0.045, "decay": 0.35, "sustain": 0.30,
          "duration": 0.55, "register": -12, "level": 0.34},
}


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


def voice(piece: str, midi: int, colour: str) -> np.ndarray:
    """One note: additive stack, envelope, then the colour's lowpass."""
    spec = VOICES[piece]
    length = max(1, int(spec["duration"] * SAMPLE_RATE))
    time = np.arange(length) / SAMPLE_RATE
    base = _frequency(midi + spec["register"])

    bend = np.zeros(length)
    if spec.get("bend_semitones"):
        span = max(1, int(spec["bend_seconds"] * SAMPLE_RATE))
        bend[:span] = np.linspace(spec["bend_semitones"], 0.0, span)
    ratio = 2.0 ** (bend / 12.0)

    tone = np.zeros(length)
    detune = spec.get("detune_cents", 0.0)
    for multiple, amplitude in spec["harmonics"]:
        phase = 2 * np.pi * base * multiple * np.cumsum(ratio) / SAMPLE_RATE
        tone += amplitude * np.sin(phase)
        if detune:
            shifted = base * multiple * 2.0 ** (detune / 1200.0)
            tone += amplitude * np.sin(
                2 * np.pi * shifted * np.cumsum(ratio) / SAMPLE_RATE
            )
    tone /= np.abs(tone).max() or 1.0

    tone *= _envelope(length, spec["attack"], spec["decay"], spec["sustain"])
    return _declick(_colour(tone, colour, base)) * spec["level"]


def _colour(samples: np.ndarray, colour: str, fundamental: float) -> np.ndarray:
    """The only difference between the sides: bright against dark."""
    cutoff = CUTOFF_HARMONIC[colour] * fundamental / (SAMPLE_RATE / 2)
    sections = signal.butter(4, min(max(cutoff, 1e-4), 0.99), btype="low",
                             output="sos")
    return signal.sosfilt(sections, samples)


def _mix(notes: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Sum notes at their onsets, then normalise once to a fixed peak."""
    if not notes:
        return np.zeros(1, dtype=np.float32)
    total = max(onset + len(samples) for onset, samples in notes)
    buffer = np.zeros(total)
    for onset, samples in notes:
        buffer[onset:onset + len(samples)] += samples
    loudest = np.abs(buffer).max()
    if loudest > 0:
        buffer *= PEAK / loudest
    return buffer.astype(np.float32)


def render(frames: list[dict], track: dict, out_path) -> None:
    """Render frames plus their state track to a 44.1 kHz mono WAV.

    `track` is the two-scope record from `tension.tension_track`: per-game fields
    alongside a `plies` list. 4a reads only the manifest from it; the parameter
    track proper arrives from 4b.
    """
    import soundfile

    log.info("rendering %d plies, layers available: %s",
             len(frames), ", ".join(track["active_layers"]) or "none")

    notes = []
    for index, frame in enumerate(frames):
        samples = voice(frame["piece"], midi_for_square(frame["to_sq"]),
                        frame["color"])
        notes.append((int(index * STEP_SECONDS * SAMPLE_RATE), samples))

    audio = _mix(notes)
    soundfile.write(str(out_path), audio, SAMPLE_RATE, subtype="FLOAT")
    log.info("wrote %s (%.1fs)", out_path, len(audio) / SAMPLE_RATE)
