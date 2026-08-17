"""The musical mapping from the state track to audio.

Everything about how a game sounds is decided in this module: pitch, scale,
register, timbre, envelopes, dissonance, timing. Nothing upstream knows that
music is the consumer, so adding a musical layer never means editing an analysis
layer.

Sub-phase 4a's palette of six additive sine stacks has been deleted, and what
survived it is the machinery that was right independently of the spectra: a
lowpass cutoff expressed as a multiple of the note's own fundamental, the declick
fade applied after filtering rather than before, the stereo mixer, and the
square-to-pitch mapping.

Percussion leads, because it is the part of an orchestra that synthesises
convincingly from numpy. A cymbal is bandpassed noise with a fast attack and a
ragged decay; a timpano is a handful of inharmonic membrane modes over a stick
transient. The sustained voices -- the ones that stay recognisably synthetic
however long they are worked on -- come after.

Two events:

- a capture rings a cymbal scaled by what came off the board, so a pawn is a
  shimmer and a queen is a crash
- accumulation rolls a timpano, entering as the stall term climbs and swelling
  toward the fifty-move threshold, pitched to the tonal centre so that it belongs
  harmonically

Neither has a schedule hidden inside it. The roll stops when the stall term
resets, which is the board's business and not the renderer's.

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

# Stereo position per colour, deliberately redundant with the cutoff. Pan carries
# colour and nothing else -- not piece, not file, not which side of the board a
# move happened on -- so that a listener who works out what the width means never
# has to unlearn it.
PAN_COLOUR = {"w": -0.4, "b": 0.4}

# --- captures ---------------------------------------------------------------
# A cymbal, scaled by the value of the piece that came off the board. The ends of
# the scale are a pawn and a queen, so every piece lands somewhere real inside the
# interpolation rather than clustering at one end of it.
CYMBAL_VALUE_RANGE = (1.0, 9.0)
CYMBAL_DECAY_S = (0.30, 1.80)
# Buffer length as a multiple of the decay constant: at 2.4 the tail is down to 9%
# of peak and the declick takes the rest.
CYMBAL_TAIL = 2.4
CYMBAL_ATTACK_S = 0.0015
# The band's lower edge falls as the value rises, because a crash has body that a
# shimmer has not. The upper edge is the colour cue, and it is deliberately the
# smaller of the two effects: a dark queen must still be bigger than a bright
# pawn, or the same capture answers the gate's question differently depending on
# who made it. Measured at 5.4x in level for the worst pairing.
CYMBAL_LOW_HZ = (3800.0, 350.0)
CYMBAL_TOP_HZ = {"w": 15000.0, "b": 9000.0}
CYMBAL_LEVEL = (0.18, 1.00)
# Noise is scaled by its power and a fixed crest factor, not by its loudest
# sample. The loudest sample of a four-second noise draw lands wherever the draw
# puts it, so normalising to it made how big a capture sounds vary by 16% between
# draws -- and how big a capture sounds is the one thing this event is for.
CYMBAL_CREST = 4.2
# Depth and rate of the slow wander that keeps the decay off a clean exponential.
# A cymbal's tail beats against itself; an exponential does not, and reads as a
# synthesised sweep. Ramped in behind the strike -- see `cymbal`.
CYMBAL_RAGGED = 0.45
CYMBAL_RAGGED_HZ = 11.0
CYMBAL_RAGGED_ONSET_S = 0.06

# --- accumulation -----------------------------------------------------------
# (frequency ratio, amplitude, decay share) per membrane mode. A kettledrum's
# modes are inharmonic, but tuned far closer to a harmonic series than an ideal
# circular membrane's, which is what gives the drum a definite pitch at all.
# Higher modes die faster, which is what makes the attack brighter than the tail.
TIMPANI_MODES = (
    (1.00, 1.00, 1.00),
    (1.50, 0.62, 0.68),
    (1.99, 0.44, 0.48),
    (2.44, 0.30, 0.34),
    (2.89, 0.20, 0.24),
)
TIMPANI_DECAY_S = 0.40
TIMPANI_TAIL = 3.0
TIMPANI_STICK_S = 0.015
TIMPANI_STICK_HZ = 3000.0
TIMPANI_STICK_LEVEL = 0.70

# The roll is pitched to the tonal centre, in a register a timpano actually
# occupies. BASE_MIDI is the root of the scale, so the drum will follow the key
# once B3 starts choosing one. An octave lower is also a real timpani note and was
# measured with only 38% of the roll's energy surviving a 150 Hz highpass, against
# 77% here -- on a small speaker that build would not be there at all.
TIMPANI_MIDI = BASE_MIDI

# Below this the roll would be inaudible, so it is not synthesised. Not a schedule
# and not a period boundary: the term it reads resets when the board resets it,
# and the roll stops there because there is nothing left to play.
TIMPANI_FLOOR = 0.02
TIMPANI_LEVEL = 0.13
# Strokes per second across the tension range. A roll is a rate rather than a
# note, and a tenser position rolls tighter.
TIMPANI_RATE_HZ = (9.0, 18.0)
# A roll played on a grid sounds like a machine. Both jitters are fractions.
TIMPANI_JITTER_TIME = 0.12
TIMPANI_JITTER_LEVEL = 0.18

# Percussion is noise-based, so it needs a generator, and a render must still come
# out the same twice. One seed per render, consumed in ply order.
RNG_SEED = 20260817


def midi_for_square(square: str) -> int:
    """File to scale degree, rank to octave."""
    file_index = ord(square[0]) - ord("a")
    rank_index = int(square[1]) - 1
    octave = rank_index // RANKS_PER_OCTAVE
    return BASE_MIDI + 12 * octave + MAJOR_DEGREES[file_index]


def _frequency(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _lerp(pair: tuple[float, float], fraction: float) -> float:
    return pair[0] + (pair[1] - pair[0]) * fraction


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _envelope(length: int, attack: float, decay: float, sustain: float) -> np.ndarray:
    """Attack ramp into an exponential fall toward a sustain floor."""
    time = np.arange(length) / SAMPLE_RATE
    envelope = sustain + (1.0 - sustain) * np.exp(-time / max(decay, 1e-4))
    attack_samples = max(1, int(attack * SAMPLE_RATE))
    if attack_samples < length:
        envelope[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples)
    return envelope


def _require_mono(samples: np.ndarray, what: str) -> None:
    """The per-note primitives are mono-only, and they fail badly on a block.

    `_declick`'s fade broadcasts against the channel axis and raises, which is
    survivable. `signal.sosfilt` filters the last axis by default, so a stereo
    block would be filtered *across its two channels* instead of along time and
    come out quietly wrong with no error at all. Hence a guard rather than
    axis-handling: the pipeline is synthesise mono, filter, fade, and pan last,
    and anything arriving here two-dimensional has that order wrong.
    """
    if samples.ndim != 1:
        raise ValueError(
            f"{what} takes a mono block, got shape {samples.shape}; filter and "
            "fade in mono and pan last"
        )


def _declick(samples: np.ndarray) -> np.ndarray:
    """Fade both ends of a copy to zero.

    Applied after filtering, not before: the lowpass rings past the end of the
    envelope, so fading first leaves a step at the note boundary that reappears
    as a click once notes are summed.

    Returns a new array. A fade applied in place to a shared or cached block
    fades it twice, and the second fade is inaudible until it is not.
    """
    _require_mono(samples, "_declick")
    faded = samples.copy()
    fade = min(int(FADE_SECONDS * SAMPLE_RATE), len(faded) // 2)
    if fade > 0:
        faded[:fade] *= np.linspace(0.0, 1.0, fade)
        faded[-fade:] *= np.linspace(1.0, 0.0, fade)
    return faded


def _colour(samples: np.ndarray, colour: str, fundamental: float) -> np.ndarray:
    """Bright against dark, at a cutoff relative to the note's fundamental."""
    _require_mono(samples, "_colour")
    cutoff = CUTOFF_HARMONIC[colour] * fundamental / (SAMPLE_RATE / 2)
    sections = signal.butter(4, min(max(cutoff, 1e-4), 0.99), btype="low",
                             output="sos")
    return signal.sosfilt(sections, samples)


def _pan(samples: np.ndarray, position: float) -> np.ndarray:
    """Mono to stereo at `position` in [-1, 1], at constant power.

    Both channels are positive scalings of the same signal, so summing to mono
    cancels nothing however many blocks overlap. Nothing here inverts a phase and
    nothing should: a stereo width that disappears on a phone was carrying
    information it did not own.
    """
    _require_mono(samples, "_pan")
    angle = (position + 1.0) * np.pi / 4.0
    return np.column_stack((samples * np.cos(angle), samples * np.sin(angle)))


def _bandpass(samples: np.ndarray, low: float, high: float) -> np.ndarray:
    _require_mono(samples, "_bandpass")
    nyquist = SAMPLE_RATE / 2
    edges = [min(max(low / nyquist, 1e-4), 0.99),
             min(max(high / nyquist, 2e-4), 0.999)]
    sections = signal.butter(4, edges, btype="band", output="sos")
    return signal.sosfilt(sections, samples)


def cymbal(value: float, colour: str, rng: np.random.Generator) -> np.ndarray:
    """A capture, as a cymbal scaled by what came off the board.

    Bandpassed noise, near-instant attack, decay proportional to the captured
    value, panned by the colour of the side that made the capture. Returns a
    stereo block.
    """
    fraction = _clamp(
        (value - CYMBAL_VALUE_RANGE[0])
        / (CYMBAL_VALUE_RANGE[1] - CYMBAL_VALUE_RANGE[0])
    )
    decay = _lerp(CYMBAL_DECAY_S, fraction)
    length = max(2, int(CYMBAL_TAIL * decay * SAMPLE_RATE))

    tone = _bandpass(rng.standard_normal(length),
                     _lerp(CYMBAL_LOW_HZ, fraction), CYMBAL_TOP_HZ[colour])
    tone /= (CYMBAL_CREST * float(np.sqrt((tone ** 2).mean()))) or 1.0

    # Slow noise rather than a sine: a periodic wobble is a tremolo, and a
    # cymbal's tail is not periodic.
    wander = signal.sosfilt(
        signal.butter(2, CYMBAL_RAGGED_HZ / (SAMPLE_RATE / 2), output="sos"),
        rng.standard_normal(length),
    )
    wander /= np.abs(wander).max() or 1.0

    # The strike itself is left clean and the wander ramps in behind it. Wandering
    # over the attack would make how big a capture sounds depend on which way the
    # noise happened to fall, and how big it sounds is the whole point.
    time = np.arange(length) / SAMPLE_RATE
    ramp = np.minimum(1.0, time / CYMBAL_RAGGED_ONSET_S)
    tone *= _envelope(length, CYMBAL_ATTACK_S, decay, 0.0)
    tone *= 1.0 - CYMBAL_RAGGED * (0.5 + 0.5 * wander) * ramp
    return _pan(_declick(tone) * _lerp(CYMBAL_LEVEL, fraction), PAN_COLOUR[colour])


def timpani(midi: int, rng: np.random.Generator) -> np.ndarray:
    """One timpani stroke, mono, normalised to unit peak.

    A few inharmonic membrane modes with a decay each, over a short noise
    transient for the stick. Mode phases are randomised so that overlapping
    strokes in a roll sum into a rumble rather than into one louder tone.
    """
    length = max(2, int(TIMPANI_TAIL * TIMPANI_DECAY_S * SAMPLE_RATE))
    time = np.arange(length) / SAMPLE_RATE
    fundamental = _frequency(midi)

    membrane = np.zeros(length)
    for ratio, amplitude, share in TIMPANI_MODES:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        membrane += amplitude * np.sin(
            2 * np.pi * fundamental * ratio * time + phase
        ) * _envelope(length, 0.0015, TIMPANI_DECAY_S * share, 0.0)
    membrane /= np.abs(membrane).max() or 1.0

    stick = np.zeros(length)
    burst = min(length, max(1, int(TIMPANI_STICK_S * SAMPLE_RATE)))
    hit = signal.sosfilt(
        signal.butter(2, TIMPANI_STICK_HZ / (SAMPLE_RATE / 2), output="sos"),
        rng.standard_normal(burst),
    )
    stick[:burst] = hit / (np.abs(hit).max() or 1.0) * np.linspace(1.0, 0.0, burst) ** 2

    stroke = membrane + TIMPANI_STICK_LEVEL * stick
    return _declick(stroke / (np.abs(stroke).max() or 1.0))


def _capture_blocks(frame: dict, onset: int,
                    rng: np.random.Generator) -> list[tuple[int, np.ndarray]]:
    """The cymbal for this ply, if it took anything off the board."""
    value = frame["captured_value"]
    if not value:
        return []
    return [(onset, cymbal(value, frame["color"], rng))]


def _roll_blocks(ply: dict, onset: int,
                 rng: np.random.Generator) -> list[tuple[int, np.ndarray]]:
    """The timpani strokes falling inside this ply's step.

    Loudness follows the stall term directly. That term is already squared toward
    the fifty-move threshold in the analysis layer, and squaring it again here
    would be the renderer second-guessing a calibrated input. Rate follows
    tension, so a tenser position rolls tighter.

    Every stroke lands inside the step that asked for it, which is what lets the
    roll stop dead on a reset instead of ringing on over it.

    Centred, and with no colour cue: the drum is not one side's, and pan means
    colour.
    """
    intensity = ply["stall"]
    if intensity < TIMPANI_FLOOR:
        return []

    rate = _lerp(TIMPANI_RATE_HZ, _clamp(ply["tension"]))
    strokes = max(1, int(round(STEP_SECONDS * rate)))
    blocks = []
    for index in range(strokes):
        jitter = rng.uniform(-TIMPANI_JITTER_TIME, TIMPANI_JITTER_TIME)
        at = onset + int((index + jitter) / rate * SAMPLE_RATE)
        level = TIMPANI_LEVEL * intensity * (
            1.0 + rng.uniform(-TIMPANI_JITTER_LEVEL, TIMPANI_JITTER_LEVEL)
        )
        blocks.append((max(0, at), _pan(timpani(TIMPANI_MIDI, rng) * level, 0.0)))
    return blocks


def percussion(frames: list[dict], plies: list[dict]) -> list[tuple[int, np.ndarray]]:
    """Every percussion block for a game, at its onset on the provisional grid."""
    if len(frames) != len(plies):
        raise ValueError(
            f"{len(frames)} frames against {len(plies)} track entries; the track "
            "must be the one computed from these frames"
        )
    rng = np.random.default_rng(RNG_SEED)
    blocks: list[tuple[int, np.ndarray]] = []
    for index, (frame, ply) in enumerate(zip(frames, plies)):
        onset = int(index * STEP_SECONDS * SAMPLE_RATE)
        blocks.extend(_capture_blocks(frame, onset, rng))
        blocks.extend(_roll_blocks(ply, onset, rng))
    return blocks


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
    alongside a `plies` list. B1 reads `captured_value` and `color` off the
    frames and `stall` and `tension` off the track; the piece voices come in B2.
    """
    import soundfile

    log.info("rendering %d plies, layers available: %s",
             len(frames), ", ".join(track["active_layers"]) or "none")

    blocks = percussion(frames, track["plies"])
    audio = _mix(blocks, int(len(frames) * STEP_SECONDS * SAMPLE_RATE))
    soundfile.write(str(out_path), audio, SAMPLE_RATE, subtype="FLOAT")
    log.info("wrote %s (%.1fs, %d percussion events)",
             out_path, len(audio) / SAMPLE_RATE, len(blocks))
