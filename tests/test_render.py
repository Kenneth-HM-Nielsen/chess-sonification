"""The renderer's surviving primitives, measured rather than asserted.

Sub-phase 4a's six additive voices are gone, and with them the assertions about
their spectra. What is measured here is what outlived them: a lowpass whose
cutoff tracks the note rather than sitting at a fixed frequency, a fade applied
after that filter rather than before it, the envelope shape, and the mixer.

The measurement harness is the one written for the voices -- spectral centroid
for brightness, spectral spread for how much of the spectrum survives, an
envelope follower for attack and decay. It is kept because the separation floors
it enforces are what tell six instrument families apart in B2, and it is pointed
at the primitives until there are voices to point it at again.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np

from src import render

PROBE_SECONDS = 0.25
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"

# Pawn, knight, rook, queen: the rungs of the capture scale. Every one of them
# occurs in tests/fixtures/tactical_decisive.pgn, which is asserted below rather
# than assumed.
VALUES = (1, 3, 5, 9)


def harmonic_probe(fundamental: float, seconds: float = PROBE_SECONDS) -> np.ndarray:
    """A test signal with every harmonic up to Nyquist, at equal-ish weight.

    A probe, not a voice: the point is to hand the filter something it can
    demonstrably take harmonics away from, so that what is being measured is the
    filter and not the choice of source spectrum.
    """
    length = int(seconds * render.SAMPLE_RATE)
    time = np.arange(length) / render.SAMPLE_RATE
    tone = np.zeros(length)
    harmonic = 1
    while fundamental * harmonic < render.SAMPLE_RATE / 2:
        tone += np.sin(2 * np.pi * fundamental * harmonic * time) / harmonic
        harmonic += 1
    return tone / np.abs(tone).max()


def spectrum(samples):
    windowed = samples * np.hanning(len(samples))
    power = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1 / render.SAMPLE_RATE)
    return freqs, power


def centroid(samples):
    freqs, power = spectrum(samples)
    return float((freqs * power).sum() / power.sum()) if power.sum() else 0.0


def spread(samples):
    freqs, power = spectrum(samples)
    mid = centroid(samples)
    return float(np.sqrt(((freqs - mid) ** 2 * power).sum() / power.sum()))


def envelope(samples, window=0.004):
    width = max(1, int(window * render.SAMPLE_RATE))
    return np.convolve(np.abs(samples), np.ones(width) / width, mode="same")


def shape(samples):
    """(attack seconds, seconds to fall to a fifth of peak, rms)."""
    follower = envelope(samples)
    peak_at = int(np.argmax(follower))
    below = np.where(follower[peak_at:] < 0.2 * follower.max())[0]
    decay = below[0] / render.SAMPLE_RATE if len(below) else len(samples) / render.SAMPLE_RATE
    return (peak_at / render.SAMPLE_RATE, decay,
            float(np.sqrt((samples ** 2).mean())))


def rms(samples):
    return float(np.sqrt((samples ** 2).mean()))


def sine(hz, seconds=0.2):
    time = np.arange(int(seconds * render.SAMPLE_RATE)) / render.SAMPLE_RATE
    return np.sin(2 * np.pi * hz * time)


def mono(block):
    """A stereo block as one channel, for measuring its spectrum."""
    return block.sum(axis=1)


def level(samples):
    """Peak of the smoothed envelope.

    Used instead of the raw sample maximum wherever two noise-based blocks are
    compared: the loudest single sample of filtered noise lands wherever the draw
    puts it, and the assertion would then be about the draw.
    """
    return float(envelope(samples).max())


def audible_seconds(samples, fraction=0.05):
    """How long a block stays above `fraction` of its own peak."""
    follower = envelope(np.abs(samples), window=0.01)
    quiet = np.where(follower < fraction * follower.max())[0]
    quiet = quiet[quiet > int(np.argmax(follower))]
    return (quiet[0] if len(quiet) else len(samples)) / render.SAMPLE_RATE


def energy(samples):
    return float((samples ** 2).sum())


def generator():
    return np.random.default_rng(render.RNG_SEED)


class Colour(unittest.TestCase):
    def test_the_cutoff_follows_the_note_rather_than_being_fixed(self):
        """A fixed cutoff sits above a low note's whole spectrum and does nothing.

        Measured at 1.00x separation for four of six voices when the cutoff was a
        fixed frequency, so the separation has to hold at the bottom of the board
        as well as the top.
        """
        for midi in (38, 62, 92):
            with self.subTest(midi=midi):
                hz = render._frequency(midi)
                probe = harmonic_probe(hz)
                bright = centroid(render._colour(probe, "w", hz))
                dark = centroid(render._colour(probe, "b", hz))
                self.assertGreater(bright / dark, 1.2)

    def test_dark_keeps_less_of_the_spectrum_than_bright(self):
        hz = render._frequency(62)
        probe = harmonic_probe(hz)
        self.assertLess(spread(render._colour(probe, "b", hz)),
                        spread(render._colour(probe, "w", hz)))

    def test_both_colours_are_defined_and_ordered(self):
        self.assertEqual(render.CUTOFF_HARMONIC, {"w": 9.0, "b": 2.6})
        self.assertGreater(render.CUTOFF_HARMONIC["w"],
                           render.CUTOFF_HARMONIC["b"])

    def test_the_lowpass_is_steep_enough_to_remove_what_it_passes_over(self):
        """Pins the filter order, which nothing else does.

        A fourth-order Butterworth is down about 24 dB an octave above its
        cutoff; a first-order one is down 7 dB and would leave the harmonics it
        claims to remove clearly present.
        """
        fundamental = 200.0
        above = 2.0 * render.CUTOFF_HARMONIC["b"] * fundamental
        tone = sine(above)
        filtered = render._colour(tone, "b", fundamental)
        half = len(tone) // 2      # past the filter's own transient
        self.assertGreater(rms(tone[half:]) / rms(filtered[half:]), 8.0)


class FadeOrder(unittest.TestCase):
    """The rule is fade *after* filtering. Pinned by consequence, not by comment."""

    def test_filtering_after_the_fade_leaves_a_step_at_the_note_end(self):
        hz = render._frequency(50)
        probe = harmonic_probe(hz)
        correct = render._declick(render._colour(probe, "b", hz))
        reversed_order = render._colour(render._declick(probe), "b", hz)
        self.assertLess(abs(float(correct[-1])), 1e-9)
        self.assertGreater(abs(float(reversed_order[-1])),
                           abs(float(correct[-1])) + 1e-4)

    def test_both_ends_are_faded(self):
        """On a constant block, so that a missing fade cannot hide in the signal.

        A probe built from sines all at phase zero starts at exactly 0.0, which
        made the assertion about the *leading* fade unfailable: deleting that
        fade left the test green.
        """
        block = np.ones(int(0.05 * render.SAMPLE_RATE))
        faded = render._declick(block)
        self.assertLess(float(faded[0]), 1e-9)
        self.assertLess(float(faded[-1]), 1e-9)
        self.assertEqual(float(faded[len(faded) // 2]), 1.0)

    def test_the_fade_is_short_enough_to_be_a_declick_and_not_an_envelope(self):
        """Pinned to a literal, and to a consequence measured at a fixed offset.

        Taking the offset from the constant instead would move the measurement
        with it, and a half-second fade in and out of every note would pass.
        """
        self.assertEqual(render.FADE_SECONDS, 0.005)
        block = np.ones(int(0.10 * render.SAMPLE_RATE))
        faded = render._declick(block)
        at = int(0.02 * render.SAMPLE_RATE)
        self.assertEqual(float(faded[at]), 1.0)
        self.assertEqual(float(faded[-at]), 1.0)

    def test_the_fade_leaves_the_caller_s_array_alone(self):
        block = np.ones(1000)
        render._declick(block)
        self.assertEqual(float(block[0]), 1.0)
        self.assertEqual(float(block[-1]), 1.0)

    def test_a_block_shorter_than_two_fades_is_still_faded(self):
        """The clamp on the fade length, which nothing else reaches."""
        short = np.ones(11)
        faded = render._declick(short)
        self.assertEqual(len(faded), 11)
        self.assertLess(float(faded[0]), 1e-9)
        self.assertLess(float(faded[-1]), 1e-9)

    def test_the_per_note_primitives_refuse_a_stereo_block(self):
        """A stereo block through `sosfilt` is filtered across the channels.

        It comes out wrong and raises nothing, so the guard is the only thing
        between that and a note that is quietly mangled.
        """
        stereo = np.ones((100, 2))
        with self.assertRaises(ValueError):
            render._declick(stereo)
        with self.assertRaises(ValueError):
            render._colour(stereo, "w", 200.0)


class Envelope(unittest.TestCase):
    def test_a_pluck_dies_away_and_a_pad_holds(self):
        length = int(0.6 * render.SAMPLE_RATE)
        pluck = render._envelope(length, attack=0.002, decay=0.05, sustain=0.0)
        pad = render._envelope(length, attack=0.20, decay=0.40, sustain=0.8)
        self.assertLess(shape(pluck)[1], shape(pad)[1])
        self.assertGreater(shape(pad)[0], shape(pluck)[0])
        self.assertLess(pluck[-1], 0.01)
        self.assertGreater(pad[-1], 0.7)

    def test_the_envelope_starts_from_silence(self):
        env = render._envelope(int(0.3 * render.SAMPLE_RATE), 0.01, 0.2, 0.5)
        self.assertEqual(env[0], 0.0)
        self.assertLessEqual(float(env.max()), 1.0)


class Pan(unittest.TestCase):
    def test_pan_is_symmetric_and_carries_colour_only(self):
        self.assertEqual(render.PAN_COLOUR, {"w": -0.4, "b": 0.4})
        self.assertAlmostEqual(abs(render.PAN_COLOUR["w"]),
                               abs(render.PAN_COLOUR["b"]))

    def test_a_pan_puts_more_signal_on_the_side_it_names(self):
        probe = harmonic_probe(220.0)
        left = render._pan(probe, render.PAN_COLOUR["w"])
        right = render._pan(probe, render.PAN_COLOUR["b"])
        self.assertGreater(rms(left[:, 0]), rms(left[:, 1]))
        self.assertLess(rms(right[:, 0]), rms(right[:, 1]))

    def test_nothing_cancels_when_summed_to_mono(self):
        """A width that vanishes on a phone was carrying what it did not own.

        Catches a phase inversion in the pan law, which leaves the stereo image
        intact and the mono downmix empty.
        """
        probe = harmonic_probe(220.0)
        for position in (-1.0, render.PAN_COLOUR["w"], 0.0,
                         render.PAN_COLOUR["b"], 1.0):
            with self.subTest(position=position):
                self.assertGreater(rms(mono(render._pan(probe, position))),
                                   0.9 * rms(probe))

    def test_a_pan_is_constant_power(self):
        probe = harmonic_probe(220.0)
        powers = [energy(render._pan(probe, p)) for p in (-1.0, -0.4, 0.0, 0.4, 1.0)]
        for power in powers[1:]:
            self.assertAlmostEqual(power / powers[0], 1.0, places=6)

    def test_centre_is_the_only_position_with_equal_channels(self):
        probe = harmonic_probe(220.0)
        centred = render._pan(probe, 0.0)
        self.assertAlmostEqual(rms(centred[:, 0]), rms(centred[:, 1]), places=9)


class Cymbal(unittest.TestCase):
    def test_a_queen_is_unmistakably_bigger_than_a_pawn(self):
        """The B1 gate, as far as measurement can carry it.

        Three independent quantities, because any one of them alone could be got
        by a constant: how loud the strike is, how long it rings, and how much
        energy there is in it altogether. The floors sit well under what is
        measured -- 4.5x, 6x and 79x -- so the constants can be retuned at the
        gate, but flattening any of the three mappings fails here.
        """
        rng = generator()
        pawn = mono(render.cymbal(1, "w", rng))
        queen = mono(render.cymbal(9, "w", rng))
        self.assertGreater(level(queen) / level(pawn), 3.0)
        self.assertGreater(audible_seconds(queen) / audible_seconds(pawn), 4.0)
        self.assertGreater(energy(queen) / energy(pawn), 20.0)

    def test_every_rung_of_the_scale_is_bigger_than_the_one_below(self):
        rng = generator()
        blocks = [mono(render.cymbal(value, "w", rng)) for value in VALUES]
        levels = [level(block) for block in blocks]
        lengths = [audible_seconds(block) for block in blocks]
        self.assertEqual(levels, sorted(levels))
        self.assertEqual(lengths, sorted(lengths))
        self.assertGreater(min(b / a for a, b in zip(levels, levels[1:])), 1.05)

    def test_a_bigger_capture_has_more_body_lower_down(self):
        """A crash has weight a shimmer has not, so the band opens downward."""
        rng = generator()
        self.assertLess(centroid(mono(render.cymbal(9, "w", rng))),
                        centroid(mono(render.cymbal(1, "w", rng))))

    def test_colour_does_not_reorder_the_values(self):
        """The colour cue is deliberately the smaller of the two effects.

        A dark queen must still be bigger than a bright pawn, or the gate's
        question -- is a queen bigger than a pawn -- gets a different answer
        depending on who captured.
        """
        rng = generator()
        for queen_colour in ("w", "b"):
            for pawn_colour in ("w", "b"):
                with self.subTest(queen=queen_colour, pawn=pawn_colour):
                    queen = mono(render.cymbal(9, queen_colour, rng))
                    pawn = mono(render.cymbal(1, pawn_colour, rng))
                    self.assertGreater(level(queen) / level(pawn), 3.0)
                    self.assertGreater(audible_seconds(queen),
                                       audible_seconds(pawn))

    def test_colour_is_audible_on_a_capture_too(self):
        rng = generator()
        for value in VALUES:
            with self.subTest(value=value):
                bright = centroid(mono(render.cymbal(value, "w", rng)))
                dark = centroid(mono(render.cymbal(value, "b", rng)))
                self.assertGreater(bright / dark, 1.2)

    def test_a_capture_is_panned_by_the_colour_that_made_it(self):
        rng = generator()
        white = render.cymbal(5, "w", rng)
        black = render.cymbal(5, "b", rng)
        self.assertGreater(rms(white[:, 0]), rms(white[:, 1]))
        self.assertLess(rms(black[:, 0]), rms(black[:, 1]))

    def test_the_attack_is_immediate(self):
        """A struck cymbal, not one swelled into.

        Measured on the envelope near the start rather than on where its peak
        lands: the follower's own 4 ms window and the wander ramping in behind the
        strike both move the peak a few milliseconds later without making the
        attack any slower.
        """
        rng = generator()
        block = mono(render.cymbal(9, "w", rng))
        follower = envelope(block)
        at_3ms = float(follower[int(0.003 * render.SAMPLE_RATE)])
        self.assertGreater(at_3ms / float(follower.max()), 0.4)
        self.assertLess(shape(block)[0], 0.025)

    def test_the_decay_is_ragged_rather_than_a_clean_exponential(self):
        """An exponential tail reads as a synthesised sweep, not as a cymbal."""
        rng = generator()
        tail = envelope(mono(render.cymbal(9, "w", rng)),
                        window=0.02)[int(0.25 * render.SAMPLE_RATE):]
        rises = int((np.diff(tail) > 0).sum())
        self.assertGreater(rises / len(tail), 0.05)

    def test_the_strike_itself_is_not_left_to_the_noise_draw(self):
        """How big a capture sounds must not depend on the random draw.

        The wander ramps in behind the attack for exactly this reason, so the
        same value struck twice comes out at the same level.
        """
        levels = [level(mono(render.cymbal(9, "w", np.random.default_rng(seed))))
                  for seed in range(8)]
        # 10% across eight draws, which is 0.9 dB. Normalising the noise to its
        # loudest sample instead of its power gave 16%, and the residue here is
        # the wander ramping in over the envelope's peak.
        self.assertLess((max(levels) - min(levels)) / min(levels), 0.15)

    def test_a_cymbal_starts_and_ends_at_silence(self):
        rng = generator()
        for value in VALUES:
            with self.subTest(value=value):
                block = render.cymbal(value, "w", rng)
                self.assertLess(float(np.abs(block[0]).max()), 1e-9)
                self.assertLess(float(np.abs(block[-1]).max()), 1e-6)

    def test_a_value_off_the_end_of_the_scale_is_clamped(self):
        """Promotion can put more than one queen on the board.

        The clamp is what stops a 27-point capture asking for a nine-second
        buffer at a level above one.
        """
        rng = generator()
        biggest = mono(render.cymbal(27, "w", rng))
        queen = mono(render.cymbal(9, "w", rng))
        self.assertEqual(len(biggest), len(queen))
        self.assertAlmostEqual(level(biggest) / level(queen), 1.0, delta=0.15)


class Timpani(unittest.TestCase):
    def test_the_stroke_is_pitched_to_the_tonal_centre(self):
        rng = generator()
        freqs, power = spectrum(render.timpani(render.TIMPANI_MIDI, rng))
        peak = float(freqs[int(np.argmax(power))])
        self.assertAlmostEqual(peak / render._frequency(render.TIMPANI_MIDI),
                               1.0, delta=0.03)

    def test_the_drum_follows_the_tonal_centre_rather_than_a_fixed_pitch(self):
        """B3 chooses a key, and the drum has to move with it."""
        self.assertEqual(render.TIMPANI_MIDI, render.BASE_MIDI)
        rng = generator()
        for midi in (render.BASE_MIDI, render.BASE_MIDI + 5):
            with self.subTest(midi=midi):
                freqs, power = spectrum(render.timpani(midi, rng))
                peak = float(freqs[int(np.argmax(power))])
                self.assertAlmostEqual(peak / render._frequency(midi),
                                       1.0, delta=0.03)

    def test_the_modes_are_inharmonic(self):
        """A membrane, not a harmonic stack: there is a partial at 1.5x.

        That is what makes it a drum rather than an organ pipe, so it is asserted
        against the neighbouring frequencies, where a harmonic series has nothing.
        """
        rng = generator()
        freqs, power = spectrum(render.timpani(render.TIMPANI_MIDI, rng))
        fundamental = render._frequency(render.TIMPANI_MIDI)

        def near(ratio, width=0.04):
            band = ((freqs > fundamental * ratio * (1 - width))
                    & (freqs < fundamental * ratio * (1 + width)))
            return float(power[band].sum())

        self.assertGreater(near(1.50), near(1.25) * 10)
        self.assertGreater(near(1.50), near(1.75) * 10)
        self.assertGreater(near(2.44), near(2.20) * 5)

    def test_the_attack_carries_the_stick_and_the_tail_does_not(self):
        """A struck drum, not a filtered tone burst.

        Measured as the share of energy above 600 Hz rather than as a centroid:
        every membrane mode here is under 430 Hz, so anything above that band is
        the stick and nothing else. The centroid of a 10 ms window is too coarse
        to be stable -- it ranged 234 to 511 Hz across six draws of the mode
        phases, while this ratio stayed above a million on every one of them.
        """
        window = int(0.02 * render.SAMPLE_RATE)

        def share_above(samples, hz):
            freqs, power = spectrum(samples)
            return float(power[freqs > hz].sum() / power.sum())

        for seed in range(4):
            with self.subTest(seed=seed):
                stroke = render.timpani(render.TIMPANI_MIDI,
                                        np.random.default_rng(seed))
                attack = share_above(stroke[:window], 600.0)
                tail = share_above(stroke[-4 * window:], 600.0)
                self.assertGreater(attack, 1e-3)
                self.assertLess(tail, 1e-4)
                self.assertGreater(attack, tail * 1000)

    def test_the_stroke_starts_and_ends_at_silence(self):
        rng = generator()
        stroke = render.timpani(render.TIMPANI_MIDI, rng)
        self.assertLess(abs(float(stroke[0])), 1e-9)
        self.assertLess(abs(float(stroke[-1])), 1e-6)

    def test_strokes_do_not_sum_into_one_louder_tone(self):
        """Randomised mode phases are what make a roll a rumble.

        With a fixed phase, n overlapping strokes add coherently to n times one
        stroke; incoherently they add to about the square root of n, which is the
        difference between a roll and a siren.
        """
        rng = generator()
        strokes = [render.timpani(render.TIMPANI_MIDI, rng) for _ in range(9)]
        summed = np.sum(strokes, axis=0)
        self.assertLess(level(summed), 9 * 0.6 * level(strokes[0]))


def track_of(stalls, tension=0.45):
    """A state track carrying only what the roll reads."""
    return [{"ply": index + 1, "stall": stall, "tension": tension}
            for index, stall in enumerate(stalls)]


def quiet_frames(count, colour="w"):
    """Frames with no captures, so that only the roll speaks."""
    return [{"captured_value": 0, "color": colour} for _ in range(count)]


class Roll(unittest.TestCase):
    def test_nothing_is_played_below_the_floor(self):
        below = render.percussion(quiet_frames(4),
                                  track_of([render.TIMPANI_FLOOR * 0.5] * 4))
        above = render.percussion(quiet_frames(4),
                                  track_of([render.TIMPANI_FLOOR * 1.5] * 4))
        self.assertEqual(below, [])
        self.assertGreater(len(above), 0)

    def test_the_roll_grows_with_the_stall_term(self):
        levels = []
        for stall in (0.10, 0.30, 0.60, 1.00):
            blocks = render.percussion(quiet_frames(6), track_of([stall] * 6))
            levels.append(sum(energy(block) for _, block in blocks))
        self.assertEqual(levels, sorted(levels))
        self.assertGreater(levels[-1] / levels[0], 20.0)

    def test_the_roll_stops_when_the_stall_resets(self):
        """Game 6 in miniature: a build, a pawn move, then nothing.

        No period logic and no schedule. The term the roll reads went to zero
        because the board reset it, and there is nothing left to play.
        """
        blocks = render.percussion(quiet_frames(12),
                                  track_of([1.0] * 6 + [0.0] * 6))
        reset_at = int(6 * render.STEP_SECONDS * render.SAMPLE_RATE)
        self.assertGreater(len(blocks), 0)
        self.assertLess(max(onset for onset, _ in blocks), reset_at)

    def test_the_rate_follows_tension(self):
        slack = render.percussion(quiet_frames(6), track_of([1.0] * 6, tension=0.0))
        taut = render.percussion(quiet_frames(6), track_of([1.0] * 6, tension=1.0))
        self.assertGreater(len(taut), len(slack))

    def test_every_stroke_falls_inside_the_ply_that_asked_for_it(self):
        """Otherwise a roll would run past a reset it is supposed to stop at."""
        for tension in (0.0, 0.5, 1.0):
            with self.subTest(tension=tension):
                blocks = render.percussion(quiet_frames(1),
                                           track_of([1.0], tension=tension))
                step = int(render.STEP_SECONDS * render.SAMPLE_RATE)
                self.assertGreater(len(blocks), 1)
                self.assertLess(max(onset for onset, _ in blocks), step)

    def test_the_roll_is_centred_because_it_is_nobody_s_colour(self):
        for _, block in render.percussion(quiet_frames(4), track_of([1.0] * 4)):
            self.assertAlmostEqual(rms(block[:, 0]), rms(block[:, 1]), places=9)

    def test_a_track_of_a_different_length_is_refused(self):
        with self.assertRaises(ValueError):
            render.percussion(quiet_frames(4), track_of([1.0] * 3))


class Corpus(unittest.TestCase):
    """The committed corpus must reach the code these tests guard.

    A separation floor no fixture exercises is a green light wired to nothing, so
    the rungs of the capture scale and the shape of a build are asserted against
    games that are actually in the repository.
    """

    @staticmethod
    def _annotated(path):
        from src import features, ingest
        return features.annotate(ingest.ingest(path))

    def test_the_committed_corpus_reaches_every_rung_of_the_capture_scale(self):
        frames = self._annotated(FIXTURES / "tactical_decisive.pgn")
        values = {frame["captured_value"] for frame in frames}
        for value in VALUES:
            with self.subTest(value=value):
                self.assertIn(value, values)

    def test_the_committed_corpus_reaches_both_colours_capturing(self):
        frames = self._annotated(FIXTURES / "tactical_decisive.pgn")
        capturing = {frame["color"] for frame in frames if frame["captured_value"]}
        self.assertEqual(capturing, {"w", "b"})

    def test_a_build_and_a_break_are_both_reached(self):
        from src import features, ingest, tension
        from tests.helpers import build_pgn, shuffle_moves, write_pgn

        # Fifty plies of shuffling saturate the stall term and a pawn move resets
        # it: the shape of Game 6's moves 82 to 110, in miniature.
        sans = shuffle_moves(50) + ["e4", "e5", "Nc3", "Nc6"]
        frames = features.annotate(ingest.ingest(write_pgn(build_pgn(sans))))
        entries = tension.tension_track(frames)["plies"]
        stalls = [entry["stall"] for entry in entries]
        self.assertAlmostEqual(max(stalls), 1.0, places=3)
        self.assertLess(stalls[-1], render.TIMPANI_FLOOR)

        blocks = render.percussion(frames, entries)
        break_at = int(50 * render.STEP_SECONDS * render.SAMPLE_RATE)
        self.assertGreater(len(blocks), 0)
        self.assertLess(max(onset for onset, _ in blocks), break_at)
        self.assertLess(break_at, int(len(frames) * render.STEP_SECONDS
                                      * render.SAMPLE_RATE))


class Pitch(unittest.TestCase):
    def test_the_scale_is_a_major_scale(self):
        """Pinned to the literal degrees and to the interval pattern.

        Comparing the measured degrees to `MAJOR_DEGREES` alone asserts nothing:
        a chromatic run passes it while the scale the module documents is gone.
        """
        self.assertEqual(render.MAJOR_DEGREES, (0, 2, 4, 5, 7, 9, 11, 12))
        steps = [b - a for a, b in zip(render.MAJOR_DEGREES,
                                       render.MAJOR_DEGREES[1:])]
        self.assertEqual(steps, [2, 2, 1, 2, 2, 2, 1])

    def test_file_selects_a_scale_degree(self):
        degrees = [render.midi_for_square(f + "1") for f in "abcdefgh"]
        self.assertEqual([d - degrees[0] for d in degrees],
                         list(render.MAJOR_DEGREES))

    def test_the_bottom_of_the_board_sits_in_a_usable_bass(self):
        """`BASE_MIDI` sets the register of the whole piece, and the drum's pitch.

        The audible-middle check alone does not pin it: moving the root up an
        octave keeps every square inside the range and moves everything the
        renderer plays, so the root is pinned to a literal and to the note it is.
        """
        self.assertEqual(render.BASE_MIDI, 50)
        self.assertAlmostEqual(render._frequency(render.BASE_MIDI), 146.83, places=1)

    def test_rank_raises_the_octave(self):
        self.assertEqual(render.midi_for_square("a3") - render.midi_for_square("a1"), 12)
        self.assertEqual(render.midi_for_square("a8") - render.midi_for_square("a1"), 36)

    def test_pitch_never_leaves_the_audible_middle(self):
        for square in (f + r for f in "abcdefgh" for r in "12345678"):
            with self.subTest(square=square):
                hz = render._frequency(render.midi_for_square(square))
                self.assertGreater(hz, 40.0)
                self.assertLess(hz, 5000.0)


class Mixer(unittest.TestCase):
    @staticmethod
    def _block(length, left, right):
        block = np.zeros((length, 2))
        block[:, 0] = left
        block[:, 1] = right
        return block

    def test_blocks_land_at_their_onsets_and_sum(self):
        audio = render._mix([(0, self._block(10, 1.0, 0.5)),
                             (5, self._block(10, 1.0, 0.5))], total=20)
        self.assertEqual(audio.shape, (20, 2))
        # The overlap is twice a single block, and normalisation is applied once
        # to the whole buffer, so the ratio survives it.
        self.assertAlmostEqual(float(audio[7, 0]) / float(audio[2, 0]), 2.0, places=5)

    def test_a_quiet_mix_is_brought_up_to_the_peak(self):
        audio = render._mix([(0, self._block(100, 0.01, 0.01))], total=100)
        self.assertAlmostEqual(float(np.abs(audio).max()), render.PEAK, places=6)

    def test_a_loud_mix_is_brought_down_below_clipping(self):
        """`PEAK` pinned to a literal and to the consequence of being one.

        Comparing the output's maximum to `PEAK` alone holds for every value of
        `PEAK`, including 1.5, where every render clips from end to end, and 0.05,
        where every render is inaudible.
        """
        self.assertEqual(render.PEAK, 0.89)
        audio = render._mix([(0, self._block(100, 4.0, -4.0))], total=100)
        self.assertLess(float(np.abs(audio).max()), 1.0)
        self.assertGreater(float(np.abs(audio).max()), 0.5)
        self.assertAlmostEqual(float(np.abs(audio).max()), render.PEAK, places=6)

    def test_a_block_past_the_stated_total_extends_the_buffer(self):
        audio = render._mix([(90, self._block(50, 1.0, 1.0))], total=100)
        self.assertEqual(len(audio), 140)

    def test_an_empty_timeline_is_silent_at_the_stated_length(self):
        audio = render._mix([], total=1000)
        self.assertEqual(audio.shape, (1000, 2))
        self.assertEqual(float(np.abs(audio).max()), 0.0)


class Output(unittest.TestCase):
    @staticmethod
    def _render(frames, track):
        out = pathlib.Path(tempfile.mkdtemp()) / "game.wav"
        render.render(frames, track, out)
        import soundfile
        audio, rate = soundfile.read(out)
        return audio, rate

    def _game(self, plies=12, sans=None, **kwargs):
        from src import features, ingest, tension
        from tests.helpers import build_pgn, shuffle_moves, write_pgn
        moves = shuffle_moves(plies) if sans is None else sans
        frames = features.annotate(ingest.ingest(write_pgn(
            build_pgn(moves, **kwargs))))
        return frames, tension.tension_track(frames)

    def _fixture(self, name):
        from src import features, ingest, tension
        frames = features.annotate(ingest.ingest(FIXTURES / name))
        return frames, tension.tension_track(frames)

    def test_writes_stereo_44100_without_clipping(self):
        audio, rate = self._render(*self._game())
        self.assertEqual(rate, render.SAMPLE_RATE)
        self.assertEqual(audio.ndim, 2)
        self.assertEqual(audio.shape[1], 2)
        self.assertLess(float(np.abs(audio).max()), 1.0)
        self.assertAlmostEqual(float(np.abs(audio).max()), render.PEAK, places=3)

    def test_the_timeline_carries_percussion(self):
        audio, _ = self._render(*self._game())
        self.assertGreater(float(np.abs(audio).max()), 0.5)
        self.assertGreater(rms(audio[:, 0]), 1e-4)

    def test_a_game_of_captures_renders_them(self):
        """The committed tactical game, through the whole path to a file."""
        frames, track = self._fixture("tactical_decisive.pgn")
        audio, _ = self._render(frames, track)
        self.assertGreater(float(np.abs(audio).max()), 0.5)
        self.assertGreater(len(render.percussion(frames, track["plies"])), 10)

    def test_a_mono_downmix_keeps_the_signal_and_does_not_clip(self):
        audio, _ = self._render(*self._game())
        summed = audio.sum(axis=1) / 2.0
        self.assertLessEqual(float(np.abs(summed).max()), render.PEAK + 1e-6)
        self.assertGreater(rms(summed),
                           0.4 * max(rms(audio[:, 0]), rms(audio[:, 1])))

    def test_a_longer_game_makes_a_longer_render(self):
        short, _ = self._render(*self._game(8))
        long, _ = self._render(*self._game(40))
        self.assertGreater(len(long), len(short) * 3)

    def test_the_timeline_is_one_step_per_ply(self):
        """Pins seconds per ply, not just that more plies is longer.

        Dropping the step out of the timeline length -- `len(frames)` samples
        instead of `len(frames)` steps -- turns a two-minute render into two
        milliseconds and leaves the longer-render comparison green.
        """
        self.assertEqual(render.STEP_SECONDS, 0.50)
        plies = 12
        audio, rate = self._render(*self._game(plies))
        self.assertGreaterEqual(len(audio) / rate, plies * render.STEP_SECONDS - 0.01)
        # A block sounding on the last ply may ring past the grid; nothing else may.
        self.assertLess(len(audio) / rate, plies * render.STEP_SECONDS + 3.0)

    def test_renders_a_game_with_no_clocks_and_no_evaluations(self):
        frames, track = self._game()
        self.assertEqual(track["active_layers"], ["board"])
        audio, rate = self._render(frames, track)
        self.assertEqual(rate, render.SAMPLE_RATE)
        self.assertGreater(float(np.abs(audio).max()), 0.5)

    def test_is_deterministic(self):
        """Load-bearing now that the renderer draws noise.

        This assertion was vacuous while the timeline was silent: two all-zero
        arrays are equal whatever the generator does.
        """
        frames, track = self._game()
        first, _ = self._render(frames, track)
        second, _ = self._render(frames, track)
        self.assertGreater(float(np.abs(first).max()), 0.5)
        np.testing.assert_array_equal(first, second)

    def test_two_different_games_do_not_render_alike(self):
        """One seed per render must not mean one render per seed."""
        shuffled, _ = self._render(*self._game(12))
        tactical, _ = self._render(*self._fixture("tactical_decisive.pgn"))
        shortest = min(len(shuffled), len(tactical))
        self.assertGreater(
            float(np.abs(shuffled[:shortest] - tactical[:shortest]).max()), 0.1)


if __name__ == "__main__":
    unittest.main()
