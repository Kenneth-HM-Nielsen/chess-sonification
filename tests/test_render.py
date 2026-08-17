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
                bright = centroid(render._colour(probe.copy(), "w", hz))
                dark = centroid(render._colour(probe.copy(), "b", hz))
                self.assertGreater(bright / dark, 1.2)

    def test_dark_keeps_less_of_the_spectrum_than_bright(self):
        hz = render._frequency(62)
        probe = harmonic_probe(hz)
        self.assertLess(spread(render._colour(probe.copy(), "b", hz)),
                        spread(render._colour(probe.copy(), "w", hz)))

    def test_both_colours_are_defined_and_ordered(self):
        self.assertEqual(set(render.CUTOFF_HARMONIC), {"w", "b"})
        self.assertGreater(render.CUTOFF_HARMONIC["w"],
                           render.CUTOFF_HARMONIC["b"])


class FadeOrder(unittest.TestCase):
    """The rule is fade *after* filtering. Pinned by consequence, not by comment."""

    def test_filtering_after_the_fade_leaves_a_step_at_the_note_end(self):
        hz = render._frequency(50)
        probe = harmonic_probe(hz)
        correct = render._declick(render._colour(probe.copy(), "b", hz))
        reversed_order = render._colour(render._declick(probe.copy()), "b", hz)
        self.assertLess(abs(float(correct[-1])), 1e-9)
        self.assertGreater(abs(float(reversed_order[-1])),
                           abs(float(correct[-1])) + 1e-4)

    def test_the_fade_touches_only_the_ends(self):
        probe = harmonic_probe(220.0)
        faded = render._declick(probe.copy())
        fade = int(render.FADE_SECONDS * render.SAMPLE_RATE)
        np.testing.assert_allclose(faded[fade:-fade], probe[fade:-fade])
        self.assertLess(abs(float(faded[0])), 1e-9)
        self.assertLess(abs(float(faded[-1])), 1e-9)


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


class Pitch(unittest.TestCase):
    def test_file_selects_a_scale_degree(self):
        degrees = [render.midi_for_square(f + "1") for f in "abcdefgh"]
        self.assertEqual([d - degrees[0] for d in degrees],
                         list(render.MAJOR_DEGREES))

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

    def test_the_peak_is_reached_exactly_once_normalised(self):
        audio = render._mix([(0, self._block(100, 0.01, 0.01))], total=100)
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

    def _game(self, plies=12, **kwargs):
        from src import features, ingest, tension
        from tests.helpers import build_pgn, shuffle_moves, write_pgn
        frames = features.annotate(ingest.ingest(write_pgn(
            build_pgn(shuffle_moves(plies), **kwargs))))
        return frames, tension.tension_track(frames)

    def test_writes_stereo_44100(self):
        audio, rate = self._render(*self._game())
        self.assertEqual(rate, render.SAMPLE_RATE)
        self.assertEqual(audio.ndim, 2)
        self.assertEqual(audio.shape[1], 2)

    def test_the_timeline_carries_no_voice_yet(self):
        """The 4a palette is deleted and Part B has not replaced it.

        Asserted rather than left implicit, so that the first sub-phase to put a
        sound in the timeline has to come here and say what it added.
        """
        audio, _ = self._render(*self._game())
        self.assertEqual(float(np.abs(audio).max()), 0.0)

    def test_a_longer_game_makes_a_longer_render(self):
        short, _ = self._render(*self._game(8))
        long, _ = self._render(*self._game(40))
        self.assertGreater(len(long), len(short) * 3)

    def test_renders_a_game_with_no_clocks_and_no_evaluations(self):
        frames, track = self._game()
        self.assertEqual(track["active_layers"], ["board"])
        audio, rate = self._render(frames, track)
        self.assertEqual(rate, render.SAMPLE_RATE)

    def test_is_deterministic(self):
        frames, track = self._game()
        first, _ = self._render(frames, track)
        second, _ = self._render(frames, track)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
