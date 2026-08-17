"""Phase 4a: the six voices.

Timbres either separate or they do not, and no later layer rescues them, so the
separation is measured rather than asserted. Spectral centroid stands in for
brightness, an envelope follower for attack and decay.
"""

from __future__ import annotations

import itertools
import pathlib
import tempfile
import unittest

import numpy as np

from src import render

PIECES = "PNBRQK"
MIDDLE_C_ISH = 62


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


class Voices(unittest.TestCase):
    def test_every_piece_has_a_voice(self):
        self.assertEqual(set(render.VOICES), set(PIECES))

    def test_no_two_voices_are_twins(self):
        """Pairwise separation over brightness, attack, decay and level."""
        points = {}
        for piece in PIECES:
            samples = render.voice(piece, MIDDLE_C_ISH, "w")
            attack, decay, rms = shape(samples)
            points[piece] = np.array([np.log(centroid(samples)), attack * 20,
                                      decay * 4, rms * 6])
        for one, other in itertools.combinations(PIECES, 2):
            with self.subTest(pair=one + other):
                self.assertGreater(
                    float(np.linalg.norm(points[one] - points[other])), 0.5)

    def test_the_specified_character_of_each_voice(self):
        measured = {p: (centroid(render.voice(p, MIDDLE_C_ISH, "w")),
                        *shape(render.voice(p, MIDDLE_C_ISH, "w")))
                    for p in PIECES}
        attacks = {p: measured[p][1] for p in PIECES}
        decays = {p: measured[p][2] for p in PIECES}
        peaks = {p: float(np.abs(render.voice(p, MIDDLE_C_ISH, "w")).max())
                 for p in PIECES}

        # Pawn: shortest note and the fastest to die away.
        self.assertEqual(min(decays, key=decays.get), "P")
        self.assertEqual(min(render.VOICES, key=lambda p: render.VOICES[p]["duration"]), "P")
        # Bishop: softest attack and the purest spectrum.
        self.assertEqual(max(attacks, key=attacks.get), "B")
        self.assertEqual(
            min(PIECES, key=lambda p: spread(render.voice(p, MIDDLE_C_ISH, "w"))), "B")
        # Rook: the lowest voice.
        self.assertEqual(min(render.VOICES, key=lambda p: render.VOICES[p]["register"]), "R")
        # King: the quietest. Measured on peak, not rms -- the pawn has the
        # lower rms only because it dies away almost at once.
        self.assertEqual(min(peaks, key=peaks.get), "K")
        # Queen: every harmonic up to the eighth.
        self.assertEqual([h[0] for h in render.VOICES["Q"]["harmonics"]],
                         list(range(1, 9)))
        # King: odd harmonics only.
        self.assertTrue(all(h % 2 for h, _ in render.VOICES["K"]["harmonics"]))
        # Knight: the only voice that detunes and bends.
        for piece in PIECES:
            with self.subTest(piece=piece):
                self.assertEqual(bool(render.VOICES[piece].get("detune_cents")),
                                 piece == "N")

    def test_black_is_darker_than_white_for_every_voice(self):
        for piece in PIECES:
            with self.subTest(piece=piece):
                self.assertLess(centroid(render.voice(piece, MIDDLE_C_ISH, "b")),
                                centroid(render.voice(piece, MIDDLE_C_ISH, "w")))

    def test_the_cutoff_follows_the_note_rather_than_being_fixed(self):
        """A fixed cutoff sits above a low note's whole spectrum and does nothing.

        Measured at 1.00x separation for four of six voices before this, so the
        separation has to hold at the bottom of the board as well as the top.
        """
        for midi in (38, 62, 92):
            with self.subTest(midi=midi):
                bright = centroid(render.voice("Q", midi, "w"))
                dark = centroid(render.voice("Q", midi, "b"))
                self.assertGreater(bright / dark, 1.2)

    def test_notes_start_and_end_at_silence(self):
        for piece in PIECES:
            with self.subTest(piece=piece):
                samples = render.voice(piece, MIDDLE_C_ISH, "w")
                self.assertLess(abs(samples[0]), 1e-6)
                self.assertLess(abs(samples[-1]), 1e-3)

    def test_no_voice_clips(self):
        for piece in PIECES:
            for midi in (38, 62, 92):
                with self.subTest(piece=piece, midi=midi):
                    self.assertLess(
                        float(np.abs(render.voice(piece, midi, "w")).max()), 1.0)


class Pitch(unittest.TestCase):
    def test_file_selects_a_scale_degree(self):
        degrees = [render.midi_for_square(f + "1") for f in "abcdefgh"]
        self.assertEqual([d - degrees[0] for d in degrees],
                         list(render.MAJOR_DEGREES))

    def test_rank_raises_the_octave(self):
        self.assertEqual(render.midi_for_square("a3") - render.midi_for_square("a1"), 12)
        self.assertEqual(render.midi_for_square("a8") - render.midi_for_square("a1"), 36)

    def test_pitch_never_leaves_the_audible_middle(self):
        squares = [f + r for f in "abcdefgh" for r in "12345678"]
        for square in squares:
            for piece in PIECES:
                offset = render.VOICES[piece]["register"]
                hz = render._frequency(render.midi_for_square(square) + offset)
                with self.subTest(square=square, piece=piece):
                    self.assertGreater(hz, 40.0)
                    self.assertLess(hz, 5000.0)


class Output(unittest.TestCase):
    @staticmethod
    def _render(frames, track):
        out = pathlib.Path(tempfile.mkdtemp()) / "game.wav"
        render.render(frames, track, out)
        import soundfile
        audio, rate = soundfile.read(out)
        return audio, rate

    def _game(self, plies=12, **kwargs):
        from src import features, ingest
        from tests.helpers import build_pgn, shuffle_moves, write_pgn
        from src import tension
        frames = features.annotate(ingest.ingest(write_pgn(
            build_pgn(shuffle_moves(plies), **kwargs))))
        return frames, tension.tension_track(frames)

    def test_writes_mono_44100_without_clipping(self):
        audio, rate = self._render(*self._game())
        self.assertEqual(rate, render.SAMPLE_RATE)
        self.assertEqual(audio.ndim, 1)
        self.assertLess(float(np.abs(audio).max()), 1.0)
        self.assertAlmostEqual(float(np.abs(audio).max()), render.PEAK, places=3)

    def test_a_longer_game_makes_a_longer_render(self):
        short, _ = self._render(*self._game(8))
        long, _ = self._render(*self._game(40))
        self.assertGreater(len(long), len(short) * 3)

    def test_renders_a_game_with_no_clocks_and_no_evaluations(self):
        frames, track = self._game()
        self.assertEqual(track["active_layers"], ["board"])
        audio, _ = self._render(frames, track)
        self.assertGreater(float(np.abs(audio).max()), 0.5)

    def test_is_deterministic(self):
        frames, track = self._game()
        first, _ = self._render(frames, track)
        second, _ = self._render(frames, track)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
