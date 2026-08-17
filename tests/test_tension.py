"""Phase 3: the tension state machine.

The three axes must stay independent, tension must be carried rather than
recomputed, cadence must fire only on board events, and the module must emit
state rather than sound.
"""

from __future__ import annotations

import copy
import logging
import pathlib
import unittest

from src import features, ingest, tension
from tests.helpers import build_pgn, clocks_from_think, shuffle_moves, write_pgn

logging.getLogger("src.ingest").setLevel(logging.CRITICAL)

SYNTHETIC = pathlib.Path(__file__).parent / "fixtures"

STATE_FIELDS = {
    "ply", "tension", "pressure_w", "pressure_b", "density", "meter",
    "cadence_strength", "stall", "active_layers", "game_tempo_scale",
}


def tracked(**kwargs):
    frames = features.annotate(ingest.ingest(write_pgn(build_pgn(**kwargs))))
    return frames, tension.tension_track(frames)


class Output(unittest.TestCase):
    def test_emits_exactly_the_specified_state(self):
        _, track = tracked(sans=shuffle_moves(6))
        self.assertEqual(set(track[0]), STATE_FIELDS)

    def test_emits_no_musical_parameters(self):
        """Mapping state onto sound belongs to render, not here."""
        _, track = tracked(sans=shuffle_moves(6))
        for forbidden in ("register", "dissonance", "scale", "pitch", "timbre"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, track[0])

    def test_scalars_stay_in_range(self):
        for name in ("wc2021_game6.pgn", "on_pace_bounded.pgn", "premove_chain.pgn"):
            path = SYNTHETIC / name
            if not path.exists():
                path = pathlib.Path("data/pgn") / name
            if not path.exists():
                continue
            with self.subTest(game=name):
                track = tension.tension_track(
                    features.annotate(ingest.ingest(path))
                )
                for entry in track:
                    for field in ("tension", "density", "cadence_strength", "stall"):
                        self.assertGreaterEqual(entry[field], 0.0)
                        self.assertLessEqual(entry[field], 1.0)

    def test_empty_frames(self):
        self.assertEqual(tension.tension_track([]), [])

    def test_does_not_mutate_the_frames(self):
        frames, _ = tracked(sans=shuffle_moves(8))
        before = copy.deepcopy(frames)
        tension.tension_track(frames)
        self.assertEqual(frames, before)


class BudgetBinding(unittest.TestCase):
    """Behavioural, not a source scan: a computed key would escape a grep."""

    def test_deleting_budget_changes_nothing(self):
        frames = features.annotate(
            ingest.ingest(SYNTHETIC / "on_pace_bounded.pgn")
        )
        baseline = tension.tension_track(copy.deepcopy(frames))
        stripped = copy.deepcopy(frames)
        for frame in stripped:
            del frame["budget_w"]
            del frame["budget_b"]
        self.assertEqual(tension.tension_track(stripped), baseline)

    def test_corrupting_budget_changes_nothing(self):
        frames = features.annotate(
            ingest.ingest(SYNTHETIC / "on_pace_bounded.pgn")
        )
        baseline = tension.tension_track(copy.deepcopy(frames))
        poisoned = copy.deepcopy(frames)
        for frame in poisoned:
            frame["budget_w"] = 99999.0
            frame["budget_b"] = -1.0
        self.assertEqual(tension.tension_track(poisoned), baseline)


class Independence(unittest.TestCase):
    def test_mobility_does_not_reach_tension(self):
        """Mobility drives density and nothing else."""
        frames, baseline = tracked(sans=shuffle_moves(10))
        altered = copy.deepcopy(frames)
        for frame in altered:
            frame["mobility_w"] = 1
            frame["mobility_b"] = 1
        changed = tension.tension_track(altered)
        self.assertEqual([e["tension"] for e in changed],
                         [e["tension"] for e in baseline])
        self.assertNotEqual([e["density"] for e in changed],
                            [e["density"] for e in baseline])

    def test_pressure_does_not_reach_tension(self):
        frames, baseline = tracked(
            sans=shuffle_moves(10),
            clocks=clocks_from_think([0, 0] + [1.0] * 8, start=600),
            time_control="600",
        )
        altered = copy.deepcopy(frames)
        for frame in altered:
            frame["time_pressure_w"] = 1.0
            frame["time_pressure_b"] = 1.0
        changed = tension.tension_track(altered)
        self.assertEqual([e["tension"] for e in changed],
                         [e["tension"] for e in baseline])

    def test_pressure_passes_straight_through(self):
        frames, track = tracked(
            sans=shuffle_moves(8),
            clocks=clocks_from_think([0, 0] + [1.0] * 6, start=600),
            time_control="600",
        )
        for frame, entry in zip(frames, track):
            self.assertEqual(entry["pressure_w"], frame["time_pressure_w"])
            self.assertEqual(entry["pressure_b"], frame["time_pressure_b"])

    def test_absent_pressure_switches_the_layer_off_rather_than_defaulting(self):
        _, track = tracked(sans=shuffle_moves(8))
        self.assertTrue(all(e["pressure_w"] is None for e in track))
        self.assertTrue(all(e["pressure_b"] is None for e in track))
        self.assertNotIn("pressure", track[0]["active_layers"])


class Accumulation(unittest.TestCase):
    def test_tension_is_carried_not_recomputed(self):
        """Two plies with identical features must not carry identical tension.

        A knight shuffle repeats the same position, so a per-ply recomputation
        would return the same number every time.
        """
        _, track = tracked(sans=shuffle_moves(12))
        values = [e["tension"] for e in track]
        self.assertGreater(len(set(values)), 1)
        # Monotone build while nothing releases.
        rising = [b >= a for a, b in zip(values[1:5], values[2:6])]
        self.assertTrue(all(rising), values)

    def test_stall_weighs_the_upper_range_more(self):
        """A counter at 45 is not twice a counter at 22."""
        low = tension._stall({"halfmove_clock": 22, "repetition_2": False})
        high = tension._stall({"halfmove_clock": 45, "repetition_2": False})
        self.assertGreater(high / low, 3.0)

    def test_repetition_only_modifies_the_stall_term(self):
        plain = tension._stall({"halfmove_clock": 20, "repetition_2": False})
        repeated = tension._stall({"halfmove_clock": 20, "repetition_2": True})
        self.assertAlmostEqual(repeated, plain * tension.REPETITION_BOOST, places=6)


class Cadence(unittest.TestCase):
    def test_never_fires_without_a_board_event(self):
        """A shuffle moves no pawn, takes nothing and traps no king.

        Only the final ply may cadence, and only because the game ends.
        """
        _, track = tracked(sans=shuffle_moves(20))
        fired = [e for e in track[:-1] if e["cadence_strength"] > 0]
        self.assertEqual(fired, [])
        self.assertGreater(track[-1]["cadence_strength"], 0.0)

    @staticmethod
    def _break_after(quiet_plies):
        """A pawn break reached after `quiet_plies` of nothing happening."""
        sans = ["e4", "e5"] + shuffle_moves(quiet_plies) + ["d4", "exd4"]
        _, track = tracked(sans=sans)
        return track[-2]          # the d4 ply: resets the counter

    def test_strength_scales_with_accumulated_height(self):
        """The same event released from a greater height is a greater event."""
        shallow = self._break_after(2)
        deep = self._break_after(30)
        self.assertGreater(deep["cadence_strength"],
                           shallow["cadence_strength"] * 2)

    def test_strength_is_not_boolean(self):
        """Three identical events at rising heights must read as three sizes.

        Knight shuffles of 4, 16 and 24 plies, each ended by a pawn move that
        resets the counter. A boolean cadence would discard the only thing that
        distinguishes them.
        """
        sans = (shuffle_moves(4) + ["a4", "h5"] + shuffle_moves(16)
                + ["b4", "g5"] + shuffle_moves(24) + ["c4", "f5"])
        _, track = tracked(sans=sans)
        breaks = [e["cadence_strength"] for e in track
                  if e["cadence_strength"] > 0][:3]
        self.assertEqual(len(breaks), 3)
        self.assertEqual(breaks, sorted(breaks), breaks)
        self.assertGreater(breaks[-1], breaks[0] * 2)

    def test_termination_discharges_completely(self):
        """The final release is total, so the last ply reads zero."""
        _, track = tracked(sans=shuffle_moves(20))
        self.assertEqual(track[-1]["tension"], 0.0)
        self.assertGreater(track[-2]["tension"], 0.0)


class MissingEvaluations(unittest.TestCase):
    def test_absence_is_not_the_same_as_a_calm_evaluation(self):
        """Renormalising gives the surviving terms more weight, by design.

        A flat evaluation is information -- the position is not swinging -- and
        correctly drags tension down. An absent one is not information, so the
        remaining terms carry the whole scale instead.
        """
        sans = shuffle_moves(12)
        _, without = tracked(sans=sans)
        _, calm = tracked(sans=sans, evals=["0.0"] * 12)
        self.assertNotIn("eval", without[0]["active_layers"])
        self.assertIn("eval", calm[-1]["active_layers"])

        peak_without = max(e["tension"] for e in without)
        peak_calm = max(e["tension"] for e in calm)
        self.assertGreater(peak_without, peak_calm)
        expected = 1.0 / (1.0 - tension.WEIGHTS["eval"])
        self.assertAlmostEqual(peak_without / peak_calm, expected, places=1)

    def test_a_game_with_no_evaluations_is_still_coherent(self):
        """Three of eight fixtures have neither clocks nor evaluations."""
        path = pathlib.Path("data/pgn") / "kasparov_topalov_1999.pgn"
        if not path.exists():
            self.skipTest("real-game fixture absent from data/pgn (gitignored): "
                          "kasparov_topalov_1999.pgn")
        track = tension.tension_track(features.annotate(ingest.ingest(path)))
        values = [e["tension"] for e in track]
        self.assertGreater(max(values), 0.0)
        self.assertGreater(len(set(values)), len(values) // 4)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))
        self.assertTrue(any(e["cadence_strength"] > 0 for e in track))
        self.assertEqual(track[0]["active_layers"], ["board"])

    def test_volatile_evaluations_raise_tension(self):
        sans = shuffle_moves(12)
        _, calm = tracked(sans=sans, evals=["0.0"] * 12)
        _, wild = tracked(sans=sans,
                          evals=[str(v) for v in (0, 5, -5, 5, -5, 5,
                                                  -5, 5, -5, 5, -5, 5)])
        self.assertGreater(max(e["tension"] for e in wild),
                           max(e["tension"] for e in calm))


class DensityAndMeter(unittest.TestCase):
    def test_meter_is_dotted_only_on_a_forcing_streak(self):
        _, track = tracked(sans=shuffle_moves(8))
        self.assertTrue(all(e["meter"] == "free" for e in track))

    def test_a_capture_sequence_turns_the_meter_dotted(self):
        _, track = tracked(sans=["e4", "d5", "exd5", "Qxd5"])
        self.assertEqual(track[2]["meter"], "free")     # first forcing ply
        self.assertEqual(track[3]["meter"], "dotted")   # streak of two

    def test_density_follows_mobility(self):
        frames, baseline = tracked(sans=shuffle_moves(8))
        cramped = copy.deepcopy(frames)
        for frame in cramped:
            frame["mobility_w"] = frame["mobility_b"] = 2
        changed = tension.tension_track(cramped)
        self.assertLess(changed[0]["density"], baseline[0]["density"])


class ActiveLayers(unittest.TestCase):
    def test_records_what_was_available(self):
        _, bare = tracked(sans=shuffle_moves(6))
        self.assertEqual(bare[0]["active_layers"], ["board"])
        _, rich = tracked(
            sans=shuffle_moves(6),
            clocks=clocks_from_think([0, 0] + [1.0] * 4, start=600),
            time_control="600",
            evals=["0.1"] * 6,
        )
        self.assertIn("eval", rich[-1]["active_layers"])
        self.assertIn("pressure", rich[-1]["active_layers"])
        self.assertIn("think", rich[-1]["active_layers"])


if __name__ == "__main__":
    unittest.main()
