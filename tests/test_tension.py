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
                self.skipTest(f"fixture absent (gitignored): {name}")
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

    @staticmethod
    def _every_game():
        """Every game available, so the check covers every code path.

        A fixture with no captures, no pawn moves and no evaluations enters none
        of the release branches and none of the eval term, so a read placed
        there would go unnoticed.
        """
        paths = sorted(SYNTHETIC.glob("*.pgn"))
        paths += sorted(pathlib.Path("data/pgn").glob("*.pgn"))
        return paths

    def test_deleting_budget_changes_nothing(self):
        for path in self._every_game():
            with self.subTest(game=path.name):
                frames = features.annotate(ingest.ingest(path))
                baseline = tension.tension_track(copy.deepcopy(frames))
                stripped = copy.deepcopy(frames)
                for frame in stripped:
                    del frame["budget_w"]
                    del frame["budget_b"]
                self.assertEqual(tension.tension_track(stripped), baseline)

    def test_corrupting_budget_changes_nothing(self):
        for path in self._every_game():
            with self.subTest(game=path.name):
                frames = features.annotate(ingest.ingest(path))
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
    def test_tension_eases_toward_its_drive_rather_than_snapping_to_it(self):
        """Hold every input constant: a carried value climbs, a recomputed one
        arrives at full height on the first ply and stays there."""
        frames, _ = tracked(sans=shuffle_moves(30))
        for frame in frames:
            frame["halfmove_clock"] = 30
            frame["repetition_2"] = False
        track = tension.tension_track(frames)
        values = [e["tension"] for e in track]
        ceiling = max(values)
        self.assertGreater(ceiling, 0.0)
        self.assertLess(values[0], 0.5 * ceiling)
        self.assertLess(values[0], values[3])
        self.assertLess(values[3], values[10])

    def test_stall_weighs_the_upper_range_more(self):
        """A counter at 45 is not twice a counter at 22."""
        low = tension._stall({"halfmove_clock": 22, "repetition_2": False})
        high = tension._stall({"halfmove_clock": 45, "repetition_2": False})
        self.assertGreater(high / low, 3.0)

    def test_repetition_lifts_the_stall_term_by_a_quarter(self):
        """Pinned to a literal: comparing the constant to itself asserts nothing
        and passes for 1.0, which removes the modifier the spec requires."""
        self.assertEqual(tension.REPETITION_BOOST, 1.25)
        plain = tension._stall({"halfmove_clock": 20, "repetition_2": False})
        repeated = tension._stall({"halfmove_clock": 20, "repetition_2": True})
        self.assertAlmostEqual(plain, 0.16, places=6)
        self.assertAlmostEqual(repeated, 0.20, places=6)

    def test_normalisers_are_pinned_to_their_measured_values(self):
        """Every one of these was silently replaceable by any other number."""
        self.assertEqual(tension.LOCKED_FULL, 6)
        self.assertEqual(tension.PAWN_TENSION_FULL, 3)
        self.assertEqual(tension.KING_PRESSURE_FULL, 8)
        self.assertEqual(tension.MOBILITY_FULL, 80)
        self.assertEqual(tension.EVAL_VOLATILITY_FULL, 150.0)
        self.assertEqual(tension.EVAL_WINDOW, 4)
        self.assertEqual(tension.STALL_SATURATION, 50)
        self.assertEqual(tension.FORCING_STREAK_FULL, 3)
        # and their consequences, so the constants are not merely recited
        self.assertEqual(tension._structure(
            {"locked_pawns": 6, "pawn_tension": 3}), 1.0)
        self.assertEqual(tension._king(
            {"king_pressure_w": 0, "king_pressure_b": 8}), 1.0)
        self.assertEqual(tension._stall(
            {"halfmove_clock": 50, "repetition_2": False}), 1.0)
        self.assertAlmostEqual(tension._stall(
            {"halfmove_clock": 25, "repetition_2": False}), 0.25, places=6)

    def test_both_structural_terms_reach_tension(self):
        """locked_pawns and pawn_tension are both primary structural terms."""
        self.assertGreater(tension._structure(
            {"locked_pawns": 0, "pawn_tension": 3}), 0.0)
        self.assertGreater(tension._structure(
            {"locked_pawns": 6, "pawn_tension": 0}), 0.0)

    def test_king_pressure_reads_whichever_king_is_hotter(self):
        """A king under attack is tension regardless of whose it is."""
        self.assertEqual(
            tension._king({"king_pressure_w": 8, "king_pressure_b": 0}),
            tension._king({"king_pressure_w": 0, "king_pressure_b": 8}),
        )
        self.assertEqual(tension._king(
            {"king_pressure_w": 8, "king_pressure_b": 0}), 1.0)


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


class ReleaseEvents(unittest.TestCase):
    """Each of the five, pinned. Deleting any one of them passed the suite."""

    @staticmethod
    def _events(sans, index):
        frames = features.annotate(ingest.ingest(write_pgn(build_pgn(sans))))
        return tension._release_events(frames[index], frames[:index],
                                       index == len(frames) - 1)

    def test_pawn_break(self):
        frames = features.annotate(ingest.ingest(write_pgn(build_pgn(
            shuffle_moves(4)))))
        # A drop of two locked pawns in one ply is the break.
        frames[2]["locked_pawns"] = 4
        frames[3]["locked_pawns"] = 1
        events = tension._release_events(frames[3], frames[:3], False)
        self.assertIn("pawn_break", events)
        frames[3]["locked_pawns"] = 3          # a drop of one is not
        self.assertNotIn("pawn_break",
                         tension._release_events(frames[3], frames[:3], False))

    def test_progress_reset(self):
        self.assertIn("progress_reset", self._events(["Nf3", "Nf6", "e4"], 2))
        self.assertNotIn("progress_reset", self._events(["Nf3", "Nf6", "Ng1"], 2))

    def test_simplification_on_a_queen_trade(self):
        sans = ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qe5+", "Qe2", "Qxe2+", "Bxe2"]
        self.assertIn("simplification", self._events(sans, 8))

    def test_simplification_ignores_an_even_trade_of_balance(self):
        """The balance is unchanged by an even trade; the board is not.

        Reading the balance instead of the material removed misses every trade
        and fires on unanswered sacrifices instead.
        """
        frames = features.annotate(ingest.ingest(write_pgn(build_pgn(
            ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qe5+", "Qe2", "Qxe2+", "Bxe2"]))))
        self.assertEqual(frames[6]["material_balance"],
                         frames[8]["material_balance"])
        self.assertGreaterEqual(
            frames[6]["material_total"] - frames[8]["material_total"], 9)

    def test_simplification_needs_a_capture(self):
        """The quiet move after a recapture must not fire the same exchange."""
        sans = ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qe5+", "Qe2", "Qxe2+",
                "Bxe2", "Nf6"]
        self.assertNotIn("simplification", self._events(sans, 9))

    def test_simplification_on_points_alone(self):
        """The points threshold, exercised without a queen leaving the board.

        A rook trade takes ten points off with both queens still on, so it can
        only fire through the material rule.
        """
        frames = features.annotate(ingest.ingest(write_pgn(build_pgn(
            shuffle_moves(4)))))
        self.assertEqual(tension.SIMPLIFICATION_POINTS, 5)
        frames[1]["material_total"] = 78
        frames[3]["is_capture"] = True
        frames[3]["material_total"] = 73          # five points off: fires
        self.assertIn("simplification",
                      tension._release_events(frames[3], frames[:3], False))
        frames[3]["material_total"] = 74          # four points off: does not
        self.assertNotIn("simplification",
                         tension._release_events(frames[3], frames[:3], False))

    def test_king_safety_needs_real_danger_not_just_a_big_drop(self):
        """Two attackers falling to none is a drop, but was never danger."""
        frames = features.annotate(ingest.ingest(write_pgn(build_pgn(
            shuffle_moves(4)))))
        frames[3]["color"] = "w"
        frames[2]["king_pressure_w"] = tension.SHELTER_PRESSURE_FLOOR - 1
        frames[3]["king_pressure_w"] = 0
        self.assertNotIn("king_safety",
                         tension._release_events(frames[3], frames[:3], False))
        frames[2]["king_pressure_w"] = tension.SHELTER_PRESSURE_FLOOR
        self.assertIn("king_safety",
                      tension._release_events(frames[3], frames[:3], False))

    def test_king_safety_on_castling(self):
        sans = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"]
        self.assertIn("king_safety", self._events(sans, 6))

    def test_king_safety_ignores_ordinary_quiet_moves(self):
        """Without a floor and a real drop this fires on endgame shuffling."""
        frames = features.annotate(ingest.ingest(write_pgn(build_pgn(
            shuffle_moves(4)))))
        frames[2]["king_pressure_w"] = 2       # never in real danger
        frames[3]["king_pressure_w"] = 1
        frames[3]["color"] = "w"
        self.assertNotIn("king_safety",
                         tension._release_events(frames[3], frames[:3], False))
        frames[2]["king_pressure_w"] = 4       # real danger, real escape
        frames[3]["king_pressure_w"] = 1
        self.assertIn("king_safety",
                      tension._release_events(frames[3], frames[:3], False))

    def test_termination_on_the_final_ply_only(self):
        frames = features.annotate(ingest.ingest(write_pgn(build_pgn(
            shuffle_moves(6)))))
        self.assertIn("termination",
                      tension._release_events(frames[5], frames[:5], True))
        self.assertNotIn("termination",
                         tension._release_events(frames[4], frames[:4], False))

    def test_queens_are_counted_on_the_board_not_in_the_castling_field(self):
        """`Q` in a full FEN also spells queenside castling rights."""
        self.assertEqual(tension._queens("4k2r/8/8/8/8/8/8/R3K3 b Qk - 10 20"), 0)
        self.assertEqual(tension._queens("3qk3/8/8/8/8/8/8/3QK3 w KQkq - 0 1"), 2)

    def test_each_event_discharges_a_different_share(self):
        self.assertEqual(tension.RELEASE_DEPTH["termination"], 1.0)
        for event in ("pawn_break", "progress_reset", "simplification",
                      "king_safety"):
            with self.subTest(event=event):
                self.assertGreater(tension.RELEASE_DEPTH[event], 0.0)
                self.assertLess(tension.RELEASE_DEPTH[event], 1.0)


class Plotting(unittest.TestCase):
    def test_plot_track_writes_a_figure(self):
        import tempfile

        _, track = tracked(
            sans=shuffle_moves(10),
            clocks=clocks_from_think([0, 0] + [1.0] * 8, start=600),
            time_control="600",
        )
        out = pathlib.Path(tempfile.mkdtemp()) / "game_tension.png"
        tension.plot_track(track, out)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 1000)

    def test_plot_track_survives_a_game_with_no_clocks(self):
        import tempfile

        _, track = tracked(sans=shuffle_moves(10))
        out = pathlib.Path(tempfile.mkdtemp()) / "bare_tension.png"
        tension.plot_track(track, out)
        self.assertTrue(out.exists())


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

    def test_every_kind_of_forcing_move_counts(self):
        """Checks and forced replies are forcing, not only captures."""
        for flags in ({"is_check": True}, {"is_capture": True},
                      {"forced": True}):
            with self.subTest(**flags):
                frame = {"is_check": False, "is_capture": False, "forced": False}
                frame.update(flags)
                self.assertTrue(tension._is_forcing(frame))
        self.assertFalse(tension._is_forcing(
            {"is_check": False, "is_capture": False, "forced": False}))

    def test_a_check_sequence_turns_the_meter_dotted(self):
        frames, _ = tracked(sans=shuffle_moves(6))
        for frame in frames[2:5]:
            frame["is_check"] = True
        track = tension.tension_track(frames)
        self.assertEqual(track[2]["meter"], "free")
        self.assertEqual(track[3]["meter"], "dotted")
        self.assertEqual(track[4]["meter"], "dotted")

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
