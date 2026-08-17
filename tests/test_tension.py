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

PLY_FIELDS = {
    "ply", "tension", "pressure_w", "pressure_b", "density", "meter",
    "cadence_strength", "stall",
}
GAME_FIELDS = {"game_tempo_scale", "active_layers", "plies"}


def tracked(**kwargs):
    """Frames and the per-ply entries. Most checks are per ply."""
    frames = features.annotate(ingest.ingest(write_pgn(build_pgn(**kwargs))))
    return frames, tension.tension_track(frames)["plies"]


def record(**kwargs):
    """The whole two-scope record, for the per-game checks."""
    frames = features.annotate(ingest.ingest(write_pgn(build_pgn(**kwargs))))
    return tension.tension_track(frames)


class Output(unittest.TestCase):
    def test_emits_exactly_the_specified_state(self):
        whole = record(sans=shuffle_moves(6))
        self.assertEqual(set(whole), GAME_FIELDS)
        self.assertEqual(set(whole["plies"][0]), PLY_FIELDS)

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
                )["plies"]
                for entry in track:
                    for field in ("tension", "density", "cadence_strength", "stall"):
                        self.assertGreaterEqual(entry[field], 0.0)
                        self.assertLessEqual(entry[field], 1.0)

    def test_empty_frames(self):
        self.assertEqual(
            tension.tension_track([]),
            {"game_tempo_scale": None, "active_layers": [], "plies": []},
        )

    def test_does_not_mutate_the_frames(self):
        frames, _ = tracked(sans=shuffle_moves(8))
        before = copy.deepcopy(frames)
        tension.tension_track(frames)
        self.assertEqual(frames, before)


class BudgetBinding(unittest.TestCase):
    """Behavioural, not a source scan: a computed key would escape a grep."""

    @staticmethod
    def _every_game():
        """Every game available. The committed fixtures alone must suffice.

        A fixture with no captures, no pawn moves and no evaluations enters none
        of the release branches and none of the eval term, so a read placed
        there would go unnoticed -- which is why the coverage is asserted below
        rather than inferred from how many files happen to be on disk.
        """
        paths = sorted(SYNTHETIC.glob("*.pgn"))
        paths += sorted(pathlib.Path("data/pgn").glob("*.pgn"))
        return paths

    def test_the_committed_corpus_reaches_every_release_branch(self):
        """Guards the guard: without this the binding test can silently stop
        covering anything, which is exactly what happened once already."""
        seen = set()
        layers = set()
        for path in sorted(SYNTHETIC.glob("*.pgn")):
            frames = features.annotate(ingest.ingest(path))
            history = []
            for index, frame in enumerate(frames):
                seen.update(tension._release_events(
                    frame, history, index == len(frames) - 1))
                history.append(frame)
            layers.update(tension.tension_track(frames)["active_layers"])
        self.assertEqual(seen, {"pawn_break", "progress_reset", "simplification",
                                "king_safety", "termination"})
        self.assertEqual(layers, {"board", "eval", "pressure", "think"})

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
        changed = tension.tension_track(altered)["plies"]
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
        changed = tension.tension_track(altered)["plies"]
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
        self.assertNotIn("pressure", record(sans=shuffle_moves(8))["active_layers"])


class Accumulation(unittest.TestCase):
    def test_tension_eases_toward_its_drive_rather_than_snapping_to_it(self):
        """Hold every input constant: a carried value climbs, a recomputed one
        arrives at full height on the first ply and stays there."""
        frames, _ = tracked(sans=shuffle_moves(30))
        for frame in frames:
            frame["halfmove_clock"] = 30
            frame["repetition_2"] = False
        track = tension.tension_track(frames)["plies"]
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
        _, track = tracked(sans=shuffle_moves(20), result="1-0")
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
        # A band, not a floor. The emitted axis is the square root of the raw
        # one, so a fourfold difference in accumulated height must read as
        # roughly twofold -- a plain floor also passes when the transform is
        # missing from cadence_strength and the raw ratio comes through whole.
        ratio = deep["cadence_strength"] / shallow["cadence_strength"]
        self.assertGreater(ratio, 1.7)
        self.assertLess(ratio, 2.6)

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
        _, track = tracked(sans=shuffle_moves(20), result="1-0")
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
            shuffle_moves(6), result="1-0"))))
        self.assertIn("termination",
                      tension._release_events(frames[5], frames[:5], True))
        self.assertNotIn("termination",
                         tension._release_events(frames[4], frames[:4], False))

    def test_queens_are_counted_on_the_board_not_in_the_castling_field(self):
        """`Q` in a full FEN also spells queenside castling rights."""
        self.assertEqual(tension._queens("4k2r/8/8/8/8/8/8/R3K3 b Qk - 10 20"), 0)
        self.assertEqual(tension._queens("3qk3/8/8/8/8/8/8/3QK3 w KQkq - 0 1"), 2)

    def test_a_draw_does_not_resolve(self):
        """A draw is a failure to resolve; forcing a release onto it would say
        something the game did not."""
        _, drawn = tracked(sans=shuffle_moves(20), result="1/2-1/2")
        self.assertEqual(drawn[-1]["cadence_strength"], 0.0)
        self.assertGreater(drawn[-1]["tension"], 0.0)

        _, decisive = tracked(sans=shuffle_moves(20), result="1-0")
        self.assertGreater(decisive[-1]["cadence_strength"], 0.0)
        self.assertEqual(decisive[-1]["tension"], 0.0)

    def test_an_unfinished_game_does_not_resolve(self):
        _, open_ended = tracked(sans=shuffle_moves(20), result="*")
        self.assertEqual(open_ended[-1]["cadence_strength"], 0.0)
        self.assertGreater(open_ended[-1]["tension"], 0.0)

    def test_both_decisive_results_resolve(self):
        for result in ("1-0", "0-1"):
            with self.subTest(result=result):
                _, track = tracked(sans=shuffle_moves(12), result=result)
                self.assertGreater(track[-1]["cadence_strength"], 0.0)

    @unittest.skipUnless(
        (pathlib.Path("data/pgn") / "spassky_petrosian_1969_closed.pgn").exists(),
        "real-game fixture absent from data/pgn (gitignored): "
        "spassky_petrosian_1969_closed.pgn",
    )
    def test_a_real_drawn_game_ends_unresolved(self):
        frames = features.annotate(ingest.ingest(
            pathlib.Path("data/pgn") / "spassky_petrosian_1969_closed.pgn"))
        self.assertEqual(frames[0]["result"], "1/2-1/2")
        track = tension.tension_track(frames)["plies"]
        self.assertEqual(track[-1]["cadence_strength"], 0.0)

    def test_each_event_discharges_a_different_share(self):
        self.assertEqual(tension.RELEASE_DEPTH["termination"], 1.0)
        for event in ("pawn_break", "progress_reset", "simplification",
                      "king_safety"):
            with self.subTest(event=event):
                self.assertGreater(tension.RELEASE_DEPTH[event], 0.0)
                self.assertLess(tension.RELEASE_DEPTH[event], 1.0)


class Calibration(unittest.TestCase):
    """One global monotone transform, chosen once and frozen."""

    def test_gamma_is_pinned_and_actually_transforms(self):
        self.assertEqual(tension.GAMMA, 0.5)
        self.assertNotEqual(tension.GAMMA, 1.0)
        self.assertAlmostEqual(tension._calibrate(0.25), 0.5, places=6)
        self.assertAlmostEqual(tension._calibrate(0.04), 0.2, places=6)

    def test_the_transform_is_monotone(self):
        rising = [tension._calibrate(v / 50) for v in range(51)]
        self.assertEqual(rising, sorted(rising))
        self.assertEqual(tension._calibrate(0.0), 0.0)
        self.assertEqual(tension._calibrate(1.0), 1.0)

    def test_it_lifts_the_low_end_rather_than_the_high(self):
        self.assertGreater(tension._calibrate(0.04) / 0.04, 4.0)
        self.assertLess(tension._calibrate(0.81) / 0.81, 1.2)

    @unittest.skipUnless(
        len(list(pathlib.Path("data/pgn").glob("*.pgn"))) >= 4,
        "real-game fixtures absent from data/pgn (gitignored)",
    )
    def test_the_corpus_ordering_survives_the_transform(self):
        """Monotone means no game overtakes another. Rescaling per game would
        make the Benoni and the Evans sound alike, which is the failure the axis
        exists to prevent."""
        order = ["spassky_petrosian_1969_closed.pgn", "karpov_kasparov_1991_kid.pgn",
                 "wc2021_game6.pgn", "kasparov_anand_1995_evans.pgn"]
        peaks = []
        for name in order:
            path = pathlib.Path("data/pgn") / name
            if not path.exists():
                self.skipTest(f"fixture absent (gitignored): {name}")
            track = tension.tension_track(
                features.annotate(ingest.ingest(path)))["plies"]
            peaks.append(max(e["tension"] for e in track))
        self.assertEqual(peaks, sorted(peaks, reverse=True), dict(zip(order, peaks)))
        self.assertGreater(peaks[-1], 0.2)      # the Evans is no longer silent

    def test_the_same_constant_is_applied_to_every_game(self):
        """No per-game fitting: emitted must be exactly raw ** GAMMA everywhere.

        A monotone transform cannot reorder anything, so ordering alone proves
        nothing. What has to hold is that one constant governs all games.
        """
        for sans, result in ((shuffle_moves(12), "1-0"),
                             (["e4", "e5", "Nf3", "Nc6", "Bc4"], "0-1")):
            with self.subTest(result=result):
                _, track = tracked(sans=sans, result=result)
                for entry in track:
                    raw = entry["tension"] ** (1.0 / tension.GAMMA)
                    self.assertAlmostEqual(
                        tension._calibrate(raw), entry["tension"], places=6)


class Plotting(unittest.TestCase):
    def test_plot_track_writes_a_figure(self):
        import tempfile

        track = record(
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

        track = record(sans=shuffle_moves(10))
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
        bare = record(sans=sans)
        annotated_eval = record(sans=sans, evals=["0.0"] * 12)
        self.assertNotIn("eval", bare["active_layers"])
        self.assertIn("eval", annotated_eval["active_layers"])

        peak_without = max(e["tension"] for e in bare["plies"])
        peak_calm = max(e["tension"] for e in annotated_eval["plies"])
        self.assertGreater(peak_without, peak_calm)
        # Raw ratio is 1/(1 - w_eval); the emitted axis is its square root.
        expected = (1.0 / (1.0 - tension.WEIGHTS["eval"])) ** tension.GAMMA
        self.assertAlmostEqual(peak_without / peak_calm, expected, places=2)

    def test_a_game_with_no_evaluations_is_still_coherent(self):
        """Three of eight fixtures have neither clocks nor evaluations."""
        path = pathlib.Path("data/pgn") / "kasparov_topalov_1999.pgn"
        if not path.exists():
            self.skipTest("real-game fixture absent from data/pgn (gitignored): "
                          "kasparov_topalov_1999.pgn")
        whole = tension.tension_track(features.annotate(ingest.ingest(path)))
        track = whole["plies"]
        values = [e["tension"] for e in track]
        self.assertGreater(max(values), 0.0)
        self.assertGreater(len(set(values)), len(values) // 4)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))
        self.assertTrue(any(e["cadence_strength"] > 0 for e in track))
        self.assertEqual(whole["active_layers"], ["board"])

    def test_volatile_evaluations_raise_tension(self):
        sans = shuffle_moves(12)
        _, calm = tracked(sans=sans, evals=["0.0"] * 12)
        _, wild = tracked(sans=sans,
                          evals=[str(v) for v in (0, 5, -5, 5, -5, 5,
                                                  -5, 5, -5, 5, -5, 5)])
        self.assertGreater(max(e["tension"] for e in wild),
                           max(e["tension"] for e in calm))


class HangingMaterialTerm(unittest.TestCase):
    """The normaliser and its route into tension. The measurement
    itself is a board feature and is tested with the others."""

    def test_it_reaches_tension(self):
        frames, baseline = tracked(sans=shuffle_moves(10))
        loose = copy.deepcopy(frames)
        for frame in loose:
            frame["hanging_material"] = tension.HANGING_FULL
        raised = tension.tension_track(loose)["plies"]
        self.assertGreater(max(e["tension"] for e in raised),
                           max(e["tension"] for e in baseline))

    def test_hanging_full_is_pinned(self):
        self.assertEqual(tension.HANGING_FULL, 12)
        self.assertEqual(tension._hanging({"hanging_material": 12}), 1.0)
        self.assertEqual(tension._hanging({"hanging_material": 6}), 0.5)


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
        track = tension.tension_track(frames)["plies"]
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
        changed = tension.tension_track(cramped)["plies"]
        self.assertLess(changed[0]["density"], baseline[0]["density"])


class ActiveLayers(unittest.TestCase):
    def test_is_a_per_game_manifest_not_a_per_ply_flag(self):
        """One answer per game, for the badge. Saying it per ply made the layer
        appear to switch on and off whenever a single reply was a premove."""
        bare = record(sans=shuffle_moves(6))
        self.assertEqual(bare["active_layers"], ["board"])
        self.assertNotIn("active_layers", bare["plies"][0])

        rich = record(
            sans=shuffle_moves(6),
            clocks=clocks_from_think([0, 0] + [1.0] * 4, start=600),
            time_control="600",
            evals=["0.1"] * 6,
        )
        self.assertEqual(rich["active_layers"],
                         ["board", "eval", "pressure", "think"])

    def test_a_single_premove_does_not_drop_the_think_layer(self):
        think = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        whole = record(sans=shuffle_moves(8),
                       clocks=clocks_from_think(think, start=600),
                       time_control="600")
        self.assertIn("think", whole["active_layers"])

    def test_a_signal_appearing_late_still_counts(self):
        """The manifest asks whether the game carried a signal at all.

        Reading it off the first ply would miss annotations that start later,
        and would miss the think layer in every game, since the opening ply of
        each colour has no baseline to measure a think time against.
        """
        whole = record(sans=shuffle_moves(8),
                       evals=[None, None, None, "0.4", "0.1", "0.3", "0.2", "0.5"])
        self.assertIn("eval", whole["active_layers"])

    def test_game_tempo_scale_is_recorded_once(self):
        whole = record(sans=shuffle_moves(6),
                       clocks=clocks_from_think([0, 0] + [2.0] * 4, start=600),
                       time_control="600")
        self.assertEqual(whole["game_tempo_scale"], 2.0)
        self.assertNotIn("game_tempo_scale", whole["plies"][0])


if __name__ == "__main__":
    unittest.main()
