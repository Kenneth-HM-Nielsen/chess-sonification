"""Phase 2 done criteria: board features, progress, pressure, think shape."""

from __future__ import annotations

import logging
import math
import pathlib
import statistics
import unittest

import chess

from src import features, ingest
from tests.helpers import build_pgn, clocks_from_think, shuffle_moves, write_pgn

logging.getLogger("src.ingest").setLevel(logging.CRITICAL)

WC_CONTROL = "40/7200:20/3600:900+30"
FIXTURES = pathlib.Path("data/pgn")


def annotated(**kwargs):
    return features.annotate(ingest.ingest(write_pgn(build_pgn(**kwargs))))


class BoardFeatures(unittest.TestCase):
    def test_does_not_mutate_the_callers_board(self):
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        before = board.fen()
        features.board_features(board)
        self.assertEqual(board.fen(), before)

    def test_locked_pawns_counts_both_colours(self):
        board = chess.Board("4k3/8/8/4p3/4P3/8/8/4K3 w - - 0 1")
        self.assertEqual(features.board_features(board)["locked_pawns"], 2)
        self.assertEqual(
            features.board_features(chess.Board())["locked_pawns"], 0
        )

    def test_pawn_tension_counts_each_pair_once(self):
        board = chess.Board("4k3/8/8/4p3/3P4/8/8/4K3 w - - 0 1")
        self.assertEqual(features.board_features(board)["pawn_tension"], 1)

    def test_king_pressure_counts_attackers_in_the_zone(self):
        quiet = features.board_features(chess.Board())
        self.assertEqual(quiet["king_pressure_w"], 0)
        self.assertEqual(quiet["king_pressure_b"], 0)
        board = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
        self.assertGreater(features.board_features(board)["king_pressure_w"], 0)

    def test_material_balance_is_white_pov(self):
        self.assertEqual(
            features.board_features(chess.Board())["material_balance"], 0
        )
        board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
        self.assertEqual(features.board_features(board)["material_balance"], 9)

    def test_centre_occupancy_is_per_side(self):
        board = chess.Board("4k3/8/8/3p4/3P4/8/8/4K3 w - - 0 1")
        result = features.board_features(board)
        self.assertEqual(result["centre_occupancy_w"], 1)
        self.assertEqual(result["centre_occupancy_b"], 1)

    def test_mobility_excludes_capturing_a_king(self):
        # Black has just given check; behind a null move White must not be
        # credited with a move that captures the black king.
        board = chess.Board("4k3/8/8/8/7q/8/8/4K3 w - - 0 1")
        board.push_san("Kd1")
        board.push_san("Qd4+")
        result = features.board_features(board)
        self.assertGreater(result["mobility_b"], 0)
        # No move in the reported count may land on a king square.
        naive = board.copy()
        naive.push(chess.Move.null())
        king_captures = sum(
            1
            for move in naive.legal_moves
            if naive.piece_type_at(move.to_square) == chess.KING
        )
        self.assertEqual(result["mobility_b"], naive.legal_moves.count() - king_captures)
        self.assertGreaterEqual(king_captures, 1)

    def test_forced_uses_the_position_before_the_move(self):
        frames = annotated(sans=shuffle_moves(4))
        self.assertFalse(any(f["forced"] for f in frames))


class ProgressFeatures(unittest.TestCase):
    def test_counters_track_and_reset(self):
        frames = annotated(sans=["Nf3", "Nf6", "e4", "d5", "exd5"])
        self.assertEqual(frames[0]["plies_since_pawn_move"], 1)
        self.assertEqual(frames[1]["plies_since_pawn_move"], 2)
        self.assertEqual(frames[2]["plies_since_pawn_move"], 0)
        self.assertEqual(frames[4]["plies_since_capture"], 0)
        self.assertEqual(frames[4]["halfmove_clock"], 0)

    def test_halfmove_clock_climbs_when_nothing_irreversible_happens(self):
        frames = annotated(sans=shuffle_moves(8))
        self.assertEqual(frames[-1]["halfmove_clock"], 8)

    def test_repetition_needs_the_move_stack(self):
        frames = annotated(sans=shuffle_moves(8))
        self.assertTrue(any(f["repetition_2"] for f in frames))

    def test_repetition_3_was_removed(self):
        frames = annotated(sans=shuffle_moves(4))
        self.assertNotIn("repetition_3", frames[0])


class Replay(unittest.TestCase):
    def test_non_standard_start_replays_correctly(self):
        fen = "8/5k2/8/8/8/8/5K2/6R1 w - - 0 1"
        frames = features.annotate(
            ingest.ingest(write_pgn(build_pgn(["Rg7+", "Kf6"], fen=fen)))
        )
        # Rook and two kings only: White is a rook up, and the move gives check.
        self.assertEqual(frames[0]["material_balance"], 5)
        self.assertTrue(frames[0]["is_check"])

    def test_inconsistent_frames_raise_rather_than_fabricate(self):
        frames = ingest.ingest(write_pgn(build_pgn(shuffle_moves(4))))
        frames[0]["uci"] = "e2e4"
        frames[0]["start_fen"] = chess.STARTING_FEN
        frames[1]["uci"] = "a7a5"
        with self.assertRaises(ValueError):
            features.annotate(frames)

    def test_empty_frames(self):
        self.assertEqual(features.annotate([]), [])


class TimePressure(unittest.TestCase):
    def test_initial_budget_scales_with_the_control(self):
        cases = {WC_CONTROL: 180.0, "180+2": 6.5, "60+0": 1.5, "600": 15.0}
        for header, expected in cases.items():
            with self.subTest(header=header):
                periods = ingest.parse_time_control(header)
                self.assertAlmostEqual(features.initial_budget(periods), expected)

    def test_unknown_allocation_gives_no_initial_budget(self):
        self.assertIsNone(features.initial_budget(ingest.parse_time_control("-")))

    def test_moves_to_threshold_counts_the_move_about_to_be_played(self):
        thresholds = ingest.period_thresholds(ingest.parse_time_control(WC_CONTROL))
        self.assertEqual(features.moves_to_threshold(thresholds, 39), 2)
        self.assertEqual(features.moves_to_threshold(thresholds, 40), 1)
        self.assertEqual(features.moves_to_threshold(thresholds, 41), 20)
        self.assertEqual(features.moves_to_threshold(thresholds, 61),
                         features.NOMINAL_HORIZON)

    def test_pressure_starts_near_zero_and_stays_bounded(self):
        for control, start, increment in ((WC_CONTROL, 7200, 0), ("180+2", 180, 2)):
            with self.subTest(control=control):
                plies = 20
                think = [0.0, 0.0] + [1.0] * (plies - 2)
                clocks = clocks_from_think(think, start=start, increment=increment)
                frames = annotated(sans=shuffle_moves(plies), clocks=clocks,
                                   time_control=control)
                self.assertLess(frames[0]["time_pressure_w"], 0.05)
                values = [f["time_pressure_w"] for f in frames
                          if f["time_pressure_w"] is not None]
                self.assertTrue(all(0.0 <= v <= 1.0 for v in values))

    def test_both_sides_carried_and_forward_filled(self):
        clocks = clocks_from_think([0, 0, 5, 5], start=600)
        frames = annotated(sans=shuffle_moves(4), clocks=clocks, time_control="600")
        self.assertIsNone(frames[0]["time_pressure_b"])
        self.assertIsNotNone(frames[0]["time_pressure_w"])
        # White's value persists onto Black's ply.
        self.assertEqual(frames[1]["time_pressure_w"], frames[0]["time_pressure_w"])

    def test_null_propagates_when_there_are_no_clocks(self):
        frames = annotated(sans=shuffle_moves(4))
        for field in ("moves_to_threshold_w", "budget_w", "time_pressure_w",
                      "think_relative", "game_tempo_scale"):
            with self.subTest(field=field):
                self.assertTrue(all(f[field] is None for f in frames))

    def test_boundary_ply_pairs_a_post_credit_clock_with_a_post_credit_horizon(self):
        clocks = [600, 540, 870, 780]
        frames = annotated(sans=shuffle_moves(4), clocks=clocks,
                           time_control="2/600:300+10")
        boundary = frames[2]
        self.assertTrue(boundary["period_boundary"])
        # Horizon is the open-ended period, not the threshold just crossed.
        self.assertEqual(boundary["moves_to_threshold_w"], features.NOMINAL_HORIZON)
        self.assertAlmostEqual(
            boundary["budget_w"], 870 / features.NOMINAL_HORIZON + 10, places=3
        )


class ThinkShape(unittest.TestCase):
    def test_premoves_are_not_decisions(self):
        think = [0.0, 0.0, 0.05, 5.0]
        clocks = clocks_from_think(think, start=600)
        frames = annotated(sans=shuffle_moves(4), clocks=clocks, time_control="600")
        self.assertFalse(frames[2]["decision"])
        self.assertIsNone(frames[2]["think_relative"])
        self.assertTrue(frames[3]["decision"])
        self.assertIsNotNone(frames[3]["think_relative"])

    def test_premove_floor_is_one_constant(self):
        self.assertEqual(features.PREMOVE_FLOOR_S, 0.15)

    def test_game_tempo_scale_is_the_median_not_the_mean(self):
        think = [0.0, 0.0] + [1.0] * 8 + [600.0, 1.0]
        clocks = clocks_from_think(think, start=7200)
        frames = annotated(sans=shuffle_moves(len(think)), clocks=clocks,
                           time_control="7200")
        decisions = [f["think_time"] for f in frames if f["decision"]]
        self.assertEqual(frames[0]["game_tempo_scale"],
                         round(statistics.median(decisions), 3))
        self.assertLess(frames[0]["game_tempo_scale"], statistics.mean(decisions))

    def test_early_plies_are_scaled_by_the_whole_game_median(self):
        think = [0.0, 0.0, 4.0, 1.0, 1.0, 1.0]
        clocks = clocks_from_think(think, start=600)
        frames = annotated(sans=shuffle_moves(6), clocks=clocks, time_control="600")
        white = [f for f in frames if f["color"] == "w" and f["decision"]]
        median = statistics.median([f["think_time"] for f in frames
                                    if f["color"] == "w" and f["decision"]])
        self.assertAlmostEqual(
            white[0]["think_relative"],
            round(math.log(white[0]["think_time"] / median), 4),
            places=3,
        )

    def test_baseline_excludes_the_ply_being_scored(self):
        # A long think must be measured against what came before it. With a
        # self-inclusive window the ratio is damped; trailing gives the full one.
        window = features.ROLLING_WINDOW_MOVES
        think = [0.0, 0.0] + [1.0] * (window * 2) + [15.0, 1.0]
        clocks = clocks_from_think(think, start=7200)
        frames = annotated(sans=shuffle_moves(len(think)), clocks=clocks,
                           time_control="7200")
        spike = max((f for f in frames if f["think_relative"] is not None),
                    key=lambda f: f["think_relative"])
        self.assertAlmostEqual(spike["think_relative"], round(math.log(15.0), 4),
                               places=3)
        self.assertLess(spike["think_relative"], features.THINK_RELATIVE_CLAMP)

    def test_clamped_to_a_sane_range(self):
        think = [0.0, 0.0] + [1.0] * 12 + [100000.0, 1.0]
        clocks = clocks_from_think(think, start=1000000)
        frames = annotated(sans=shuffle_moves(len(think)), clocks=clocks,
                           time_control="1000000")
        values = [f["think_relative"] for f in frames
                  if f["think_relative"] is not None]
        self.assertTrue(all(abs(v) <= features.THINK_RELATIVE_CLAMP for v in values))
        self.assertEqual(max(values), features.THINK_RELATIVE_CLAMP)

    def test_a_game_of_only_premoves_does_not_divide_by_zero(self):
        think = [0.0] * 10
        clocks = clocks_from_think(think, start=60)
        frames = annotated(sans=shuffle_moves(10), clocks=clocks, time_control="60")
        self.assertIsNone(frames[0]["game_tempo_scale"])
        self.assertTrue(all(f["think_relative"] is None for f in frames))


@unittest.skipUnless(
    (FIXTURES / "karpov_kasparov_1991_kid.pgn").exists()
    and (FIXTURES / "kasparov_anand_1995_evans.pgn").exists(),
    "local PGN fixtures not present (data/pgn is gitignored)",
)
class Phase2DoneCriteria(unittest.TestCase):
    """The stated criterion: a closed King's Indian against an open gambit."""

    @classmethod
    def setUpClass(cls):
        cls.kid = features.annotate(
            ingest.ingest(FIXTURES / "karpov_kasparov_1991_kid.pgn")
        )
        cls.evans = features.annotate(
            ingest.ingest(FIXTURES / "kasparov_anand_1995_evans.pgn")
        )

    @staticmethod
    def _window(frames, low, high, field):
        return statistics.mean(
            f[field] for f in frames if low <= (f["ply"] + 1) // 2 <= high
        )

    def test_locked_pawns_separates_the_two_games(self):
        kid = self._window(self.kid, 8, 22, "locked_pawns")
        evans = self._window(self.evans, 8, 22, "locked_pawns")
        self.assertGreater(kid, 3.0)
        self.assertEqual(evans, 0.0)

    def test_pawn_tension_separates_the_two_games(self):
        self.assertGreater(
            self._window(self.kid, 8, 22, "pawn_tension"),
            4 * self._window(self.evans, 8, 22, "pawn_tension"),
        )

    def test_mobility_does_not_separate_them(self):
        """Documents the measured failure that moved mobility to density.

        The spec expected materially lower mobility in the King's Indian. It is
        not there: a closed centre pushes play to the wings without reducing the
        legal move count.
        """
        kid = self._window(self.kid, 8, 22, "mobility_w")
        kid += self._window(self.kid, 8, 22, "mobility_b")
        evans = self._window(self.evans, 8, 22, "mobility_w")
        evans += self._window(self.evans, 8, 22, "mobility_b")
        self.assertGreater(kid / evans, 0.85)


if __name__ == "__main__":
    unittest.main()
