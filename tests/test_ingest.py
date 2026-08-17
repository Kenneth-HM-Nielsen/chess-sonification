"""Phase 1 done criteria: PGN parsing, clocks, evals, multi-period controls."""

from __future__ import annotations

import logging
import unittest

from src import ingest
from tests.helpers import build_pgn, clocks_from_think, shuffle_moves, write_pgn

logging.getLogger("src.ingest").setLevel(logging.CRITICAL)

WC_CONTROL = "40/7200:20/3600:900+30"


class ParseTimeControl(unittest.TestCase):
    def test_spec_examples(self):
        cases = {
            WC_CONTROL: [(40, 7200, 0), (20, 3600, 0), (None, 900, 30)],
            "180+2": [(None, 180, 2)],
            "600": [(None, 600, 0)],
            "-": [(None, None, 0)],
            "?": [(None, None, 0)],
            "*180": [(None, None, 0)],
            None: [(None, None, 0)],
        }
        for header, expected in cases.items():
            with self.subTest(header=header):
                self.assertEqual(ingest.parse_time_control(header), expected)

    def test_malformed_falls_back_to_unknown(self):
        for header in ("junk", "40/abc", "", "G/90+30"):
            with self.subTest(header=header):
                self.assertEqual(
                    ingest.parse_time_control(header), [(None, None, 0)]
                )

    def test_increment_is_per_period(self):
        periods = ingest.parse_time_control(WC_CONTROL)
        for move, expected in ((1, 0), (40, 0), (60, 0), (61, 30), (136, 30)):
            with self.subTest(move=move):
                self.assertEqual(ingest.increment_for_move(periods, move), expected)

    def test_thresholds_are_derived_not_hardcoded(self):
        self.assertEqual(
            ingest.period_thresholds(ingest.parse_time_control(WC_CONTROL)), [40, 60]
        )
        self.assertEqual(
            ingest.period_thresholds(ingest.parse_time_control("40/9000+30:1800+30")),
            [40],
        )
        self.assertEqual(
            ingest.period_thresholds(ingest.parse_time_control("180+2")), []
        )

    def test_boundary_and_credited_base_agree(self):
        periods = ingest.parse_time_control(WC_CONTROL)
        for move in range(1, 140):
            boundary = ingest.is_period_boundary(periods, move)
            base = ingest.credited_base(periods, move)
            self.assertEqual(boundary, base is not None, f"move {move}")
        self.assertEqual(ingest.credited_base(periods, 40), 3600)
        self.assertEqual(ingest.credited_base(periods, 60), 900)


class ThinkTime(unittest.TestCase):
    def test_first_move_of_each_colour_has_no_baseline(self):
        frames = ingest.ingest(
            write_pgn(build_pgn(shuffle_moves(6), clocks=[600 - i for i in range(6)],
                                time_control="600"))
        )
        self.assertIsNone(frames[0]["think_time"])
        self.assertIsNone(frames[1]["think_time"])
        self.assertIsNotNone(frames[2]["think_time"])

    def test_increment_is_added(self):
        think = [0, 0, 5.0, 7.0]
        clocks = clocks_from_think(think, start=180, increment=2)
        frames = ingest.ingest(
            write_pgn(build_pgn(shuffle_moves(4), clocks=clocks, time_control="180+2"))
        )
        self.assertAlmostEqual(frames[2]["think_time"], 5.0, places=3)
        self.assertAlmostEqual(frames[3]["think_time"], 7.0, places=3)

    def test_negative_is_clamped_when_allocation_known(self):
        clocks = [600, 600, 590, 700]
        frames = ingest.ingest(
            write_pgn(build_pgn(shuffle_moves(4), clocks=clocks, time_control="600+5"))
        )
        self.assertEqual(frames[3]["think_time"], 0.0)

    def test_unknown_allocation_reports_none_not_zero(self):
        clocks = [600, 540, 870, 780]
        frames = ingest.ingest(
            write_pgn(build_pgn(shuffle_moves(4), clocks=clocks, time_control="-"))
        )
        self.assertIsNone(frames[2]["think_time"])

    def test_clock_gap_is_not_measured_across(self):
        clocks = [600, 595, None, None, 540, 540]
        frames = ingest.ingest(
            write_pgn(build_pgn(shuffle_moves(6), clocks=clocks, time_control="600"))
        )
        self.assertTrue(all(f["think_time"] is None for f in frames))

    def test_period_boundary_is_reconstructed_not_clamped(self):
        # Control at move 2; the clock rises by the new period's allocation.
        clocks = [600, 540, 870, 780]
        frames = ingest.ingest(
            write_pgn(build_pgn(shuffle_moves(4), clocks=clocks,
                                time_control="2/600:300+10"))
        )
        self.assertTrue(frames[2]["period_boundary"])
        self.assertTrue(frames[3]["period_boundary"])
        # 600 - (870 - 300) + 0 = 30
        self.assertAlmostEqual(frames[2]["think_time"], 30.0, places=3)

    def test_impossible_boundary_reconstruction_is_none(self):
        clocks = [600, 540, 1200, 540]
        frames = ingest.ingest(
            write_pgn(build_pgn(shuffle_moves(4), clocks=clocks,
                                time_control="2/600:300+10"))
        )
        self.assertIsNone(frames[2]["think_time"])


class Evaluations(unittest.TestCase):
    def test_mate_becomes_plus_minus_ten_thousand(self):
        frames = ingest.ingest(
            write_pgn(build_pgn(shuffle_moves(3), evals=["#2", "#-1", "3.5"]))
        )
        self.assertEqual(frames[0]["eval_cp"], 10000)
        self.assertEqual(frames[1]["eval_cp"], -10000)
        self.assertEqual(frames[2]["eval_cp"], 350)

    def test_absent_evals_stay_none(self):
        frames = ingest.ingest(write_pgn(build_pgn(shuffle_moves(4))))
        self.assertTrue(all(f["eval_cp"] is None for f in frames))


class FrameShape(unittest.TestCase):
    def test_records_carry_the_specified_fields(self):
        frames = ingest.ingest(
            write_pgn(build_pgn(shuffle_moves(2), clocks=[600, 599],
                                time_control="600"))
        )
        expected = {
            "ply", "san", "uci", "color", "piece", "from_sq", "to_sq",
            "clock_remaining", "think_time", "eval_cp", "fen_after",
            "period_boundary", "time_control", "start_fen", "think_time_clamped",
            "result",
        }
        self.assertEqual(set(frames[0]), expected)
        self.assertEqual(frames[0]["color"], "w")
        self.assertEqual(frames[1]["color"], "b")
        self.assertEqual(frames[0]["piece"], "N")

    def test_non_standard_start_is_recorded(self):
        fen = "8/5k2/8/8/8/8/5K2/6R1 w - - 0 1"
        frames = ingest.ingest(write_pgn(build_pgn(["Rg7+", "Kf6"], fen=fen)))
        self.assertEqual(frames[0]["start_fen"], fen)

    def test_result_comes_from_the_header(self):
        pgn = '[Event "T"]\n[Result "0-1"]\n\n1. Nf3 Nf6 0-1\n'
        self.assertEqual(ingest.ingest(write_pgn(pgn))[0]["result"], "0-1")

    def test_an_unrecognised_result_is_reported(self):
        pgn = '[Event "T"]\n[Result "1\u20130"]\n\n1. Nf3 Nf6 *\n'
        with self.assertLogs("src.ingest", level="WARNING") as captured:
            frames = ingest.ingest(write_pgn(pgn))
        self.assertIn("unrecognised Result", "\n".join(captured.output))
        self.assertEqual(frames[0]["result"], "1\u20130")

    def test_no_game_raises(self):
        with self.assertRaises(ValueError):
            ingest.ingest(write_pgn(""))


if __name__ == "__main__":
    unittest.main()
