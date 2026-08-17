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
SUDDEN = [(None, 600, 0)]
FIXTURES = pathlib.Path("data/pgn")
SYNTHETIC = pathlib.Path(__file__).parent / "fixtures"

# Real historical games, gitignored. Every invariant runs without them.
REAL_GAMES = ("karpov_kasparov_1991_kid.pgn", "kasparov_anand_1995_evans.pgn")
MISSING_REAL = [name for name in REAL_GAMES if not (FIXTURES / name).exists()]


def synthetic(name):
    return features.annotate(ingest.ingest(SYNTHETIC / name))


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

    def test_locked_pawns_separates_a_blocked_centre_from_an_open_one(self):
        """The Phase 2 divergence, without depending on gitignored fixtures.

        Four head-to-head pawn pairs against a board with no pawn contact at all.
        This is the measure that actually distinguishes a closed position;
        mobility does not (see Phase2DoneCriteria).
        """
        blocked = chess.Board("4k3/pp3ppp/2p1p3/2PpPp2/3P1P2/8/PP4PP/4K3 w - - 0 1")
        open_board = chess.Board("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1")
        self.assertEqual(features.board_features(blocked)["locked_pawns"], 8)
        self.assertEqual(features.board_features(open_board)["locked_pawns"], 0)

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

    def test_king_pressure_includes_the_king_square_itself(self):
        # A knight on c2 attacks e1 and nothing adjacent to it, so the count is
        # non-zero only if the king's own square is part of the zone.
        board = chess.Board("4k3/8/8/8/8/8/2n5/4K3 w - - 0 1")
        self.assertEqual(features.board_features(board)["king_pressure_w"], 1)

    def test_material_balance_weights_every_piece_type(self):
        self.assertEqual(
            features.board_features(chess.Board())["material_balance"], 0
        )
        cases = {
            "4k3/8/8/8/8/8/P7/4K3 w - - 0 1": 1,
            "4k3/8/8/8/8/8/8/N3K3 w - - 0 1": 3,
            "4k3/8/8/8/8/8/8/B3K3 w - - 0 1": 3,
            "4k3/8/8/8/8/8/8/R3K3 w - - 0 1": 5,
            "4k3/8/8/8/8/8/8/3QK3 w - - 0 1": 9,
            "3qk3/8/8/8/8/8/8/4K3 w - - 0 1": -9,
        }
        for fen, expected in cases.items():
            with self.subTest(fen=fen):
                self.assertEqual(
                    features.board_features(chess.Board(fen))["material_balance"],
                    expected,
                )

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

    def test_forced_is_true_when_only_one_move_is_legal(self):
        # Black is in check from Rh7; g7 and g8 are covered by the white king,
        # so Kxh7 is the only legal reply.
        board = chess.Board("7k/5K1R/8/8/8/8/8/8 b - - 0 1")
        self.assertEqual(board.legal_moves.count(), 1)
        move = board.parse_san("Kxh7")
        self.assertTrue(features.move_features(board, move)["forced"])

    def test_forced_reads_the_position_before_not_after(self):
        board = chess.Board("7k/5K1R/8/8/8/8/8/8 b - - 0 1")
        move = board.parse_san("Kxh7")
        after = board.copy()
        after.push(move)
        # The resulting position has many legal moves, so a check made after the
        # push would report False where the spec requires True.
        self.assertGreater(after.legal_moves.count(), 1)
        self.assertTrue(features.move_features(board, move)["forced"])

    def test_forced_is_false_in_an_ordinary_position(self):
        board = chess.Board()
        self.assertFalse(features.move_features(board, board.parse_san("e4"))["forced"])

    def test_king_capture_filter_is_a_no_op_in_ordinary_positions(self):
        for fen in (chess.STARTING_FEN,
                    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"):
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                self.assertEqual(features._move_count(board),
                                 board.legal_moves.count())


class HangingMaterial(unittest.TestCase):
    """The fifth tension source: the transient half of tactical volatility."""

    def test_nothing_is_loose_when_nothing_is_attacked(self):
        self.assertEqual(features._hanging_material(chess.Board()), 0)

    def test_an_adequately_defended_piece_is_not_loose(self):
        """Reaches the comparison with both halves false, which the starting
        position never does -- there, no piece is attacked at all."""
        # Nd4 attacked once by Nf5 and defended once by the c3 pawn. Equal
        # numbers, and the cheapest attacker is worth the same as the piece.
        board = chess.Board("4k3/8/8/5n2/3N4/2P5/8/4K3 w - - 0 1")
        self.assertEqual(features._hanging_material(board), 3)   # only Nf5

    def test_an_equal_valued_attacker_does_not_make_a_piece_loose(self):
        """`cheapest < value`, not `<=`: a knight taken by a knight is a trade."""
        board = chess.Board("4k3/8/8/5n2/3N4/2P5/8/4K3 w - - 0 1")
        self.assertEqual(features._hanging_material(board), 3)
        self.assertEqual(features.PIECE_VALUES[chess.KNIGHT], 3)

    def test_both_colours_are_summed(self):
        """A loose white rook, a loose black rook and a loose white knight."""
        board = chess.Board("r3k1r1/8/8/8/8/8/8/R3K1N1 w - - 0 1")
        self.assertEqual(features._hanging_material(board), 13)

    def test_an_undefended_attacked_piece_is_loose(self):
        # Knight on g1 attacked down the file by the rook, defended by nobody.
        board = chess.Board("4k1r1/8/8/8/8/8/8/4K1N1 w - - 0 1")
        self.assertEqual(features._hanging_material(board), 3)

    def test_a_cheaper_attacker_makes_a_defended_piece_loose(self):
        # Rook on d5 attacked by the c6 pawn and defended by the d1 rook: still
        # loose, because the pawn is worth less than what it attacks.
        board = chess.Board("4k3/8/2p5/3R4/8/8/8/3RK3 w - - 0 1")
        self.assertEqual(features._hanging_material(board), 5)

    def test_a_king_attacker_is_never_the_cheap_one(self):
        # Queen attacked only by the enemy king and defended once: a king can
        # capture but can never be traded, so it is not the cheap attacker.
        board = chess.Board("8/8/3k4/3Q4/8/8/8/3RK3 b - - 0 1")
        self.assertEqual(features._hanging_material(board), 0)


class MoveFeaturesReachTheFrames(unittest.TestCase):
    """Testing move_features directly leaves its wiring into annotate uncovered."""

    def test_forced_and_capture_arrive_on_the_annotated_frame(self):
        # Rh7+ leaves Kxh7 as Black's only legal reply, and it is a capture.
        fen = "7k/5K2/8/8/8/8/8/7R w - - 0 1"
        frames = features.annotate(
            ingest.ingest(write_pgn(build_pgn(["Rh7+", "Kxh7"], fen=fen)))
        )
        self.assertFalse(frames[0]["forced"])
        self.assertFalse(frames[0]["is_capture"])
        self.assertTrue(frames[1]["forced"])
        self.assertTrue(frames[1]["is_capture"])

    def test_promotion_arrives_on_the_annotated_frame(self):
        fen = "7k/P7/8/8/8/5K2/8/8 w - - 0 1"
        frames = features.annotate(
            ingest.ingest(write_pgn(build_pgn(["a8=Q", "Kh7"], fen=fen)))
        )
        self.assertTrue(frames[0]["is_promotion"])
        self.assertFalse(frames[1]["is_promotion"])


class ProgressFeatures(unittest.TestCase):
    def test_halfmove_clock_tracks_and_resets(self):
        frames = annotated(sans=["Nf3", "Nf6", "e4", "d5", "exd5"])
        self.assertEqual(frames[0]["halfmove_clock"], 1)
        self.assertEqual(frames[1]["halfmove_clock"], 2)
        self.assertEqual(frames[2]["halfmove_clock"], 0)   # pawn move
        self.assertEqual(frames[4]["halfmove_clock"], 0)   # capture

    def test_redundant_counters_were_removed(self):
        frames = annotated(sans=shuffle_moves(4))
        self.assertNotIn("plies_since_pawn_move", frames[0])
        self.assertNotIn("plies_since_capture", frames[0])

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

    def test_moves_to_threshold_counts_the_moves_still_ahead(self):
        periods = ingest.parse_time_control(WC_CONTROL)
        self.assertEqual(features.moves_to_threshold(periods, 38), 2)
        self.assertEqual(features.moves_to_threshold(periods, 39), 1)
        # The control has just been passed, so the horizon runs to the next one.
        self.assertEqual(features.moves_to_threshold(periods, 40), 20)
        self.assertEqual(features.moves_to_threshold(periods, 41), 19)
        self.assertEqual(features.moves_to_threshold(periods, 60),
                         features.NOMINAL_HORIZON)

    def test_horizon_decrements_by_one_every_move(self):
        """No two consecutive moves may report the same bounded horizon.

        Pairing a post-credit clock with a pre-credit horizon made moves 40 and
        41 both report 20, understating every budget before a control.
        """
        periods = ingest.parse_time_control(WC_CONTROL)
        horizons = [features.moves_to_threshold(periods, m) for m in range(1, 60)]
        for earlier, later in zip(horizons, horizons[1:]):
            if earlier != features.NOMINAL_HORIZON:
                self.assertEqual(later, earlier - 1 if earlier > 1 else 20)

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

    def test_boundary_into_an_open_ended_period(self):
        clocks = [600, 540, 870, 780]
        frames = annotated(sans=shuffle_moves(4), clocks=clocks,
                           time_control="2/600:300+10")
        boundary = frames[2]
        self.assertTrue(boundary["period_boundary"])
        self.assertEqual(boundary["moves_to_threshold_w"], features.NOMINAL_HORIZON)
        self.assertAlmostEqual(
            boundary["budget_w"], 870 / features.NOMINAL_HORIZON + 10, places=3
        )

    def test_boundary_into_a_bounded_period(self):
        """The classical case: move 40 credits time for the 20 moves to move 60.

        A credited clock divided by the threshold just passed would report the
        whole fresh hour as one move's budget and read as zero pressure.
        """
        control = "3/600:2/300:60+5"
        clocks = [590, 585, 580, 575, 870, 865, 860, 855, 900, 895]
        frames = annotated(sans=shuffle_moves(10), clocks=clocks,
                           time_control=control)
        boundary = frames[4]                       # White's move 3
        self.assertTrue(boundary["period_boundary"])
        # Two moves remain before the next control at move 5, not zero or one.
        self.assertEqual(boundary["moves_to_threshold_w"], 2)
        self.assertAlmostEqual(boundary["budget_w"], 870 / 2, places=3)
        self.assertLess(frames[6]["moves_to_threshold_w"],
                        boundary["moves_to_threshold_w"])

    def test_pressure_is_none_when_the_allocation_is_unknown(self):
        """Clocks present but no TimeControl: pressure must not fabricate a 0."""
        clocks = [600, 595, 590, 585]
        frames = annotated(sans=shuffle_moves(4), clocks=clocks, time_control="-")
        self.assertIsNone(frames[0]["time_pressure_w"])
        self.assertIsNone(frames[3]["time_pressure_b"])
        self.assertTrue(all(f["time_pressure_w"] is None for f in frames))


class ThinkShape(unittest.TestCase):
    def test_the_decision_states_are_spelled_as_frames_carry_them(self):
        """Pinned to literals, because these strings leave the process.

        Every other assertion here compares a frame against the same constant
        the code wrote into it, which holds for any spelling including the
        boolean-era `"decision"`. Frames are written to JSON and read back, and
        the renderer gates reverb on these exact values, so the spelling is part
        of the contract and not an implementation detail.
        """
        self.assertEqual(
            (features.DECIDED, features.PREMOVE, features.UNKNOWN),
            ("decided", "premove", "unknown"),
        )
        frames = annotated(sans=shuffle_moves(2))
        self.assertIn("decision_state", frames[0])
        self.assertNotIn("decision", frames[0])

    def test_the_floor_itself_counts_as_a_decision(self):
        """Spec says non-decision when think falls *below* the floor.

        Pins both the constant and the comparison: at exactly 0.15 the ply is a
        decision, one hundredth under it is not.
        """
        think = [0.0, 0.0, 0.15, 0.14]
        clocks = clocks_from_think(think, start=600)
        frames = annotated(sans=shuffle_moves(4), clocks=clocks, time_control="600")
        self.assertAlmostEqual(frames[2]["think_time"], 0.15, places=3)
        self.assertEqual(frames[2]["decision_state"], features.DECIDED)
        self.assertEqual(frames[3]["decision_state"], features.PREMOVE)

    def test_premove_floor_sits_between_these_two_think_times(self):
        """Pins the floor behaviourally, either side of 0.15s."""
        think = [0.0, 0.0, 0.14, 0.16, 0.16, 0.14]
        clocks = clocks_from_think(think, start=600)
        frames = annotated(sans=shuffle_moves(6), clocks=clocks, time_control="600")
        self.assertAlmostEqual(frames[2]["think_time"], 0.14, places=2)
        self.assertAlmostEqual(frames[3]["think_time"], 0.16, places=2)
        self.assertEqual(frames[2]["decision_state"], features.PREMOVE)
        self.assertIsNone(frames[2]["think_relative"])
        self.assertEqual(frames[3]["decision_state"], features.DECIDED)
        self.assertIsNotNone(frames[3]["think_relative"])

    def test_premove_floor_is_not_tuned_per_time_control(self):
        think = [0.0, 0.0, 0.14, 0.16]
        for control, start in (("600", 600), ("180+2", 180), ("60+0", 60)):
            with self.subTest(control=control):
                increment = 2 if control == "180+2" else 0
                clocks = clocks_from_think(think, start=start, increment=increment)
                frames = annotated(sans=shuffle_moves(4), clocks=clocks,
                                   time_control=control)
                self.assertEqual(frames[2]["decision_state"], features.PREMOVE)
                self.assertEqual(frames[3]["decision_state"], features.DECIDED)

    def test_game_tempo_scale_is_the_median_not_the_mean(self):
        think = [0.0, 0.0] + [1.0] * 8 + [600.0, 1.0]
        clocks = clocks_from_think(think, start=7200)
        frames = annotated(sans=shuffle_moves(len(think)), clocks=clocks,
                           time_control="7200")
        decisions = [f["think_time"] for f in frames if f["decision_state"] == features.DECIDED]
        self.assertEqual(frames[0]["game_tempo_scale"],
                         round(statistics.median(decisions), 3))
        self.assertLess(frames[0]["game_tempo_scale"], statistics.mean(decisions))

    def test_partial_window_falls_back_to_the_whole_game_median(self):
        """Before the window fills, the baseline is the whole-game median.

        A partial-window median would use only what has been seen so far. White's
        decisions are 10, 1, 1, 1: the whole-game median is 1, so the second
        decision reports log(1) = 0, where a partial window over [10] would give
        log(1/10). The spec forbids leaving early moves unscaled, and forbids
        this cheaper substitute.
        """
        think = [0.0, 0.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        clocks = clocks_from_think(think, start=600)
        frames = annotated(sans=shuffle_moves(len(think)), clocks=clocks,
                           time_control="600")
        white = [f for f in frames if f["color"] == "w" and f["decision_state"] == features.DECIDED]
        self.assertAlmostEqual(white[0]["think_relative"], round(math.log(10.0), 4),
                               places=3)
        self.assertAlmostEqual(white[1]["think_relative"], 0.0, places=3)

    def test_early_plies_are_scaled_by_the_whole_game_median(self):
        think = [0.0, 0.0, 4.0, 1.0, 1.0, 1.0]
        clocks = clocks_from_think(think, start=600)
        frames = annotated(sans=shuffle_moves(6), clocks=clocks, time_control="600")
        white = [f for f in frames if f["color"] == "w" and f["decision_state"] == features.DECIDED]
        median = statistics.median([f["think_time"] for f in frames
                                    if f["color"] == "w" and f["decision_state"] == features.DECIDED])
        self.assertAlmostEqual(
            white[0]["think_relative"],
            round(math.log(white[0]["think_time"] / median), 4),
            places=3,
        )

    def test_baseline_excludes_the_ply_being_scored(self):
        """The scored ply must not sit in its own baseline window.

        Built so the two conventions cannot agree: White's twelve preceding
        decisions are six 1s and six 100s, whose median is 50.5. Including the
        100s spike would push the median to 100 and report log(1) = 0, damping
        exactly the outlier the measure exists to find. The fixture is derived
        from ROLLING_WINDOW_MOVES, so it pins the trailing behaviour but
        deliberately not the window length, which no specification fixes.
        """
        window = features.ROLLING_WINDOW_MOVES
        half = window // 2
        think = [0.0, 0.0]
        for value in [1.0] * half + [100.0] * half:
            think += [value, 1.0]          # White's decision, then Black's
        think += [100.0, 1.0]              # the spike, on White
        clocks = clocks_from_think(think, start=100000)
        frames = annotated(sans=shuffle_moves(len(think)), clocks=clocks,
                           time_control="100000")

        white = [f for f in frames if f["color"] == "w" and f["decision_state"] == features.DECIDED]
        self.assertEqual(len(white), window + 1)
        spike = white[-1]
        self.assertAlmostEqual(spike["think_time"], 100.0, places=2)
        self.assertAlmostEqual(spike["think_relative"],
                               round(math.log(100.0 / 50.5), 4), places=2)
        self.assertGreater(spike["think_relative"], 0.5)

    def test_clamped_relative_think_is_bounded(self):
        think = [0.0, 0.0] + [1.0] * 12 + [100000.0, 1.0]
        clocks = clocks_from_think(think, start=1000000)
        frames = annotated(sans=shuffle_moves(len(think)), clocks=clocks,
                           time_control="1000000")
        values = [f["think_relative"] for f in frames
                  if f["think_relative"] is not None]
        self.assertEqual(max(values), features.THINK_RELATIVE_CLAMP)

    def test_premoves_do_not_enter_the_median(self):
        """A premove is not a think, so it must not drag the tempo scale down."""
        think = [0.0, 0.0] + [0.0, 0.0] * 4 + [5.0, 5.0] * 3
        clocks = clocks_from_think(think, start=600)
        frames = annotated(sans=shuffle_moves(len(think)), clocks=clocks,
                           time_control="600")
        self.assertEqual(frames[0]["game_tempo_scale"], 5.0)

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


class SyntheticFixtures(unittest.TestCase):
    """Invariants, against committed games so they run on a fresh clone."""

    def test_on_pace_player_reads_zero_pressure_in_a_bounded_period(self):
        """The load-bearing definition: budget == initial_budget, pressure == 0."""
        frames = synthetic("on_pace_bounded.pgn")
        pace = features.initial_budget(ingest.parse_time_control(WC_CONTROL))
        self.assertEqual(pace, 180.0)
        for side in ("w", "b"):
            budgets = {f[f"budget_{side}"] for f in frames
                       if f[f"budget_{side}"] is not None}
            pressures = {f[f"time_pressure_{side}"] for f in frames
                         if f[f"time_pressure_{side}"] is not None}
            self.assertEqual(budgets, {180.0}, side)
            self.assertEqual(pressures, {0.0}, side)

    def test_on_pace_player_reads_zero_pressure_in_sudden_death(self):
        """Identical to the bounded case: the horizon must decrement here too.

        With a fixed horizon the divisor never shrinks, so an on-pace player
        drifts from 0 towards 1 across the game and the axis ends up measuring
        move number rather than pressure.
        """
        frames = synthetic("on_pace_sudden_death.pgn")
        pace = features.initial_budget(ingest.parse_time_control("600"))
        self.assertEqual(pace, 15.0)
        for side in ("w", "b"):
            budgets = {f[f"budget_{side}"] for f in frames
                       if f[f"budget_{side}"] is not None}
            pressures = {f[f"time_pressure_{side}"] for f in frames
                         if f[f"time_pressure_{side}"] is not None}
            self.assertEqual(budgets, {15.0}, side)
            self.assertEqual(pressures, {0.0}, side)

    def test_on_pace_player_reads_zero_pressure_in_a_bounded_period_with_increment(self):
        """A bounded period counts down whether or not it carries an increment.

        Pacing at 180+30 on `40/7200+30` drains 180s a move, so the budget must
        sit at the opening 210 throughout.
        """
        moves = 30
        clocks = []
        for n in range(1, moves + 1):
            clocks += [7200 - 180 * n] * 2
        frames = annotated(sans=shuffle_moves(moves * 2), clocks=clocks,
                           time_control="40/7200+30")
        pace = features.initial_budget(ingest.parse_time_control("40/7200+30"))
        self.assertEqual(pace, 210.0)
        budgets = {f["budget_w"] for f in frames if f["budget_w"] is not None}
        pressures = {f["time_pressure_w"] for f in frames
                     if f["time_pressure_w"] is not None}
        self.assertEqual(budgets, {210.0})
        self.assertEqual(pressures, {0.0})

    def test_an_increment_period_does_not_count_down(self):
        """With an increment the situation is memoryless: no endpoint to count to.

        Counting down would make pressure fall on a flat clock, which is the
        move-number dependence the horizon exists to avoid.
        """
        with_increment = ingest.parse_time_control("40/7200:20/3600:900+30")
        horizons = {features.moves_to_threshold(with_increment, m)
                    for m in range(61, 140)}
        self.assertEqual(horizons, {features.NOMINAL_HORIZON})
        # The same period without an increment does count down.
        without = ingest.parse_time_control("40/7200:20/3600:900")
        self.assertGreater(features.moves_to_threshold(without, 61),
                           features.moves_to_threshold(without, 75))

    def test_flat_clock_gives_flat_pressure_under_an_increment(self):
        """Pressure must not move when only the move number does."""
        moves = 24
        clocks = []
        for _ in range(1, moves + 1):
            clocks += [300.0, 300.0]           # perfectly flat: pace == increment
        frames = annotated(sans=shuffle_moves(moves * 2), clocks=clocks,
                           time_control="900+30")
        pressures = {f["time_pressure_w"] for f in frames
                     if f["time_pressure_w"] is not None}
        # 900/40 + 30 = 52.5 opening pace; 300/40 + 30 = 37.5 held.
        self.assertEqual(pressures, {round(1 - 37.5 / 52.5, 4)})

    def test_an_open_ended_period_that_is_not_last_still_reads_memoryless(self):
        """The increment in force is the period's, not the final entry's.

        '900+30:40/7200' is open-ended from move one, so the game never leaves
        the first period. Reading the increment off the last period would see
        zero and start counting down.
        """
        periods = ingest.parse_time_control("900+30:40/7200")
        horizons = {features.moves_to_threshold(periods, m) for m in range(1, 90)}
        self.assertEqual(horizons, {features.NOMINAL_HORIZON})

    def test_empty_periods_do_not_raise(self):
        self.assertEqual(features.moves_to_threshold([], 5),
                         features.NOMINAL_HORIZON)

    def test_horizon_floors_at_a_literal_ten(self):
        """Asserted against a literal, not against the constant.

        The floor is the budget's divisor, so a zero would divide by zero on any
        long sudden-death game; comparing it to itself would not notice.
        """
        self.assertEqual(features.MIN_HORIZON, 10)
        self.assertEqual(features.moves_to_threshold(SUDDEN, 39), 10)
        self.assertEqual(features.moves_to_threshold(SUDDEN, 80), 10)
        self.assertEqual(features.moves_to_threshold(SUDDEN, 500), 10)
        self.assertEqual(features.moves_to_threshold(SUDDEN, 1),
                         features.NOMINAL_HORIZON - 1)

    def test_the_floor_takes_over_at_move_thirty_not_forty(self):
        """Where the countdown meets the floor, stated as a fact not a comment."""
        self.assertEqual(features.moves_to_threshold(SUDDEN, 29), 11)
        self.assertEqual(features.moves_to_threshold(SUDDEN, 30), 10)
        self.assertEqual(features.moves_to_threshold(SUDDEN, 31), 10)
        # A later period counts down only when it has no increment.
        no_inc = ingest.parse_time_control("40/7200:20/3600:900")
        self.assertEqual(features.moves_to_threshold(no_inc, 89), 11)
        self.assertEqual(features.moves_to_threshold(no_inc, 90), 10)

    def test_the_floor_regime_is_survivable(self):
        """A long sudden-death game must annotate past the floor without dividing
        by zero, and keep producing bounded pressure."""
        moves = 45
        clocks = []
        for n in range(1, moves + 1):
            clocks += [max(600 - 12 * n, 1.0)] * 2
        frames = annotated(sans=shuffle_moves(moves * 2), clocks=clocks,
                           time_control="600")
        values = [f["time_pressure_w"] for f in frames
                  if f["time_pressure_w"] is not None]
        # White has a reading from ply 1; only Black lacks one there.
        self.assertEqual(len(values), moves * 2)
        self.assertEqual(
            len([f for f in frames if f["time_pressure_b"] is not None]),
            moves * 2 - 1,
        )
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))
        self.assertGreater(max(values), 0.5)

    def test_a_bounded_final_period_still_bounds_the_horizon(self):
        """'40/7200:20/3600' ends at move 60 without crediting anything there.

        The bound is not a credit point, but it still limits how many moves the
        clock has to cover, so it belongs to the horizon.
        """
        periods = ingest.parse_time_control("40/7200:20/3600")
        self.assertEqual(ingest.period_bounds(periods), [40, 60])
        self.assertEqual(ingest.period_thresholds(periods), [40])
        self.assertEqual(features.moves_to_threshold(periods, 59), 1)
        self.assertEqual(features.moves_to_threshold(periods, 45), 15)

    def test_annotate_uses_bounds_not_credit_points_for_the_horizon(self):
        """End to end: the two lists differ only when the last period is bounded.

        With '3/600:3/300' the game ends at move 6 and nothing is credited there,
        so a horizon built from credit points alone would fall through to the
        open-ended countdown and report tens of moves where one remains.
        """
        white = [580, 560, 840, 820, 800, 780]
        black = [575, 555, 835, 815, 795, 775]
        clocks = [c for pair in zip(white, black) for c in pair]
        frames = annotated(sans=shuffle_moves(12), clocks=clocks,
                           time_control="3/600:3/300")
        by_move = {(f["ply"] + 1) // 2: f for f in frames if f["color"] == "w"}
        self.assertEqual(by_move[4]["moves_to_threshold_w"], 2)
        self.assertEqual(by_move[5]["moves_to_threshold_w"], 1)
        self.assertAlmostEqual(by_move[5]["budget_w"], 800.0, places=3)

    def test_both_controls_reconstruct_rather_than_clamp(self):
        frames = synthetic("two_controls.pgn")
        boundaries = [f for f in frames if f["period_boundary"] and f["color"] == "w"]
        self.assertEqual([(f["ply"] + 1) // 2 for f in boundaries], [3, 5])
        for frame in boundaries:
            self.assertIsNotNone(frame["think_time"])
            self.assertGreater(frame["think_time"], 0.0)

    def test_pressure_stops_forward_filling_across_a_long_gap(self):
        frames = synthetic("clock_gap.pgn")
        by_ply = {f["ply"]: f for f in frames}
        # White's last reading is ply 3; the fill survives four plies, not more.
        self.assertIsNotNone(by_ply[7]["time_pressure_w"])
        self.assertIsNone(by_ply[8]["time_pressure_w"])
        self.assertIsNone(by_ply[9]["time_pressure_w"])
        # A fresh reading revives it, and Black is unaffected throughout.
        self.assertIsNotNone(by_ply[11]["time_pressure_w"])
        self.assertTrue(all(f["time_pressure_b"] is not None
                            for f in frames if f["ply"] > 1))

    def test_premove_chain_is_classified_as_premoves(self):
        frames = synthetic("premove_chain.pgn")
        states = [f["decision_state"] for f in frames]
        self.assertEqual(states.count(features.PREMOVE), 4)
        self.assertIn(features.DECIDED, states)

    def test_boundary_reconstruction_fixture(self):
        frames = synthetic("boundary_reconstruction.pgn")
        boundary = frames[2]
        self.assertTrue(boundary["period_boundary"])
        self.assertAlmostEqual(boundary["think_time"], 30.0, places=3)


class ClampedIsNotAPremove(unittest.TestCase):
    """A measurement artifact must not be classified as player behaviour."""

    def test_clamped_zero_reads_unknown(self):
        # The clock rises by more than the increment: rounding noise in the
        # source, not an instant reply.
        clocks = [600, 600, 590, 700]
        frames = annotated(sans=shuffle_moves(4), clocks=clocks,
                           time_control="600+5")
        clamped = frames[3]
        self.assertTrue(clamped["think_time_clamped"])
        self.assertEqual(clamped["think_time"], 0.0)
        self.assertEqual(clamped["decision_state"], features.UNKNOWN)
        self.assertIsNone(clamped["think_relative"])

    def test_a_genuinely_measured_zero_is_a_premove(self):
        clocks = [600, 600, 600, 595]
        frames = annotated(sans=shuffle_moves(4), clocks=clocks, time_control="600")
        self.assertFalse(frames[2]["think_time_clamped"])
        self.assertEqual(frames[2]["think_time"], 0.0)
        self.assertEqual(frames[2]["decision_state"], features.PREMOVE)

    def test_clamped_plies_stay_out_of_the_medians(self):
        clocks = [600, 600, 590, 700, 580, 690, 570, 680]
        frames = annotated(sans=shuffle_moves(8), clocks=clocks,
                           time_control="600+5")
        decisions = [f["think_time"] for f in frames
                     if f["decision_state"] == features.DECIDED]
        self.assertNotIn(0.0, decisions)


@unittest.skipUnless(
    not MISSING_REAL,
    "real-game PGN fixtures absent from data/pgn (gitignored): "
    + ", ".join(MISSING_REAL),
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
