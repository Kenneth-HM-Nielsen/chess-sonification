"""Short, single-purpose renders for the listening gates.

Neither Claude Code nor Claude in chat has audio access, so every listening gate
belongs to Kenneth. This produces the artifacts he rules on and nothing else, and
it is committed so that a gate can be re-run months later against the same
command rather than against a remembered one-off.

    python -m tools.demo captures
    python -m tools.demo excerpt data/pgn/wc2021_game6.pgn --from-move 95 --to-move 115

Output goes to `out/`, which is gitignored: the WAVs are artifacts of a gate, not
part of the repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src import render

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"

# Long enough that each capture rings out before the next one is struck. The queen
# is the longest at about 4.3 seconds, so the last one is not cut off by the file.
CAPTURE_GAP_S = 5.0

# Pawn, rook, queen. Three captures at different values, which is what the gate
# asks: is a queen unmistakably bigger than a pawn, with a middle rung to show the
# scale is graded rather than a switch.
CAPTURE_DEMO = ((1, "pawn"), (5, "rook"), (9, "queen"))


def _write(audio: np.ndarray, path: Path) -> Path:
    import soundfile

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), audio, render.SAMPLE_RATE, subtype="FLOAT")
    print(f"wrote {path} ({len(audio) / render.SAMPLE_RATE:.1f}s)")
    return path


def cmd_captures(args: argparse.Namespace) -> int:
    """Three captures at different values, struck in order, White capturing."""
    rng = np.random.default_rng(render.RNG_SEED)
    blocks = []
    for index, (value, name) in enumerate(CAPTURE_DEMO):
        at = int(index * CAPTURE_GAP_S * render.SAMPLE_RATE)
        block = render.cymbal(value, "w", rng)
        blocks.append((at, block))
        print(f"  {at / render.SAMPLE_RATE:5.1f}s  {name:5s} "
              f"value {value}  {len(block) / render.SAMPLE_RATE:.2f}s long")
    total = int(len(CAPTURE_DEMO) * CAPTURE_GAP_S * render.SAMPLE_RATE)
    _write(render._mix(blocks, total), OUT_DIR / "demo_captures.wav")
    return 0


def cmd_excerpt(args: argparse.Namespace) -> int:
    """A window of moves from a real game, rendered in full context.

    The analysis runs over the whole game and only then is the window taken, so
    the tension and stall the excerpt plays are the ones the game actually
    reached at that point. Slicing the PGN first would restart the accumulation
    from zero and the build would not be there.
    """
    from src import features, ingest, tension
    from src.ingest import move_number_for_ply

    frames = features.annotate(ingest.ingest(args.pgn))
    track = tension.tension_track(frames)

    # Ply-to-move arithmetic is `ingest`'s, not restated here.
    keep = [index for index, frame in enumerate(frames)
            if args.from_move <= move_number_for_ply(frame["ply"]) <= args.to_move]
    if not keep:
        print(f"{args.pgn.name} has no plies in moves "
              f"{args.from_move}-{args.to_move}", file=sys.stderr)
        return 1

    window = [frames[index] for index in keep]
    entries = [track["plies"][index] for index in keep]
    print(f"  moves {args.from_move}-{args.to_move}: {len(window)} plies, "
          f"stall {min(e['stall'] for e in entries):.3f} to "
          f"{max(e['stall'] for e in entries):.3f}, "
          f"{sum(1 for f in window if f['captured_value'])} captures")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"demo_{args.pgn.stem}_{args.from_move}_{args.to_move}.wav"
    render.render(window, {**track, "plies": entries}, out)
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.demo",
        description="Short renders for the listening gates.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_captures = sub.add_parser(
        "captures", help="three captures at different values"
    )
    p_captures.set_defaults(func=cmd_captures)

    p_excerpt = sub.add_parser("excerpt", help="a window of moves from a PGN")
    p_excerpt.add_argument("pgn", type=Path)
    p_excerpt.add_argument("--from-move", type=int, required=True, dest="from_move")
    p_excerpt.add_argument("--to-move", type=int, required=True, dest="to_move")
    p_excerpt.set_defaults(func=cmd_excerpt)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
