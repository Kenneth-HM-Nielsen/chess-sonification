"""Command line entry point.

    python -m src.cli render data/pgn/game.pgn --out out/game.wav
    python -m src.cli frames data/pgn/game.pgn
    python -m src.cli plot data/frames/game.json

Heavy imports happen inside the command handlers so that `--help` stays fast and
works before the audio dependencies are installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "data" / "frames"
OUT_DIR = ROOT / "out"


def cmd_render(args: argparse.Namespace) -> int:
    from . import features, ingest, render, tension

    frames = features.annotate(ingest.ingest(args.pgn))
    track = tension.tension_track(frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render.render(frames, track, args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_frames(args: argparse.Namespace) -> int:
    from . import features, ingest

    frames = features.annotate(ingest.ingest(args.pgn))
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    path = ingest.write_frames(frames, args.pgn.stem, FRAMES_DIR)
    print(f"wrote {path} ({len(frames)} plies)")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    import json

    from . import tension

    frames = json.loads(args.frames.read_text())
    track = tension.tension_track(frames)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.frames.stem}_tension.png"
    tension.plot_track(track, path)
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Render a chess game as music driven by the structure of the game.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser("render", help="render a PGN to a WAV")
    p_render.add_argument("pgn", type=Path, help="path to a PGN file")
    p_render.add_argument(
        "--out", type=Path, required=True, help="path to write the WAV to"
    )
    p_render.set_defaults(func=cmd_render)

    p_frames = sub.add_parser(
        "frames", help="extract per-ply features to data/frames/ without rendering"
    )
    p_frames.add_argument("pgn", type=Path, help="path to a PGN file")
    p_frames.set_defaults(func=cmd_frames)

    p_plot = sub.add_parser("plot", help="plot the tension track for extracted frames")
    p_plot.add_argument("frames", type=Path, help="path to a frames JSON file")
    p_plot.set_defaults(func=cmd_plot)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
