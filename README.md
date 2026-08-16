# chess-sonification

Turns a chess game into music that follows the *shape of the game* — where the
tension builds, where it breaks — rather than mapping notation to notes through a
lookup table.

Feed it a PGN, get a WAV. A locked positional game should come out sounding like
sustained, unresolved thriller tension. A sharp attacking game should come out
fast, dense and metrical. You should be able to tell two such games apart by ear
without being told which is which.

## How it works

The renderer reads two signal layers off the game.

**Note level** decides what each individual move sounds like: the piece moved
picks the timbre (each of the six piece types gets its own additive-synthesis
harmonic spectrum and envelope — pawns are short plucks, queens are a full
harmonic stack, kings are hollow odd harmonics), the destination square picks the
pitch (file to scale degree, rank to octave), and the player's think time picks
the reverb tail, so a long think blooms and a snap recapture lands dry.

**Structural level** is the part that makes the result non-arbitrary. Per ply, the
board is measured for mobility, blocked and mutually-attacking pawns, pressure
around each king, forcing sequences and material balance. Those measurements feed a
stateful tension value in `[0, 1]` that is *carried across the game* rather than
recomputed move by move. Tension drives harmony, note density and dynamics — and it
only resolves on a real release event, such as a pawn break, a queen trade, a king
reaching shelter, or the end of the game. It never resolves on a timer, which is
what keeps a grinding position sounding grinding.

Where the PGN carries `[%eval]` annotations, evaluation volatility is folded in as
an additional tension term. Where it does not, that term drops out and the
board-feature weights renormalise — the output stays coherent either way. **No
chess engine is used or required**; every other signal is derived from board state
alone.

## Install

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Put a PGN in `data/pgn/` and render it:

```bash
python -m src.cli render data/pgn/game.pgn --out out/game.wav
```

Two inspection commands are available for looking at the analysis without
rendering audio:

```bash
python -m src.cli frames data/pgn/game.pgn    # extract per-ply features to data/frames/
python -m src.cli plot data/frames/game.json  # plot the tension track to out/
```

Output is 44.1 kHz mono. A 40-move game renders to roughly two to four minutes of
audio regardless of the game's actual time control, because note onsets come from
the density track rather than from elapsed clock time.

## Input

Any standard PGN works. Lichess and broadcast exports that carry `[%clk]` and
`[%eval]` annotations give the richest result, since think time drives the reverb
layer and evaluation swings feed tension. Bare historical PGNs with neither
annotation still render — they simply use the board-derived signal only.

`data/pgn/`, `data/frames/` and `out/` are gitignored. This repository is code
only; bring your own games.

## License

MIT — see [LICENSE](LICENSE).
