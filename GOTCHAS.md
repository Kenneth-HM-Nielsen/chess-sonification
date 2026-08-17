# Gotchas

Things that cost time here, written down so they cost it once. Each entry is the
general rule first, then the specific case that produced it.

## Never assert an implementation constant against itself

A test that compares a value to the same constant the code uses asserts nothing.
It passes for every possible value of that constant, including the ones that
break the program. Pin the constant to a literal instead, and pin the behaviour
that depends on it separately.

The horizon floor is the divisor of a per-move budget. The test read:

```python
self.assertGreaterEqual(features.moves_to_threshold([], move),
                        features.MIN_HORIZON)     # asserts nothing
```

Setting `MIN_HORIZON = 0` passed the entire suite and then raised
`ZeroDivisionError` on three real games. The fix is to assert the value and its
consequences:

```python
self.assertEqual(features.MIN_HORIZON, 10)
self.assertEqual(features.moves_to_threshold(SUDDEN, 80), 10)
```

The same shape appears whenever a test derives its fixture from a constant it is
meant to be pinning — the fixture moves with the constant and the assertion
follows it.

## A clock reading describes the position after its move

`[%clk]` is recorded once a move has been played, so anything paired with it must
describe the state that move left behind: the moves still ahead, and the
increment that will be paid for them. Mixing a post-move clock with a pre-move
horizon double-counts, and produced a budget of one whole time control for a
single move at exactly the tensest ply of a classical game.

## Behind a null move, the side giving check can capture a king

`board.push(chess.Move.null())` is the documented way to count the waiting side's
mobility, and it works even when the mover is in check. But the resulting
position is illegal, and python-chess will report a king capture as a legal move.
Left in, mobility is inflated by one on every checking ply — a systematic error
landing precisely on the sharpest moves in the game.

## Lichess truncates clocks to whole seconds; chess.com keeps hundredths

A Lichess PGN export writes `[%clk 0:00:59]`, so in bullet every think time
quantises to 0, 1, 2 or 3 seconds and any measure built on their ratios collapses.
chess.com writes `[%clk 0:00:58.8]`. Choose the source to match the resolution the
measurement needs, and check the resolution before concluding a feature is weak.

## Formatting a clock: round before you decompose

Formatting `599.95` by taking `divmod` first and rounding the seconds field last
yields `0:09:60.0`, which is not a valid clock and which python-chess reads back
as `600`. Round into integer units first, then decompose.

This lived in a test helper. It degraded every sub-second value to a tenth, and
where the rounding crossed a minute boundary it lost the difference entirely —
which is exactly what happened to the fixture then in use, so the premove floor
was never actually exercised by the test written for it.

## Mobility barely notices whether a position is closed

Raw legal-move count separates a locked King's Indian from an open gambit by
about eight percent, because a closed centre pushes play to the wings without
reducing the number of available moves. Pawn-structure measures separate the same
two games categorically. Mobility is a good measure of how much is happening and
a poor measure of how tense things are.

## A skipped test is not a passing test

A suite whose fixtures are gitignored reports green on a fresh clone while
verifying nothing. Either commit fixtures the tests can rely on, or make the
runner name every skip and fail under `--strict`. Both, here.

## A behavioural test must be shown to enter the branches it guards

A test that never reaches the code it exists to protect is a green light wired to
nothing. Run it over the whole fixture corpus, not the smallest case that makes
it pass, and check which branches it actually enters before trusting it.

The binding test for "the tension layer must not read `budget`" deleted those
fields and required an identical track. Sound in design — a source scan cannot
catch a computed key, this can — but it ran on a single sixty-ply knight shuffle
with no captures, no pawn moves and no evaluations. It therefore entered none of
the four release branches and none of the evaluation term, and this passed the
whole suite:

```python
if "eval" in terms:
    key = "bud" + "get_" + frame["color"]      # invisible to an AST scan
    terms["eval"] = _clamp(terms["eval"] + frame[key] / 1000.0)
```

Running it over every fixture catches it — but only if the fixtures reaching
those branches are committed. The first attempt at this fix ran over the whole
corpus and still caught nothing on a fresh clone, because every committed
fixture was a knight shuffle: no captures, no evaluations, and after a later
change no result either, so not one release branch was entered by anything in
the repository. Assert the coverage itself, as `test_the_committed_corpus_
reaches_every_release_branch` does, rather than trusting a file count.

The related habit: when a mutation survives, the finding is usually not "add a
test" but "the test you have never runs that line".
