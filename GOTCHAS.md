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

Assert that the committed corpus enters every branch the test claims to guard. A
fixture count is a proxy, and proxies drift from the property: the first attempt
at this fix ran over the whole corpus and still caught nothing on a fresh clone,
because every committed fixture was a knight shuffle with no captures and no
evaluations, and after a later change no result either — so not one release
branch was entered by anything in the repository.
`test_the_committed_corpus_reaches_every_release_branch` asserts the property
directly.

The related habit: when a mutation survives, the finding is usually not "add a
test" but "the test you have never runs that line".

## Restore a mutation from a copy, never from git

A mutation run is edit, test, revert, and the obvious revert is
`git checkout -- src/thing.py`. It also destroys every uncommitted change in that
file, which on a mutation run is the work being tested. Copy the file first and
copy it back.

This happened three times in one session, and the second failure was worse than
the first: the run had already reverted the file, so every later mutation applied
to the *committed* version, matched nothing, and was reported killed when it had
never been applied. A mutation run that silently stops mutating reads exactly
like a mutation run where everything is pinned. Have the harness compare the file
against the pristine copy and print NO-MATCH when a mutation did not apply.

Commit before mutating, and the whole class goes away.

## A test over synthesised noise can measure the draw and not the mapping

Anything built from noise -- a cymbal, a Karplus-Strong pluck, a reverb impulse
-- gives statistics that move with the random draw. Two failures, opposite in
shape, both from B1:

**The statistic does not move at all.** Raggedness was measured as the fraction
of rising steps in an envelope follower over the decay. Bandpassed noise already
fluctuates, so that fraction is 0.49495 whether the wander is at full depth or
deleted outright -- identical to five figures, against a floor of 0.05. The
mapping could be removed and the test named after it stayed green. Detrend against
the shape the signal is *supposed* to have and measure what is left: the spread of
the log-envelope after taking the exponential out is 0.07-0.11 with the wander and
0.027-0.032 without it.

**The statistic moves far too much.** The attack was asserted as `argmax` of the
envelope being under 25 ms. Over 200 draws that lands anywhere from 5 ms to
320 ms, and breaks the bound on 31 of them; it was green only because the seed
and the draw order happened to put it at 12 ms. Worse, it was the only assertion
killing two unrelated mutations, so fixing it properly would have released them.

Before trusting an assertion about noise, run it over twenty seeds and look at the
range, and run it against the mutation it exists to catch. If the two ranges
overlap, the assertion is decoration. Prefer aggregates -- energy, rms, a
smoothed envelope peak, a band's share of the spectrum -- over anything derived
from a single sample or a single extremum.
