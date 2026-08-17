# CLAUDE.md — chess-sonification

Read this first, every session. Then read the CCI for the current phase in `docs/cci/`.

**Reconcile before trusting.** Verify constant names, module paths and fixture
counts against the actual code, and correct this file where it has drifted. Do
not assume a file exists because it is named here.

*Last reconciled against the repo at `da2215a` (sub-phase 4a).*

---

## What this is

Rendering chess games as music, driven by the structure and drama of the game
rather than by a notation-to-pitch mapping.

Target: a locked positional game should sound like sustained thriller tension
that does not resolve; a forcing attacking game should sound fast, dense and
metrical. Aesthetic direction is orchestral, not synth.

Public personal repo under `Kenneth-HM-Nielsen`. **Not** AIQubed-Consulting. No
client names, no internal tooling references, no company boilerplate in code,
comments or commit messages.

---

## Stack constraints — non-negotiable

- Python 3.11+, plain. No frameworks, no config systems.
- Dependencies: `chess`, `numpy`, `scipy`, `soundfile`, plus `matplotlib` for
  diagnostic plots. Nothing else without asking. (The library is imported as
  `chess`; `python-chess` on PyPI is a deprecation shim.)
- **No chess engine.** Stockfish is unavailable and must never become a
  dependency. Evaluations come only from `[%eval]` annotations already in the
  PGN. Everything else is derived from board state.
- **No sample libraries or SoundFonts.** Audio is synthesised from numpy arrays.
  This constraint is what keeps the browser port viable, and it is why
  "orchestral" means synthesised orchestral timbre, not sampled instruments.
- Never commit PGN files or rendered audio. The only committed game data is
  `tests/fixtures/*.pgn` — short synthetic games written for the suite.
  `data/pgn/`, `data/frames/` and `out/` are all gitignored; no frame JSON is
  tracked.

---

## Architecture

```
ingest → features → tension → render
```

**One-way dependency. No back edges.** `render` may read the parameter track and
the note-level frame fields; nothing else may read audio concerns. `tension.py`
must not import anything audio-related and must not know that music is the
consumer. Enforced by `tests/test_layering.py`, which parses the modules rather
than grepping them.

- `ingest` — PGN → per-ply frames. Clocks, evals, think time, result.
- `features` — board and clock measurements. Pure functions, no mutation of the
  caller's board.
- `tension` — interprets features into a parameter track. **State only, no
  musical parameters.**
- `render` — owns every musical mapping: scale, register, dissonance, timbre,
  envelopes, timing.

`tension.py` must contain no reference to `budget`. It reads `pressure_w` /
`pressure_b`. Verified behaviourally by deleting the `budget_*` keys and
requiring an identical track, over every fixture.

`tension_track()` returns a two-scope record: `game_tempo_scale` and
`active_layers` per game, alongside a `plies` list.

---

## The three axes

Independence is the central design decision. Do not collapse them.

| Axis | Property of | Meaning |
|---|---|---|
| `tension` | the position | unresolved accumulation |
| `pressure` | the players | how far behind their own starting pace |
| `density` | activity | how much is happening |

A dead-drawn endgame in a scramble is low tension, high pressure. A gambit is
low tension, high density. Both must be representable.

---

## Locked decisions — do not re-litigate

These were each settled with measurement. Reversing one needs new evidence, not
a fresh opinion.

- **Pawn structure carries tension; mobility carries density.** Measured:
  `locked_pawns` separates closed from open categorically, mobility by 8.5%.
  Mobility must never return to tension.
- **A sharp attack reads LOW tension.** Tension measures unresolved
  accumulation; a gambit resolves constantly. The Evans ranking last is correct.
  Sharpness lives in density, meter and percussion.
- **Pressure is relative to the player's own starting pace**, with increment
  included in both terms. An on-pace player reads exactly 0.0.
- **The pressure ceiling is accepted.** With an increment you can never reach
  zero time, so pressure caps at `1 − increment/initial_budget`. Do not rescale
  to recover range — that destroys cross-format comparability.
- **Think time splits.** `think_relative` (per-ply, scale-free) drives reverb;
  `game_tempo_scale` (per-game, absolute) drives global character. Never
  normalise `game_tempo_scale` away — it keeps bullet distinguishable from
  classical.
- **`cadence_strength` is graded, never boolean.** A release from a stall of 55
  and one from 3 are the same event type; only accumulated height distinguishes
  them.
- **A draw is not a release.** Decisive results fire a terminal cadence; draws
  and `*` do not.
- **No period-boundary special cases anywhere.** The pulse follows pressure
  directly. The move-40 cut-out emerges because pressure genuinely falls to
  zero; move 60 correctly produces no cut-out.
- **Calibration lives in the analysis layer, globally.** One constant, identical
  for every game, chosen once, pinned against a literal. Never per-game.
- **Absence is information.** `None` inputs switch their layer off. Never
  substitute a neutral default.
- **A premove is not a fast decision, and a clamped value is neither.**
  `decision_state` is three-valued. The constants are `features.DECISION`
  (`"decision"`), `features.PREMOVE`, `features.UNKNOWN`.

---

## Invariants — assert these, they are load-bearing

- On-pace player in a bounded period: budget exactly `initial_budget`, pressure
  exactly 0.0
- Same for sudden death, and for a bounded period with increment
- Identical clock at two different move numbers must give identical pressure
  (no move-number dependence)
- Corpus tension ordering: Benoni > KID > Game 6 > scrambles > Evans
- Setting any calibration constant to its identity value must FAIL the suite

Horizon rule — the countdown applies only where the clock must cover a finite
number of moves:

```
bounded period   → period_end_move - current_move
increment > 0    → NOMINAL_HORIZON            (memoryless, no countdown)
otherwise        → max(NOMINAL_HORIZON - moves_played, MIN_HORIZON)
```

The nominal is a **prior on game length, not a rule.** Say so in comments
wherever it governs.

### Calibration constants, and where they live

| Constant | Value | Module |
|---|---|---|
| `NOMINAL_HORIZON` | 40 | `features` |
| `MIN_HORIZON` | 10 | `features` |
| `PREMOVE_FLOOR_S` | 0.15 | `features` |
| `FORWARD_FILL_PLIES` | 4 | `features` |
| `ROLLING_WINDOW_MOVES` | 12 | `features` |
| `GAMMA` | 0.5 | `tension` |
| `HANGING_FULL` | 12 | `tension` |
| `STALL_SATURATION` | 50 | `tension` |
| `DENSITY_MOBILITY_SHARE` | 0.55 | `tension` |

---

## Engineering discipline

**DRY.** One definition per concept. The Python/JS duplication in the planned
web app is the standing hazard; golden-file tests are the contract.

**YAGNI.** No fields nothing reads. No abstraction with one implementation. If a
feature measures zero across all fixtures, remove it.

**SOLID.** Single responsibility per module; one-way dependencies; adding a
musical layer must not require editing analysis code.

**Incremental.** One phase at a time, gated. No refactoring while implementing
something else.

**QA subagent at the end of every phase or sub-phase.** Spawn via the Task tool.
It receives the phase spec and the diff. It does **not** receive your summary,
rationale, or account of what you did — a reviewer given the narrative reviews
the narrative and agrees with it. QA reports; the builder fixes. Report QA
findings including ones you disagree with, with reasoning.

**Read `GOTCHAS.md` before writing tests, not after QA finds the same defect
again.** Both entries have already been repeated by the session that added them.

Tests run with `python -m tests.run` (`--strict` makes a skip a failure).

---

## Listening gates

Neither Claude Code nor Claude in chat has audio access. **All listening gates
belong to Kenneth.**

Claude Code produces artifacts and measurements, and never claims a listening
gate passed on measurement alone — it reports what was measured and what that
implies. Demo WAVs should be short and single-purpose.

---

## Current state

Update this section at the end of every phase.

- **Phase 0 (scaffold)** — complete. Repo, licence, CLI skeleton.
- **Phase 1 (ingest)** — complete. Multi-period time controls, boundary
  reconstruction, think time, result.
- **Phase 2 (features)** — complete and QA-clean. Board features, pressure,
  progress/stall, hanging material.
- **Phase 3 (tension)** — complete and QA-clean. Five tension sources including
  `hanging_material`, graded cadence, global GAMMA calibration.
- **Phase 4 (render)** — 4a of CCI 4.0 built (six synthesised voices, colour by
  pitch-relative lowpass, fixed major scale) and committed at `da2215a`, but
  **superseded by the orchestral redesign**, whose spec is not yet in the repo.
  4a's listening gate was never returned, and the bishop colour question is
  still open: a lowpass cannot darken a near-pure sine, so five voices carry
  colour at 1.11–1.83× and the bishop only 1.08×.
- **Phase 5 (CLI)** — the three commands (`render`, `frames`, `plot`) exist and
  work; they were built with the scaffold. Nothing further is specified.
- **Web app** — not started. Its CCI needs two corrections before use: it
  assumes mono output, and it frames Lichess as strictly the better source when
  chess.com records clocks at tenths and is better for bullet and blitz.

170 tests, green. `tests/fixtures/` holds 7 committed synthetic games;
`data/pgn/` holds 9 real games locally and is gitignored.

---

## GitHub hygiene

Branch each piece of work off up-to-date `main`. Converge to `main` and delete
the branch, local and remote, once merged.

*Currently diverged: all 19 commits of phases 1–4a sit on
`feat/sonification-core`, and `main` is still at the scaffold. This wants
resolving before more branches are cut.*
