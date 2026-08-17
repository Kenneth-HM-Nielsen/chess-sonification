# CCI 4.1 — Close-out and Phase 4 (orchestral renderer)

Self-contained. References nothing outside this document except `CLAUDE.md` and `GOTCHAS.md`, both already in the repo.

**First action:** save the full text of this instruction, exactly as pasted, to `docs/cci/CCI-4.1-phase4.md`. Create the directory if absent. This is the phase spec, so a future session with no history can read it. Replace any index file you previously created there with nothing — the directory holds specs, not indexes.

---

## Standing block

Applies to every part of this document.

**Repo:** public, personal GitHub `Kenneth-HM-Nielsen`. Not AIQubed-Consulting. No client names, no internal tooling references, no company boilerplate in code, comments or commit messages.

**Stack:** Python 3.11+, plain. Dependencies limited to the chess library, numpy, scipy, soundfile, and matplotlib for diagnostics. No engine — Stockfish is unavailable and must never become a dependency. **No sample libraries or SoundFonts** — audio is synthesised from numpy arrays, which is what keeps a future browser port viable.

**Architecture is one-way:** `ingest → features → tension → render`. No back edges. `tension.py` must not import anything audio-related, must not know music is the consumer, and must not reference `budget`. `render.py` owns every musical mapping decision — scale, register, dissonance, timbre, envelopes, timing.

**DRY.** One definition per concept. Find what exists before adding.

**YAGNI.** No fields nothing reads, no abstraction with one implementation, no hooks for imagined future phases.

**SOLID.** Single responsibility per module. Adding a musical layer must not require editing analysis code.

**Incremental.** One sub-phase at a time, gated. No refactoring while implementing something else.

**Discovery-first.** Read files before editing. Never assert repo state.

**QA subagent at the end of every sub-phase.** Spawn via the Task tool. It receives this spec and the diff. It does **not** receive your summary, rationale, or account of what you did — a reviewer given the narrative reviews the narrative and agrees with it. QA reports; you fix. Report QA findings including ones you disagree with, with your reasoning.

**Read `GOTCHAS.md` before writing tests**, not after QA finds the same defect again. Two of its entries have already been repeated by the session that added them.

**Listening gates belong to Kenneth.** Neither you nor Claude in chat has audio access. Produce artifacts and measurements; never claim a listening gate passed on measurement alone. State what was measured and what it implies. Demo WAVs must be short and single-purpose.

---

# Part A — Close-out

No item deferred. Complete all of Part A before starting Part B.

## A1 — Converge git

`main` is 19 commits behind; all work since the scaffold sits on `feat/sonification-core`.

Merge into `main`, push, delete the branch local and remote. Cut new branches off up-to-date `main` from here.

## A2 — `CLAUDE.md` corrections

- **Remove every reference to a document that is not in the repo**, including the web app spec and its "needs two corrections" note. The web app status line becomes: not started, spec to be written when the phase begins. No absent or known-defective document may be referenced anywhere.
- Point the current-phase line at `docs/cci/CCI-4.1-phase4.md`.
- **Phase 5 (CLI):** mark complete. `render`, `frames` and `plot` exist and work.
- **Frame JSON:** only synthetic PGN fixtures are committed; no frame JSON is tracked. Correct the exceptions sentence.
- **Dependency naming:** the import is `chess`; the PyPI package `python-chess` is a deprecation shim. State once what `requirements.txt` actually installs from, and verify it resolves on a clean install.

Keep the commit stamp mechanism. It is right.

## A3 — Rename `decision` to `decision_state`

The constant is `features.DECISION = "decision"`. The field was boolean, became three-valued (`decided` / `premove` / `unknown`), and kept its boolean-era name.

Rename constant and frame key to `decision_state`. Not cosmetic: a future reader seeing `decision` will assume a boolean, which is the exact conflation the three-valued field was introduced to remove. No alias, no compatibility shim. Update `CLAUDE.md` to match.

## A4 — Dispose of the 4a voices

4a is committed and superseded. Do not leave dead code and do not keep anything "in case".

**Delete:** the six additive sine spectra. Part B replaces every voice with an instrument-family target; no harmonic stack survives.

**Keep and retarget:**
- Pitch-relative lowpass cutoff — a fixed frequency removes nothing from a low note
- Fade **after** filtering, never before — filter ringing past the envelope clicks on every note
- Stereo pan at roughly ±0.4 as the colour cue, deliberately redundant with the lowpass
- Separation-floor assertions and the voice-demo harness

**The bishop colour question is closed.** It existed because a lowpass cannot darken a near-pure sine. Part B gives the bishop a bowed-string target with harmonics to filter, and pan covers colour regardless. No residual task.

Report what was deleted and what was retargeted so the disposal is visible in the diff.

---

# Part B — Phase 4, orchestral renderer

## Why the redesign

The 4a palette was additive sine stacks and sounded like it. The target is orchestral.

Honest scope, by family — do not spend iterations fighting this:

- **Percussion is genuinely convincing** from numpy. Timpani as a few inharmonic membrane modes plus a noise transient and exponential decay. Cymbal as bandpassed noise, fast attack, long ragged decay.
- **Plucked strings are good** via Karplus-Strong.
- **Sustained strings and brass stay recognisably synthetic.** A passable pad and a passable swell are the ceiling. Nobody will mistake them for players.
- **MIDI export (B6) is the only route to real orchestral sound.** The WAV path stays primary because it is what a browser can do.

"Orchestral, not synth" means aiming at instrument families rather than abstract spectra. It does not mean photorealism.

Output is **stereo**, 44.1 kHz, float32, via soundfile with shape `(n, 2)`.

## B1 — Percussion and capture events

Lead here: most audible payoff, least synthesis risk.

**Cymbal on captures.** `is_capture` exists and material removed is already computed. Scale by captured value — a pawn is a small shimmer, a queen a full crash. Bandpassed noise, fast attack, decay proportional to value.

**Timpani under accumulation.** Drive from `stall` and rising `tension`. A soft roll enters as the stall term climbs and swells toward the fifty-move threshold — in Game 6 that means a roll building across moves 80–109, breaking on the reset. Pitch it to the current tonal centre so it belongs harmonically.

**Gate B1 artifacts:** a demo of three captures at different values, and a 30-second excerpt of Game 6 moves 95–115.

**Gate B1 passes when** Kenneth hears a queen capture as unmistakably bigger than a pawn capture, and hears the timpani build.

## B2 — Piece voices

Six voices, each synthesised toward an instrument family:

- **Pawn** — pizzicato string, Karplus-Strong, short
- **Knight** — muted brass stab, detuned, slightly nasal
- **Bishop** — bowed string, soft attack, sustained, light vibrato
- **Rook** — low brass, blunt, odd-harmonic-heavy
- **Queen** — full brass, harmonic profile rising through the attack so it brightens as it swells
- **King** — low woodwind, hollow, quiet

Colour: stereo pan ±0.4 plus the pitch-relative lowpass. Pan encodes colour only — never piece, never file, never anything else. Verify that summing to mono cancels nothing and that the sum does not clip.

Pitch: file → scale degree, rank → octave.

**Gate B2 artifacts:** the retargeted voice demo, each piece white then black.

**Gate B2 passes when** the six voices are distinguishable and none sounds like a sine.

## B3 — Archetype theme

Classify each game from the axes already measured — tension mean and peak, density mean, `dotted%`, stall profile — into one of four archetypes: **manoeuvring**, **attacking**, **grinding**, **scrambling**.

The archetype selects key and mode, instrumentation emphasis (strings for manoeuvring, brass for attacking, low winds and timpani for grinding, thinned high strings for scrambling), and a short thematic cell of four to six notes stated at the opening.

**Classify from the measurements, never from the ECO code.** A label driving the music is weaker than the music following the game. The measured profile *is* the game.

**Gate B3 passes when** Kenneth can hear that two contrasting games are different kinds of piece, not merely differently pitched.

## B4 — Tension, crescendo, cadence

- `tension` → scale selection, dissonance, added tones. This mapping is render's alone.
- Sustained high tension → low pedal, removed on cadence.
- **Crescendo into release.** The whole parameter track is known before rendering, so ramp dynamics across the plies *preceding* a large release. Non-causal rendering is fine — the game is parsed up front offline and in a browser alike.
- `cadence_strength` is graded. A release from 0.9 lands hard; one from 0.1 is barely perceptible. **Never threshold it into a boolean.**
- Drawn games get no terminal cadence and must end unresolved.

**Gate B4 passes when** Kenneth hears the approach to Game 6's move-109 peak as a build and the release as a release.

## B5 — Pressure and timing

- `pressure` → tempo acceleration, harmonic thinning, voices dropping out.
- Pulse element above a threshold, following pressure directly, **with no period-boundary logic anywhere**. The move-40 cut-out emerges because pressure genuinely falls to zero; move 60 correctly produces none.
- `pressure is None` → whole layer silent. Absent layers switch off; never substitute a neutral default.
- Onsets from `density`. `meter` `"dotted"` gives the gallop on forcing streaks.
- Think time: `think_relative` → convolution reverb tail length via `scipy.signal.fftconvolve`, IR as decaying noise clamped roughly 0.15–6 s. `game_tempo_scale` sets global spaciousness. `decision_state` gates it — `premove` gets the minimum dry IR, `unknown` the neutral default, and the three must be audibly different. Cache IRs by quantised duration in roughly 20 buckets, shared across channels since the IR is a property of think time, not colour.
- Every game lands in 2–4 minutes, bullet to Game 6.

**Gate B5 passes when** the move-40 pulse cut-out is audible without being told where to listen.

## B6 — MIDI export

A second output, not a replacement.

Multi-track MIDI from the same parameter track: one track per piece voice, one for percussion, one for the pedal. Think time maps to note length plus reverb-send CC91 instead of convolution; dynamics to velocity plus CC11.

This exists so the analysis can be rendered through real orchestral libraries. MIDI must never become the primary renderer.

If this requires a dependency beyond the current set, stop and ask before adding it.

---

## Validation

Render the corpus, hash the filenames, hand them to Kenneth to classify as manoeuvring, attacking, grinding or scrambling.

Report failures plainly. A misclassified render is the most useful output of this phase, and a report claiming a perfect score is less credible than one that does not.

## Done when

- Part A complete: main converged, branch deleted, `CLAUDE.md` referencing only files that exist, rename done, 4a disposal visible in the diff
- This document saved at `docs/cci/CCI-4.1-phase4.md`
- Each of B1–B6 gated, with artifacts and measurements handed to Kenneth and his ruling recorded
- Blind classification reported honestly, errors included
- QA reported per sub-phase, including disagreement

## What not to do

- Do not add sample libraries or SoundFonts
- Do not build multiple sub-phases before a listening gate
- Do not let MIDI become the primary path
- Do not classify archetypes from ECO codes
- Do not threshold `cadence_strength`
- Do not special-case period boundaries
- Do not substitute neutral defaults for absent layers
- Do not chase realism in the sustained voices
- Do not begin the web app
