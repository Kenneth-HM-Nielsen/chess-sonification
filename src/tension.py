"""Stateful reading of how a game is going, as a per-ply state track.

Three axes, kept independent because they answer different questions. Tension is
a property of the position: how locked the pawns are, how much heat is on the
kings, how violently the evaluation is swinging, how long nothing irreversible
has happened. Pressure is a property of the players, and passes straight through
from the clock. Density is how much is going on. A dead-drawn endgame in a time
scramble is low tension and high pressure; a sharp attack with both players at an
hour is the reverse, and both have to be representable.

Tension is carried across plies rather than recomputed at each one, and it comes
down only when something on the board brings it down. There is no timer and no
schedule: sustained tension that refuses to resolve is the whole point.

This module emits *state*, not sound. It has no opinion on pitch, scale, timbre
or register, imports nothing audio-related, and does not know that music is the
consumer -- adding a musical layer must not require editing anything here.
Mapping state onto sound belongs to `render`.

Mobility is deliberately absent from tension: measured over a locked King's
Indian against an open gambit it separates them by only 8.5%, because a closed
centre pushes play to the wings without reducing the legal move count. It feeds
density instead.
"""

from __future__ import annotations

from collections import deque

# Normalisers, taken from the observed range across the fixture set rather than
# invented: a value at or beyond these reads as the full measure of that term.
LOCKED_FULL = 6
PAWN_TENSION_FULL = 3
KING_PRESSURE_FULL = 8
HANGING_FULL = 12
MOBILITY_FULL = 80
EVAL_VOLATILITY_FULL = 150.0
EVAL_WINDOW = 4

# The fifty-move rule is claimable at 100 halfmoves, but a grind is audible long
# before it: by 50 the players are already handling the position as one where
# nothing is happening. Saturating here rather than at the rule keeps the term
# expressive over the range games actually reach.
STALL_SATURATION = 50

# A counter at 45 is not twice a counter at 22. Squaring pushes the weight into
# the upper range, where a draw claim is in reach and the next irreversible move
# carries real stakes.
STALL_EXPONENT = 2.0

# A repeated position says both players are testing whether the other deviates.
# Four occurrences in game six and no threefold: a modifier on the stall term,
# not a subsystem of its own.
REPETITION_BOOST = 1.25

# Fraction of the way tension moves toward its current drive each ply. Slow
# enough that a single quiet ply does not erase a build.
ATTACK = 0.12

STRUCTURE_LOCKED_SHARE = 0.65

WEIGHTS = {
    "structure": 0.30,
    "stall": 0.24,
    "king": 0.18,
    "hanging": 0.14,
    "eval": 0.14,
}

# How much of the accumulated tension each kind of event discharges.
RELEASE_DEPTH = {
    "pawn_break": 0.55,
    "progress_reset": 0.45,
    "simplification": 0.60,
    "king_safety": 0.40,
    "termination": 1.0,
}

# Chosen once, after the hanging-material term landed, against the corpus below.
# The median ply moves from 0.06 to 0.24 and the share sitting under 0.05 falls
# from 43% to 4%, while the most locked game reaches 0.69 and leaves headroom.
GAMMA = 0.5

DECISIVE_RESULTS = ("1-0", "0-1")

PAWN_BREAK_DROP = 2
SIMPLIFICATION_POINTS = 5

# A king walking out of real danger is an event. A king whose zone happens to be
# one attacker quieter than last ply is not: without a floor and a meaningful
# drop this fires on ordinary shuffling and discharges the build continuously.
SHELTER_PRESSURE_FLOOR = 3
SHELTER_PRESSURE_DROP = 2
DENSITY_MOBILITY_SHARE = 0.55
FORCING_STREAK_FULL = 3
DOTTED_STREAK = 2


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _calibrate(value: float) -> float:
    """Spread the raw scale over the range the axis is defined on.

    The terms combine to a weighted mean, so a real game lands low even when it
    is tense: before this, nearly half of all plies in the corpus sat under 0.05
    and the axis was not calibrated to its own definition. One global exponent,
    identical for every game, applied at emission only -- the carried value stays
    raw so the dynamics are untouched. Monotone, so the ordering across games is
    exactly preserved and no game is rescaled against itself.

    Applied to `tension` and to `cadence_strength`, which is a quantity of
    tension and has to share its scale. Not to `stall`, a raw input term reported
    for diagnosis, nor to `density`, which is a different axis with its own
    meaning -- transforming those would put three unrelated scales through one
    correction fitted to none of them.

    Calibrated against an eight-game corpus. That makes it a calibration, not a
    law; deciding what the number sounds like remains the renderer's business.
    Note that the transform is not additive: a cadence and the tension left
    behind no longer sum to the tension before it, so `cadence_strength` is the
    size of the event, not the fraction discharged.
    """
    return _clamp(value) ** GAMMA


def _structure(frame: dict) -> float:
    locked = _clamp(frame["locked_pawns"] / LOCKED_FULL)
    contact = _clamp(frame["pawn_tension"] / PAWN_TENSION_FULL)
    return STRUCTURE_LOCKED_SHARE * locked + (1 - STRUCTURE_LOCKED_SHARE) * contact


def _king(frame: dict) -> float:
    """Heat on either king. Whose it is does not matter -- a king under attack
    is tension regardless of who stands better."""
    worst = max(frame["king_pressure_w"], frame["king_pressure_b"])
    return _clamp(worst / KING_PRESSURE_FULL)


def _hanging(frame: dict) -> float:
    """Loose material, the transient half of tactical volatility.

    Structure, stall and king heat all read near zero in an open gambit, whose
    whole content is that pieces are hanging. Without this the axis cannot say
    "sharp attack" at all.
    """
    return _clamp(frame["hanging_material"] / HANGING_FULL)


def _stall(frame: dict) -> float:
    raw = _clamp(frame["halfmove_clock"] / STALL_SATURATION) ** STALL_EXPONENT
    if frame["repetition_2"]:
        raw *= REPETITION_BOOST
    return _clamp(raw)


def _density(frame: dict, forcing_streak: int) -> float:
    mobility = _clamp((frame["mobility_w"] + frame["mobility_b"]) / MOBILITY_FULL)
    forcing = _clamp(forcing_streak / FORCING_STREAK_FULL)
    return _clamp(
        DENSITY_MOBILITY_SHARE * mobility + (1 - DENSITY_MOBILITY_SHARE) * forcing
    )


def _is_forcing(frame: dict) -> bool:
    return bool(frame["is_check"] or frame["is_capture"] or frame["forced"])


def _queens(fen: str) -> int:
    """Queens on the board.

    Counted over the piece placement alone: the castling field of a full FEN
    also spells rights with Q and q, so a substring test over the whole string
    reports queens that are not there and misses queens that are.
    """
    return fen.split(" ", 1)[0].count("Q") + fen.split(" ", 1)[0].count("q")


def _release_events(frame: dict, history: list[dict], is_last: bool) -> list[str]:
    """Which release events, if any, this ply constitutes."""
    events: list[str] = []

    if history:
        previous = history[-1]
        if previous["locked_pawns"] - frame["locked_pawns"] >= PAWN_BREAK_DROP:
            events.append("pawn_break")
        if frame["halfmove_clock"] == 0 and previous["halfmove_clock"] > 0:
            events.append("progress_reset")

        # Material *removed*, not the balance between the sides: an even trade
        # leaves the balance where it was while emptying the board. Gated on the
        # ply actually being a capture, so the quiet move after a recapture does
        # not fire the same exchange a second time.
        two_back = history[-2] if len(history) >= 2 else previous
        removed = two_back["material_total"] - frame["material_total"]
        queens_gone = _queens(two_back["fen_after"]) > 0 and _queens(frame["fen_after"]) == 0
        if frame["is_capture"] and (queens_gone or removed >= SIMPLIFICATION_POINTS):
            events.append("simplification")

        mover_pressure = f"king_pressure_{frame['color']}"
        before, now = previous[mover_pressure], frame[mover_pressure]
        escaped = (
            before >= SHELTER_PRESSURE_FLOOR
            and before - now >= SHELTER_PRESSURE_DROP
        )
        if frame["san"].startswith("O-O") or escaped:
            events.append("king_safety")

    if is_last and frame["result"] in DECISIVE_RESULTS:
        # Only a decisive game resolves. A draw is a failure to resolve and an
        # unfinished game is an absence, and forcing a release onto either would
        # say something the game did not.
        events.append("termination")
    return events


def _active_layers(frames: list[dict]) -> list[str]:
    """Which signals this *game* carried, for the badge the web app shows.

    A manifest, not a per-ply flag: per-ply availability is already `think_time`
    and `decision_state`, and saying it twice made the layer appear to switch on
    and off dozens of times because individual replies were premoves.
    """
    layers = ["board"]
    if any(frame["eval_cp"] is not None for frame in frames):
        layers.append("eval")
    if any(frame["time_pressure_w"] is not None
           or frame["time_pressure_b"] is not None for frame in frames):
        layers.append("pressure")
    if any(frame["think_relative"] is not None for frame in frames):
        layers.append("think")
    return layers


def tension_track(frames: list[dict]) -> dict:
    """Single stateful pass over frames, emitting the state track.

    Two scopes. Per game: `game_tempo_scale` and `active_layers`. Per ply, under
    `plies`: `tension`, `pressure_w`, `pressure_b`, `density`, `meter`,
    `cadence_strength` and `stall`.
    """
    plies: list[dict] = []
    if not frames:
        return {"game_tempo_scale": None, "active_layers": [], "plies": plies}

    tension = 0.0
    forcing_streak = 0
    evals: deque[int] = deque(maxlen=EVAL_WINDOW + 1)
    history: list[dict] = []
    last_index = len(frames) - 1

    for index, frame in enumerate(frames):
        terms = {
            "structure": _structure(frame),
            "king": _king(frame),
            "stall": _stall(frame),
            "hanging": _hanging(frame),
        }
        if frame["eval_cp"] is not None:
            evals.append(frame["eval_cp"])
            if len(evals) >= 2:
                swings = [abs(b - a) for a, b in zip(evals, list(evals)[1:])]
                terms["eval"] = _clamp(
                    (sum(swings) / len(swings)) / EVAL_VOLATILITY_FULL
                )

        # Absent terms drop out and the rest renormalise, so a game with no
        # evaluations is read on the same scale as one with them.
        total = sum(WEIGHTS[name] for name in terms)
        drive = sum(WEIGHTS[name] * value for name, value in terms.items()) / total

        tension += ATTACK * (drive - tension)

        events = _release_events(frame, history, index == last_index)
        cadence_strength = 0.0
        if events:
            depth = max(RELEASE_DEPTH[event] for event in events)
            cadence_strength = _clamp(tension * depth)
            tension = _clamp(tension - cadence_strength)

        forcing_streak = forcing_streak + 1 if _is_forcing(frame) else 0

        plies.append(
            {
                "ply": frame["ply"],
                "tension": round(_calibrate(tension), 4),
                "pressure_w": frame["time_pressure_w"],
                "pressure_b": frame["time_pressure_b"],
                "density": round(_density(frame, forcing_streak), 4),
                "meter": "dotted" if forcing_streak >= DOTTED_STREAK else "free",
                "cadence_strength": round(_calibrate(cadence_strength), 6),
                "stall": round(terms["stall"], 4),
            }
        )
        history.append(frame)

    return {
        "game_tempo_scale": frames[0]["game_tempo_scale"],
        "active_layers": _active_layers(frames),
        "plies": plies,
    }


def plot_track(track: dict, out_path) -> None:
    """Plot the state track and save it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e1dd"
    series = ("#2a78d6", "#eb6834", "#1baf7a")
    entries = track["plies"]
    moves = [(entry["ply"] + 1) / 2 for entry in entries]

    figure, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True,
                                gridspec_kw={"hspace": 0.2})
    figure.patch.set_facecolor(surface)

    axes[0].plot(moves, [e["tension"] for e in entries], color=series[0], lw=1.8)
    cadences = [(m, e) for m, e in zip(moves, entries) if e["cadence_strength"] > 0]
    if cadences:
        axes[0].scatter(
            [m for m, _ in cadences],
            [e["tension"] for _, e in cadences],
            s=[12 + 340 * e["cadence_strength"] for _, e in cadences],
            facecolor="none", edgecolor=series[1], lw=1.3, zorder=3,
        )
    axes[0].set_ylabel("tension", fontsize=9.5, color=ink2)
    axes[0].set_ylim(-0.03, 1.03)

    for side, colour, label in (("w", series[0], "White"), ("b", series[1], "Black")):
        points = [(m, e[f"pressure_{side}"]) for m, e in zip(moves, entries)
                  if e[f"pressure_{side}"] is not None]
        if points:
            axes[1].plot([m for m, _ in points], [v for _, v in points],
                         color=colour, lw=1.5, label=label)
    if any(e["pressure_w"] is not None for e in entries):
        axes[1].legend(frameon=False, fontsize=9, labelcolor=ink2, loc="upper left")
    else:
        axes[1].annotate("no clock data — pressure layer off", xy=(0.015, 0.5),
                         xycoords="axes fraction", fontsize=9, color=ink2)
    axes[1].set_ylabel("pressure", fontsize=9.5, color=ink2)
    axes[1].set_ylim(-0.03, 1.03)

    axes[2].plot(moves, [e["density"] for e in entries], color=series[2], lw=1.5)
    dotted = [m for m, e in zip(moves, entries) if e["meter"] == "dotted"]
    for move in dotted:
        axes[2].axvline(move, color=ink2, lw=0.6, alpha=0.18)
    axes[2].set_ylabel("density", fontsize=9.5, color=ink2)
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("Move number", fontsize=9.5, color=ink2)

    for axis in axes:
        axis.set_facecolor(surface)
        axis.grid(True, color=grid, lw=0.8)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(grid)
        axis.tick_params(colors=ink2, labelsize=9)

    axes[0].set_title(str(out_path).rsplit("/", 1)[-1].replace("_tension.png", ""),
                      fontsize=12, color=ink, loc="left", pad=10)
    figure.savefig(out_path, dpi=140, facecolor=surface, bbox_inches="tight")
    plt.close(figure)
