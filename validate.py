"""
Is the expected points fit any good, and does the metric built on it measure
anything?

    make validate ARGS="--league nfl --root ./plays"

`make rates` and `make bounds` measure the constants. This measures the
*models*, and it answers a different kind of question, so it works a
different way: it refits both of them on early seasons only and reports
everything against seasons neither fit has seen. A number quoted off the
shipped release would be a number about games that release was fitted on.

    fit    2014-2022      test  2023-2025

Four things come out, in the order a sceptic asks for them.

**Does it know anything?** `skill` scores the fit against the two baselines
worth beating -- a uniform guess and the base rates -- in log loss over the
seven outcomes and in mean absolute error over the points.

**Is it right, or only confident?** `calibration` bins held-out snaps by what
the fit said and reports what actually happened in each bin. A model can have
skill and still be systematically wrong somewhere; this is where that shows.
The bins are also the chart, because a calibration plot is the one picture
that can't be argued with.

**Is it football?** `surface` is expected points across the field for each
down, next to the observed mean at the same spots. Nobody has to take the
coefficients on trust: a first down on your own 10 is worth what everyone
already knows it's worth, or the fit is wrong.

**Does the metric measure the team?** `split_half` alternates each
team-season's games into halves and asks whether the number from one half
agrees with the other, and whether it predicts that team's scoring. EPA per
play is compared against points per play -- the box-score alternative it has
to beat to be worth computing -- and the two knobs are swept, with the
caveat that `--min-live-share` documents.

Everything is written to `--out` (default `docs/`) as one JSON file and the
charts that go with it, so the report in `docs/epa-validation.md` is a view
of a file this produced rather than a set of numbers someone typed.
"""

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import fire
import numpy as np

import charts
from lucky_ones import GamePlays, group_by_game
from lucky_ones.arrow import DatasetPlaySource, StorePlaySource
from lucky_ones.epa import DEFAULT_CLIP, DEFAULT_WEIGHT_POWER, competitiveness
from lucky_ones.metrics import mean_absolute_error, multiclass_log_loss
from lucky_ones.model import LogisticWinProbability
from lucky_ones.plays import Play, PlaySource
from lucky_ones.points import (
    SCORE_VALUES,
    MultinomialExpectedPoints,
    ScoreKind,
    next_scores,
    scoring_plays,
)
from lucky_ones.state import GameState, final_outcome, iter_states

logger = logging.getLogger("validate")

FIT_SEASONS = range(2014, 2023)
TEST_SEASONS = range(2023, 2026)
WEEKS = range(0, 21)
INFINITE = float("inf")

CLIPS = (1.0, 2.0, 3.0, 4.0, 5.0, 7.0, INFINITE)
# Finely spaced below 0.5, because that is where an interior optimum would
# sit if there is one: a little down-weighting could plausibly improve the
# estimate -- garbage-time snaps are different football -- before the sample
# it costs takes over. A sweep that steps straight from 0 to 0.5 cannot see
# the difference between a peak and a slope.
POWERS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0)
# Wider at the top for the descriptive trial, which is the one where a large
# power stops being an average over the live game and starts being an average
# over the coin-flip snaps inside it.
DESCRIPTION_POWERS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)

LANDMARKS: tuple[tuple[str, dict], ...] = (
    ("1st and 10, own 1", dict(down=1, distance=10, yardline=1)),
    ("1st and 10, own 25", dict(down=1, distance=10, yardline=25)),
    ("1st and 10, midfield", dict(down=1, distance=10, yardline=50)),
    ("1st and 10, opponent 25", dict(down=1, distance=10, yardline=75)),
    ("1st and goal, the 2", dict(down=1, distance=2, yardline=98)),
    ("3rd and 15, own 5", dict(down=3, distance=15, yardline=5)),
    ("4th and 1, midfield", dict(down=4, distance=1, yardline=50)),
)


class Game(dict):
    """One game's material, pulled off the plays once and reused everywhere."""


def _source(root: str | None) -> PlaySource:
    """The local tree if one was named, otherwise the bucket. See `train.py`."""
    if root is not None:
        return DatasetPlaySource(root)
    from endgame_aws.pbp_parquet import get_processed_plays_store

    return StorePlaySource(get_processed_plays_store())


async def _load(source: PlaySource, league: str, seasons: Iterable[int]) -> list[Play]:
    plays: list[Play] = []
    for season in seasons:
        season_plays = await source.load_weeks(league, season, WEEKS)
        logger.info("  %s %s: %d plays", league, season, len(season_plays))
        plays.extend(season_plays)
    return plays


def _prepared(source: PlaySource, league: str, seasons: Sequence[int]) -> list[Game]:
    """Every game of `seasons`, walked once into the shape both commands use."""
    plays = asyncio.run(_load(source, league, seasons))
    games = [game for game in (_prepare(g) for g in group_by_game(plays)) if game]
    if not games:
        raise ValueError(
            f"No {league} games in {list(seasons)}. With --root, check the path "
            "points at the `processed/plays` directory."
        )
    return games


def _prepare(game: GamePlays) -> Game | None:
    """
    Everything downstream needs from one game, computed once.

    States and labels come out together because they are produced by one walk
    and consumed in lockstep; the final score comes along because the
    split-half half of this compares against the box score.
    """
    states = list(iter_states(game))
    if not states:
        return None
    outcome = final_outcome(game)
    return Game(
        game_id=game.game_id,
        season=game.season,
        home=game.home_team_id,
        away=game.away_team_id,
        states=states,
        labels=next_scores(game),
        scored=scoring_plays(list(game.plays)),
        home_score=None if outcome is None else outcome.home_score,
        away_score=None if outcome is None else outcome.away_score,
    )


def _regulation(games: Sequence[Game]) -> tuple[list[GameState], list[ScoreKind]]:
    states: list[GameState] = []
    labels: list[ScoreKind] = []
    for game in games:
        for state, label in zip(game["states"], game["labels"]):
            if not state.is_overtime:
                states.append(state)
                labels.append(label)
    return states, labels


def _fit_models(
    games: Sequence[Game],
) -> tuple[MultinomialExpectedPoints, LogisticWinProbability]:
    """
    Both fits, on the fit seasons alone.

    The win probability model is refit here rather than taken from `MODELS`
    for one reason: it decides the weight on every play in the split-half
    section, and a shipped fit has seen the test seasons. Nothing in this
    script is allowed to have.
    """
    states, labels = _regulation(games)
    points = MultinomialExpectedPoints.fit(states, labels)

    wp_states: list[GameState] = []
    wp_won: list[bool] = []
    for game in games:
        if game["home_score"] is None or game["home_score"] == game["away_score"]:
            continue
        wp_states.extend(game["states"])
        wp_won.extend([game["home_score"] > game["away_score"]] * len(game["states"]))
    return points, LogisticWinProbability.fit(wp_states, wp_won)


def _state(**situation) -> GameState:
    """A landmark situation, early in the first quarter."""
    return GameState(
        game_id="landmark",
        play_id="landmark",
        play_number=1,
        period=1,
        clock_seconds=900,
        seconds_remaining=3600,
        is_overtime=False,
        home_score=0,
        away_score=0,
        offense_is_home=True,
        **situation,
    )


# --- the four measurements ---------------------------------------------


def _skill(
    points: MultinomialExpectedPoints,
    states: Sequence[GameState],
    labels: Sequence[ScoreKind],
    fit_labels: Sequence[ScoreKind],
) -> dict:
    """
    Log loss and mean absolute error against the two baselines worth beating.

    The base rates are the honest baseline and the one to read: a model that
    doesn't beat "how often does each outcome happen, ignoring the
    situation" has learned nothing about football. Uniform is there because
    `log(k)` is the number people know.
    """
    index = {kind: position for position, kind in enumerate(points.kinds)}
    actual_index = np.array([index[kind] for kind in labels])
    actual_value = np.array([SCORE_VALUES[kind] for kind in labels])
    predicted = points.predict(states)

    base_rates = np.bincount(
        [index[kind] for kind in fit_labels], minlength=len(points.kinds)
    ) / len(fit_labels)
    base_proba = np.tile(base_rates, (len(states), 1))
    base_value = float(base_rates @ points.values)

    log_loss = multiclass_log_loss(points.predict_proba(states), actual_index)
    base_log_loss = multiclass_log_loss(base_proba, actual_index)
    mae = mean_absolute_error(predicted, actual_value)
    base_mae = mean_absolute_error(np.full(len(actual_value), base_value), actual_value)
    return {
        "n_snaps": len(states),
        "log_loss": round(log_loss, 4),
        "log_loss_base_rates": round(base_log_loss, 4),
        "log_loss_uniform": round(float(np.log(len(points.kinds))), 4),
        "log_loss_skill": round(1 - log_loss / base_log_loss, 4),
        "mean_absolute_error": round(mae, 3),
        "mean_absolute_error_base_rates": round(base_mae, 3),
        "mean_absolute_error_skill": round(1 - mae / base_mae, 4),
        "variance_explained": round(
            float(
                1
                - np.sum((predicted - actual_value) ** 2)
                / np.sum((actual_value - actual_value.mean()) ** 2)
            ),
            4,
        ),
        "mean_predicted": round(float(predicted.mean()), 4),
        "mean_actual": round(float(actual_value.mean()), 4),
        "bias": round(float(predicted.mean() - actual_value.mean()), 4),
    }


def _calibration(
    points: MultinomialExpectedPoints,
    states: Sequence[GameState],
    labels: Sequence[ScoreKind],
    bins: int = 20,
) -> dict:
    """
    Held-out snaps grouped by what the fit said, against what happened.

    Equal-count bins rather than equal-width, so every point on the chart
    carries the same weight and the tails aren't three snaps wide.
    """
    predicted = points.predict(states)
    actual = np.array([SCORE_VALUES[kind] for kind in labels])
    rows = []
    for chunk in np.array_split(np.argsort(predicted), bins):
        rows.append(
            {
                "predicted": round(float(predicted[chunk].mean()), 4),
                "actual": round(float(actual[chunk].mean()), 4),
                "n": int(len(chunk)),
            }
        )
    gaps = [abs(row["predicted"] - row["actual"]) for row in rows]
    return {
        "bins": rows,
        "worst_gap": round(max(gaps), 4),
        "mean_absolute_gap": round(float(np.mean(gaps)), 4),
    }


def _surface(
    points: MultinomialExpectedPoints,
    states: Sequence[GameState],
    labels: Sequence[ScoreKind],
    step: int = 5,
) -> dict:
    """
    Expected points across the field for each down, fitted against observed.

    The observed side is the whole point: a curve nobody can check is a
    curve. Distance is held at 10 for the fitted line (the modal situation);
    the observed mean takes every distance in the bucket, which is why the
    two can differ a little without either being wrong.
    """
    yardlines = list(range(step, 100, step))
    fitted = {
        str(down): [
            round(
                float(
                    points.predict([_state(down=down, distance=10, yardline=yardline)])[
                        0
                    ]
                ),
                4,
            )
            for yardline in yardlines
        ]
        for down in (1, 2, 3, 4)
    }

    actual = np.array([SCORE_VALUES[kind] for kind in labels])
    down = np.array([state.down for state in states])
    yardline = np.array([state.yardline for state in states])
    observed: dict[str, list] = {}
    for value in (1, 2, 3, 4):
        row = []
        for centre in yardlines:
            picked = (
                (down == value)
                & (yardline >= centre - step / 2)
                & (yardline < centre + step / 2)
            )
            count = int(picked.sum())
            row.append(
                {
                    "yardline": centre,
                    "actual": round(float(actual[picked].mean()), 4) if count else None,
                    "n": count,
                }
            )
        observed[str(value)] = row
    return {"yardlines": yardlines, "fitted": fitted, "observed": observed}


def _landmarks(points: MultinomialExpectedPoints) -> list[dict]:
    """Situations whose value is common knowledge, priced by the fit."""
    return [
        {
            "situation": label,
            "expected_points": round(
                float(points.predict([_state(**situation)])[0]), 3
            ),
        }
        for label, situation in LANDMARKS
    ]


def _play_rows(
    points: MultinomialExpectedPoints,
    win: LogisticWinProbability,
    game: Game,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    One game's raw EPA, win probability and which side had the ball.

    Computed unclipped and unweighted, once, because every variant in the
    sweep is a different reduction of these same three columns -- pricing the
    game once per variant would be thirty-five passes over a season for
    nothing.
    """
    from lucky_ones.epa import play_epa

    plays = play_epa(
        points,
        win,
        game["states"],
        game["scored"],
        clip=INFINITE,
        weight_power=0.0,
    )
    if not plays:
        return None
    return (
        np.array([play.epa for play in plays]),
        np.array([play.win_probability for play in plays]),
        np.array([play.offense_is_home for play in plays]),
    )


def _distribution(
    points: MultinomialExpectedPoints,
    win: LogisticWinProbability,
    games: Sequence[Game],
    clip: float,
) -> dict:
    """The raw EPA distribution the bound is a statement about."""
    everything = []
    weights = []
    for game in games:
        rows = _play_rows(points, win, game)
        if rows is None:
            continue
        epa, wp, _ = rows
        everything.append(epa)
        weights.append(competitiveness(wp, DEFAULT_WEIGHT_POWER))
    raw = np.concatenate(everything)
    weight = np.concatenate(weights)
    edges = np.arange(-10, 10.25, 0.25)
    counts, _ = np.histogram(raw, bins=edges)
    quantiles = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
    absolute = (0.9, 0.95, 0.975, 0.99, 0.995, 0.999)
    return {
        "n_plays": int(len(raw)),
        "histogram": {
            "edges": [round(float(edge), 3) for edge in edges],
            "counts": [int(count) for count in counts],
        },
        "quantiles": {
            f"p{q * 100:g}": round(float(v), 3)
            for q, v in zip(quantiles, np.quantile(raw, quantiles))
        },
        "abs_quantiles": {
            f"p{q * 100:g}": round(float(v), 3)
            for q, v in zip(absolute, np.quantile(np.abs(raw), absolute))
        },
        "beyond_clip": round(float(np.mean(np.abs(raw) > clip)), 4),
        "effective_play_share": round(float(weight.mean()), 4),
    }


def _description_trial(
    points: MultinomialExpectedPoints,
    win: LogisticWinProbability,
    games: Sequence[Game],
    *,
    clip: float,
    powers: Sequence[float],
    min_live_snaps: int,
) -> dict:
    """
    The other half of the weighting question, and the one no correlation can
    answer: is the weighted number a better *description* of one game?

    Everything else here is predictive -- it asks what a number from half a
    season says about the other half. The complaint the weighting exists to
    answer isn't predictive at all. It is "this team's 0.31 wasn't real, it
    was a 45-point rout", and that is a claim about a single game.

    So: take the unweighted mean over only the snaps where the game was still
    in doubt, 0.2 to 0.8 win probability, and call that what the team did
    while it mattered. Then ask how close each whole-game number lands to it.

    **This is a definition, not an outside truth, and the direction of the
    result is nearly baked in** -- a weighting designed to fade out garbage
    time will of course land nearer a garbage-time-free average than one that
    doesn't. Two things make it worth measuring anyway.

    The first is the size. Nothing so far says how much garbage time actually
    moves a team's game number, and "how big is the problem" is the question
    that decides whether solving it is worth a third of the sample.

    The second is that the answer is *not* monotone in the power, which is
    what stops this from being arithmetic dressed up as a test. `4p(1-p)`
    fades inside the live window too -- a snap at 0.2 carries 0.64, not 1 --
    so as the power climbs the weight concentrates on the coin-flip snaps and
    the number stops being an average over the live game at all. There is a
    power that recovers the live average best and it is neither 0 nor
    infinity. Where it falls is a real finding, and it is a finding about the
    shipped default.
    """
    from lucky_ones.epa import play_epa

    rows: list[dict] = []
    for game in games:
        plays = play_epa(
            points, win, game["states"], game["scored"], clip=clip, weight_power=0.0
        )
        if not plays:
            continue
        for home in (True, False):
            side = [play for play in plays if play.offense_is_home == home]
            if not side:
                continue
            epa = np.array([play.bounded for play in side])
            wp = np.array([play.win_probability for play in side])
            live = (wp > 0.2) & (wp < 0.8)
            if live.sum() < min_live_snaps:
                continue
            estimates, effective = {}, {}
            for power in powers:
                weight = competitiveness(wp, power)
                total, square = weight.sum(), np.square(weight).sum()
                estimates[power] = float(epa @ weight / total)
                # What this power leaves of the sample, so the descriptive
                # gain it buys can be read against the precision it spends.
                effective[power] = float((total * total / square) / len(epa))
            rows.append(
                {
                    "live_mean": float(epa[live].mean()),
                    "live_share": float(live.mean()),
                    "estimates": estimates,
                    "effective": effective,
                }
            )

    if len(rows) < 50:
        return {"team_games": len(rows), "note": "too few"}

    shares = np.array([row["live_share"] for row in rows])
    cuts = np.quantile(shares, [1 / 3, 2 / 3])
    bands = {
        "mostly decided": shares <= cuts[0],
        "mixed": (shares > cuts[0]) & (shares <= cuts[1]),
        "mostly live": shares > cuts[1],
    }

    def summarise(power: float, picked: np.ndarray) -> dict:
        error = np.array(
            [
                row["estimates"][power] - row["live_mean"]
                for row, keep in zip(rows, picked)
                if keep
            ]
        )
        kept = np.array(
            [row["effective"][power] for row, keep in zip(rows, picked) if keep]
        )
        plain = np.array(
            [
                abs(row["estimates"][0.0] - row["live_mean"])
                for row, keep in zip(rows, picked)
                if keep
            ]
        )
        return {
            "team_games": int(len(error)),
            "mean_abs_error": round(float(np.mean(np.abs(error))), 4),
            "rmse": round(float(np.sqrt(np.mean(np.square(error)))), 4),
            "bias": round(float(np.mean(error)), 4),
            "p90_abs_error": round(float(np.quantile(np.abs(error), 0.9)), 4),
            # The two halves of the trade side by side: how much of the
            # garbage-time gap this power closes, and what it leaves of the
            # sample to close it with.
            "gap_closed": round(
                1.0 - float(np.mean(np.abs(error)) / np.mean(plain)), 4
            ),
            "effective_sample_share": round(float(np.mean(kept)), 4),
        }

    everything = np.ones(len(rows), dtype=bool)
    by_power = {str(power): summarise(power, everything) for power in powers}
    best = min(by_power, key=lambda key: by_power[key]["mean_abs_error"])
    return {
        "team_games": len(rows),
        "min_live_snaps": min_live_snaps,
        "live_share": {
            "median": round(float(np.median(shares)), 4),
            "p10": round(float(np.quantile(shares, 0.1)), 4),
            "p90": round(float(np.quantile(shares, 0.9)), 4),
        },
        "by_power": by_power,
        "best_power": float(best),
        "by_competitiveness": {
            name: {str(power): summarise(power, picked) for power in powers}
            for name, picked in bands.items()
        },
    }


def _correlate(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson r over the rows where both sides are finite.

    A team-half can be missing a column -- a dealt set with no competitive
    game in it has no live number -- and dropping those pairs is right, but
    dropping so many that the correlation is off a handful of teams is not.
    Below ten usable pairs it says nothing rather than something noisy.
    """
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 10:
        return np.nan
    return float(np.corrcoef(x[usable], y[usable])[0, 1])


def _split_half(
    points: MultinomialExpectedPoints,
    win: LogisticWinProbability,
    games: Sequence[Game],
    *,
    min_games: int,
    min_live_share: float,
    draws: int,
    seed: int,
) -> dict:
    """
    Does the number measure the team?

    Each team-season's games are dealt at random into two sets, the number is
    computed over each, and the two are correlated across teams. Then it is
    done again, `draws` times, with a fresh random deal and a fresh bootstrap
    resample of the team-seasons every time -- so the spread reported covers
    both which games landed together and which teams were in the sample.

    **Randomly, not by alternating games**, which was the first design and was
    wrong. Schedules tend to alternate home and away, so every-other-game can
    put most of a team's home games in one set and its away games in the
    other, and home teams score more. That would have shown up as measurement
    noise in every variant and been read as the metric being unreliable.

    Three targets, because they disagree and the disagreement is the finding:

    - `reliability` -- one set's number against the other's. A pure
      measurement of noise, and the weighting can only lose on it:
      down-weighting shrinks the effective sample, and a smaller sample agrees
      with itself less.
    - `all_scoring` -- one set's number against the other set's points per
      play. Tilted towards the unweighted number, whose target includes
      exactly the garbage-time points the weighting removes.
    - `live_scoring` -- the same, over only those games where at least
      `min_live_share` of snaps were still in doubt. The one built to favour
      the weighting.

    **What none of them can settle is the weighting**, and the reason is the
    schedule again. Down-weighting discards blowouts, blowouts are the
    mismatches, and so the weighted metric's two sets are built from a smaller
    and differently-selected pool of games than the unweighted one -- most of
    all in NCAAFB, which is the only league here with the team-seasons to
    resolve a small difference. The noise isn't common to the two variants, so
    the comparison between them isn't clean. Read the sweep as a description
    of what each setting costs, not as a fitted value. `DEFAULT_CLIP` is
    settled by `train.py bounds` and by the football; `DEFAULT_WEIGHT_POWER`
    is a statement about what you want the number to mean, and this is the
    measurement that says so rather than assuming it.

    What the comparison *is* clean for is EPA against points per play: both
    are computed over the same games and the same deal, so whatever the
    schedule does, it does to both.
    """
    variants: dict[str, tuple[float, float]] = {
        "shipped": (DEFAULT_CLIP, DEFAULT_WEIGHT_POWER),
        "unadjusted": (INFINITE, 0.0),
    }
    variants.update({f"clip {clip:g}": (clip, 0.0) for clip in CLIPS})
    variants.update({f"power {power:g}": (DEFAULT_CLIP, power) for power in POWERS})

    keys = ["shipped", "unadjusted", "points/play"]
    keys += [name for name in variants if name.startswith(("clip", "power"))]
    targets = ("__all", "__live")

    # team-season -> (games, len(keys) + len(targets), 2), the numerator and
    # denominator each game contributes to each column. Dealing games into
    # sets is then a sum over an axis, which is what makes `draws` deals
    # affordable.
    per_team: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    columns = keys + list(targets)
    for game in games:
        rows = _play_rows(points, win, game)
        if rows is None:
            continue
        epa, wp, is_home = rows
        for team, home in ((game["home"], True), (game["away"], False)):
            side = is_home == home
            if not side.any():
                continue
            mine, theirs = epa[side], wp[side]
            entry = {}
            for name, (clip, power) in variants.items():
                weight = competitiveness(theirs, power)
                entry[name] = (
                    float(np.sum(weight * np.clip(mine, -clip, clip))),
                    float(np.sum(weight)),
                )
            scored = float((game["home_score"] if home else game["away_score"]) or 0)
            snaps = float(side.sum())
            entry["points/play"] = (scored, snaps)
            entry["__all"] = (scored, snaps)
            # Points can't be attributed per snap, so "live" has to select
            # games rather than plays: this game counts only if most of it was
            # still in doubt. The target stays box-score points either way.
            live = float(((theirs > 0.2) & (theirs < 0.8)).mean())
            entry["__live"] = (scored, snaps) if live >= min_live_share else (0.0, 0.0)
            per_team[team][game["season"]].append([entry[column] for column in columns])

    seasons = [
        np.array(rows, dtype=float)
        for team in per_team.values()
        for rows in team.values()
        if len(rows) >= min_games
    ]
    if len(seasons) < 20:
        return {"team_seasons": len(seasons), "metrics": {}, "note": "too few"}

    position = {name: index for index, name in enumerate(columns)}
    rng = np.random.default_rng(seed)

    def deal() -> tuple[np.ndarray, np.ndarray]:
        """
        One random deal: `(2, team_seasons, columns)` of ratios.

        A team-season whose half has no denominator for some column -- a set
        with no live game in it, most often -- is dropped from this deal
        rather than from the study, which is why the deals are averaged
        rather than pooled.
        """
        halves = []
        for table in seasons:
            order = rng.permutation(len(table))
            cut = len(table) // 2
            pair = []
            for picked in (order[:cut], order[cut : cut * 2]):
                totals = table[picked].sum(axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    pair.append(
                        np.where(totals[:, 1] > 0, totals[:, 0] / totals[:, 1], np.nan)
                    )
            halves.append(pair)
        first = np.array([pair[0] for pair in halves])
        second = np.array([pair[1] for pair in halves])
        return first, second

    # Every draw is a fresh deal *and* a fresh resample of team-seasons, so
    # the spread covers both. Every column is scored on the same draw, which
    # is what makes the paired differences below meaningful.
    samples = {
        key: {
            target: np.empty(draws)
            for target in ("reliability", "all_scoring", "live_scoring")
        }
        for key in keys
    }
    for draw in range(draws):
        first, second = deal()
        picked = rng.integers(0, len(seasons), len(seasons))
        first, second = first[picked], second[picked]
        for key in keys:
            column = position[key]
            samples[key]["reliability"][draw] = _correlate(
                first[:, column], second[:, column]
            )
            for name, target in (("all_scoring", "__all"), ("live_scoring", "__live")):
                # Pooled both ways: each set's number against the other set's
                # outcome, so no team contributes only one direction.
                metric = np.concatenate([first[:, column], second[:, column]])
                outcome = np.concatenate(
                    [second[:, position[target]], first[:, position[target]]]
                )
                samples[key][name][draw] = _correlate(metric, outcome)

    def summarise(sample: np.ndarray) -> dict:
        clean = sample[np.isfinite(sample)]
        low, high = np.percentile(clean, [2.5, 97.5])
        return {
            "r": round(float(clean.mean()), 4),
            "low": round(float(low), 4),
            "high": round(float(high), 4),
        }

    def compare(key: str, against: str) -> dict:
        out = {}
        for target in ("reliability", "all_scoring", "live_scoring"):
            difference = samples[key][target] - samples[against][target]
            clean = difference[np.isfinite(difference)]
            low, high = np.percentile(clean, [2.5, 97.5])
            out[target] = {
                "difference": round(float(clean.mean()), 4),
                "low": round(float(low), 4),
                "high": round(float(high), 4),
                "p_better": round(float((clean > 0).mean()), 3),
            }
        return out

    return {
        "team_seasons": len(seasons),
        "min_games": min_games,
        "min_live_share": min_live_share,
        "draws": draws,
        "metrics": {
            key: {
                target: summarise(samples[key][target])
                for target in ("reliability", "all_scoring", "live_scoring")
            }
            for key in keys
        },
        "epa_vs_points": compare("shipped", "points/play"),
        "unadjusted_vs_points": compare("unadjusted", "points/play"),
        "against_unadjusted": {
            key: compare(key, "unadjusted")
            for key in keys
            if key.startswith(("clip", "power"))
        },
    }


def _weighting_trial(
    points: MultinomialExpectedPoints,
    win: LogisticWinProbability,
    games: Sequence[Game],
    *,
    power: float,
    min_games: int,
    min_live_share: float,
    draws: int,
    seed: int,
) -> dict:
    """
    The test `_split_half` can't do: what the weighting costs *at a fixed
    sample*, and what it does when every variant sees the same games.

    The sweep in `_split_half` compares a weighted number against an
    unweighted one, and those two are not computed over the same football.
    Down-weighting a blowout to near nothing effectively removes it, so the
    weighted variant works from a smaller and differently-chosen pool of
    games. Two things move at once -- how snaps are combined, which is the
    thing being argued about, and how many snaps survive, which is an
    artifact of the argument. Either could produce the penalty that sweep
    reports.

    So this holds each of them still in turn.

    **Equal effective sample.** A weighted mean over snaps carries the
    precision of a smaller unweighted one, and Kish's formula says how much
    smaller: `n_eff = (sum w)^2 / sum w^2`. So the control is the *unweighted*
    number over a random thinning of the same snaps down to that same
    `n_eff` -- same estimand as the full unweighted number, same precision as
    the weighted one. Then:

    - weighted beats thinned  -> the weighting picks better snaps, and the
      penalty in the sweep was the sample it spent.
    - weighted ties thinned   -> the weighting is precisely a sample cost and
      buys no accuracy at all.
    - weighted loses to thinned -> the weighting picks *worse* snaps, and
      spends sample to do it.

    Thinning is Bernoulli at `n_eff / n`, drawn fresh every time, so the
    control carries real sampling noise rather than a normal approximation to
    it.

    **Fixed game set.** Separately, both variants are restricted to the games
    that stayed in doubt -- at least `min_live_share` of that team's snaps
    between 0.2 and 0.8 win probability. Now the pool is identical by
    construction and the weighting can only re-combine snaps *within* games it
    was always going to keep. Whatever survives here is the weighting's own
    effect, with the selection confound gone.

    Reported against the same three targets as `_split_half`, on the same
    random deals, so every number on the page is comparable.
    """
    rng = np.random.default_rng(seed)

    per_team: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for game in games:
        rows = _play_rows(points, win, game)
        if rows is None:
            continue
        epa, wp, is_home = rows
        for team, home in ((game["home"], True), (game["away"], False)):
            side = is_home == home
            if not side.any():
                continue
            mine, theirs = epa[side], wp[side]
            live = float(((theirs > 0.2) & (theirs < 0.8)).mean())
            per_team[team][game["season"]].append(
                {
                    "epa": mine,
                    "weight": competitiveness(theirs, power),
                    "points": float(
                        (game["home_score"] if home else game["away_score"]) or 0
                    ),
                    "snaps": float(side.sum()),
                    "live": live >= min_live_share,
                }
            )

    seasons = [
        rows
        for team in per_team.values()
        for rows in team.values()
        if len(rows) >= min_games
    ]
    if len(seasons) < 20:
        return {"team_seasons": len(seasons), "note": "too few"}

    def reduce(picked: Sequence[dict]) -> dict[str, float]:
        """Every variant over one dealt half of one team-season."""
        epa = np.concatenate([row["epa"] for row in picked])
        weight = np.concatenate([row["weight"] for row in picked])
        total, square = weight.sum(), np.square(weight).sum()
        out: dict[str, float] = {
            "unadjusted": float(epa.mean()),
            "weighted": float(epa @ weight / total) if total > 0 else np.nan,
        }
        # Kish: the unweighted sample size that would carry this precision.
        effective = (total * total / square) if square > 0 else 0.0
        keep = rng.random(len(epa)) < min(effective / len(epa), 1.0)
        out["thinned"] = float(epa[keep].mean()) if keep.any() else np.nan
        out["n_eff_share"] = float(effective / len(epa))

        live = [row for row in picked if row["live"]]
        if live:
            live_epa = np.concatenate([row["epa"] for row in live])
            live_weight = np.concatenate([row["weight"] for row in live])
            live_total = live_weight.sum()
            live_square = np.square(live_weight).sum()
            out["live_unadjusted"] = float(live_epa.mean())
            out["live_weighted"] = (
                float(live_epa @ live_weight / live_total) if live_total > 0 else np.nan
            )
            # Thinned here too. Fixing the game set removes the confound about
            # *which games* are in the pool, but the weighting still spends
            # sample inside the games it keeps -- so without this the
            # same-games comparison would just be the sample cost again.
            live_effective = (
                (live_total * live_total / live_square) if live_square > 0 else 0.0
            )
            live_keep = rng.random(len(live_epa)) < min(
                live_effective / len(live_epa), 1.0
            )
            out["live_thinned"] = (
                float(live_epa[live_keep].mean()) if live_keep.any() else np.nan
            )
            out["live_n_eff_share"] = float(live_effective / len(live_epa))
        else:
            out["live_unadjusted"] = np.nan
            out["live_weighted"] = np.nan
            out["live_thinned"] = np.nan
            out["live_n_eff_share"] = np.nan

        scored = sum(row["points"] for row in picked)
        snaps = sum(row["snaps"] for row in picked)
        out["__all"] = scored / snaps if snaps else np.nan
        live_scored = sum(row["points"] for row in live)
        live_snaps = sum(row["snaps"] for row in live)
        out["__live"] = live_scored / live_snaps if live_snaps else np.nan
        return out

    variants = (
        "unadjusted",
        "weighted",
        "thinned",
        "live_unadjusted",
        "live_weighted",
        "live_thinned",
    )
    targets = ("reliability", "all_scoring", "live_scoring")
    samples = {
        name: {target: np.empty(draws) for target in targets} for name in variants
    }
    shares: list[float] = []
    live_shares: list[float] = []

    for draw in range(draws):
        halves = []
        for rows in seasons:
            order = rng.permutation(len(rows))
            cut = len(rows) // 2
            halves.append(
                (
                    reduce([rows[index] for index in order[:cut]]),
                    reduce([rows[index] for index in order[cut : cut * 2]]),
                )
            )
        picked = rng.integers(0, len(halves), len(halves))
        first = [halves[index][0] for index in picked]
        second = [halves[index][1] for index in picked]
        if draw == 0:
            shares = [half["n_eff_share"] for pair in halves for half in pair]
            live_shares = [half["live_n_eff_share"] for pair in halves for half in pair]

        def column(rows, key):
            return np.array([row[key] for row in rows])

        for name in variants:
            samples[name]["reliability"][draw] = _correlate(
                column(first, name), column(second, name)
            )
            for target, key in (("all_scoring", "__all"), ("live_scoring", "__live")):
                metric = np.concatenate([column(first, name), column(second, name)])
                outcome = np.concatenate([column(second, key), column(first, key)])
                samples[name][target][draw] = _correlate(metric, outcome)

    def summarise(sample: np.ndarray) -> dict:
        clean = sample[np.isfinite(sample)]
        low, high = np.percentile(clean, [2.5, 97.5])
        return {
            "r": round(float(clean.mean()), 4),
            "low": round(float(low), 4),
            "high": round(float(high), 4),
        }

    def compare(key: str, against: str) -> dict:
        out = {}
        for target in targets:
            difference = samples[key][target] - samples[against][target]
            clean = difference[np.isfinite(difference)]
            low, high = np.percentile(clean, [2.5, 97.5])
            out[target] = {
                "difference": round(float(clean.mean()), 4),
                "low": round(float(low), 4),
                "high": round(float(high), 4),
                "p_better": round(float((clean > 0).mean()), 3),
            }
        return out

    usable = [share for share in live_shares if np.isfinite(share)]
    return {
        "power": power,
        "team_seasons": len(seasons),
        "draws": draws,
        "min_live_share": min_live_share,
        "effective_sample_share": round(float(np.mean(shares)), 4),
        "live_effective_sample_share": (
            round(float(np.mean(usable)), 4) if usable else None
        ),
        "metrics": {
            name: {target: summarise(samples[name][target]) for target in targets}
            for name in variants
        },
        # The decisive one: same precision, different snaps.
        "weighted_vs_thinned": compare("weighted", "thinned"),
        # How much of the sweep's penalty was simply the sample it spent.
        "thinned_vs_unadjusted": compare("thinned", "unadjusted"),
        "weighted_vs_unadjusted": compare("weighted", "unadjusted"),
        # Same games for both, so only the within-game re-combination is left.
        "live_weighted_vs_live_unadjusted": compare("live_weighted", "live_unadjusted"),
        "live_weighted_vs_live_thinned": compare("live_weighted", "live_thinned"),
    }


# --- the charts ---------------------------------------------------------


def _calibration_chart(report: dict) -> str:
    rows = report["calibration"]["bins"]
    predicted = [row["predicted"] for row in rows]
    actual = [row["actual"] for row in rows]
    low = min(min(predicted), min(actual)) - 0.4
    high = max(max(predicted), max(actual)) + 0.4
    axes = charts.Axes(560, 400, (low, high), (low, high))
    ticks = charts.nice_ticks(low, high, 7)
    axes.frame(
        ticks,
        ticks,
        xlabel="expected points the fit predicted",
        ylabel="points the next score was actually worth",
        xformat="{:+g}",
        yformat="{:+g}",
    )
    axes.line([(low, low), (high, high)], color=charts.MUTED, width=1.2, dash="5 4")
    axes.text(
        high - 0.15,
        high - 0.75,
        "perfect calibration",
        size=10,
        color=charts.MUTED,
        anchor="end",
        italic=True,
    )
    axes.dots(zip(predicted, actual), color=charts.BLUE, radius=4, edge=charts.GROUND)
    league = report["league"].upper()
    return axes.render(
        f"{league} expected points is calibrated out of sample",
        f"{report['test_seasons'][0]}-{report['test_seasons'][-1]}, "
        f"{report['skill']['n_snaps']:,} held-out snaps in 20 equal-count bins; "
        f"worst bin off by {report['calibration']['worst_gap']:.2f} points",
    )


def _surface_chart(report: dict) -> str:
    surface = report["surface"]
    yardlines = surface["yardlines"]
    axes = charts.Axes(620, 400, (0, 100), (-3.0, 7.0))
    axes.frame(
        [0, 10, 25, 50, 75, 90, 100],
        charts.nice_ticks(-3, 7, 6),
        xlabel="yards from the offense's own goal line",
        ylabel="expected points",
        yformat="{:+g}",
        xtick_labels=["0", "own 10", "own 25", "50", "opp 25", "opp 10", "100"],
    )
    axes.hline(0.0, color=charts.MUTED, dash="2 3")
    for index, down in enumerate(("1", "2", "3", "4")):
        color = charts.SERIES[index]
        axes.line(list(zip(yardlines, surface["fitted"][down])), color=color, width=2.0)
        axes.dots(
            (
                (row["yardline"], row["actual"])
                for row in surface["observed"][down]
                if row["actual"] is not None and row["n"] >= 200
            ),
            color=color,
            radius=2.6,
            opacity=0.55,
        )
    axes.legend(
        [
            (f"{down} down", charts.SERIES[i])
            for i, down in enumerate(("1st", "2nd", "3rd", "4th"))
        ],
        axes.left + 14,
        axes.top + 16,
    )
    axes.text(
        axes.right - 6,
        axes.top + 16,
        "lines: the fit    dots: observed",
        size=10,
        color=charts.MUTED,
        anchor="end",
        pixels=True,
        italic=True,
    )
    return axes.render(
        f"{report['league'].upper()} expected points is the football everyone knows",
        f"fitted on {report['fit_seasons'][0]}-{report['fit_seasons'][-1]}, "
        f"dots are held-out {report['test_seasons'][0]}-"
        f"{report['test_seasons'][-1]} means over buckets of 200+ snaps",
    )


def _distribution_chart(report: dict) -> str:
    histogram = report["distribution"]["histogram"]
    edges = histogram["edges"]
    counts = histogram["counts"]
    total = sum(counts) or 1
    share = [count / total for count in counts]
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    axes = charts.Axes(620, 360, (-10, 10), (0, max(share) * 1.12))
    axes.frame(
        charts.nice_ticks(-10, 10, 9),
        charts.nice_ticks(0, max(share) * 1.12, 5),
        xlabel="expected points added on one play",
        ylabel="share of plays",
        xformat="{:+g}",
        yformat="{:.0%}",
    )
    clip = report["clip"]
    inside = [(c, s) for c, s in zip(centres, share) if abs(c) <= clip]
    outside = [(c, s) for c, s in zip(centres, share) if abs(c) > clip]
    axes.bars(inside, width=0.25, color=charts.BLUE, opacity=0.75)
    axes.bars(outside, width=0.25, color=charts.RED, opacity=0.85)
    for sign in (-1, 1):
        axes.vline(sign * clip, color=charts.RED, dash="4 3")
    beyond = report["distribution"]["beyond_clip"]
    axes.text(
        clip + 0.4,
        max(share) * 0.86,
        f"bound at ±{clip:g}",
        size=10.5,
        color=charts.RED,
        weight="600",
    )
    axes.text(
        clip + 0.4,
        max(share) * 0.78,
        f"{beyond:.1%} of plays",
        size=10,
        color=charts.RED,
    )
    return axes.render(
        f"{report['league'].upper()} the bound touches one play in a hundred",
        f"{report['distribution']['n_plays']:,} held-out plays, "
        f"{report['test_seasons'][0]}-{report['test_seasons'][-1]}; "
        f"99th percentile of |EPA| is "
        f"{report['distribution']['abs_quantiles']['p99']:.2f}",
    )


def _weighting_chart(report: dict) -> str:
    axes = charts.Axes(560, 330, (0, 1), (0, 1.05))
    axes.frame(
        [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        charts.nice_ticks(0, 1, 6),
        xlabel="home win probability at the snap",
        ylabel="weight the play carries",
        xformat="{:g}",
    )
    for index, power in enumerate((0.5, 1.0, 2.0, 3.0)):
        curve = [(p / 100, competitiveness(p / 100, power)) for p in range(0, 101)]
        axes.line(curve, color=charts.SERIES[index], width=2.0)
    axes.line([(0, 1.0), (1, 1.0)], color=charts.MUTED, width=1.4, dash="5 4")
    axes.text(
        0.03, 1.0, "power 0 — no weighting", size=10, color=charts.MUTED, italic=True
    )
    axes.legend(
        [
            (
                f"power {power:g}"
                + (" (shipped)" if power == report["weight_power"] else ""),
                color,
            )
            for power, color in zip((0.5, 1.0, 2.0, 3.0), charts.SERIES)
        ],
        axes.right - 118,
        axes.top + 16,
    )
    share = report["distribution"]["effective_play_share"]
    return axes.render(
        f"The weighting: 4p(1−p) raised to the power, shipped at "
        f"{report['weight_power']:g}",
        f"{report['league'].upper()} keeps {share:.0%} of its snaps' weight at "
        f"the shipped power of {report['weight_power']:g}",
    )


def _split_half_chart(report: dict) -> str:
    split = report["split_half"]
    if not split.get("metrics"):
        return ""
    rows = [
        ("EPA/play, shipped", "shipped", charts.BLUE),
        ("EPA/play, unadjusted", "unadjusted", charts.GREEN),
        ("points/play (box score)", "points/play", charts.ORANGE),
    ]
    targets = (
        ("reliability", "agrees with its own other half"),
        ("all_scoring", "predicts points per play"),
        ("live_scoring", "predicts points in live games"),
    )
    # The axis comes from the numbers rather than from a guess about them: a
    # correlation is only bounded by (-1, 1), and on a small or noisy sample
    # the interval really can cross zero.
    spread = [
        split["metrics"][key][target][field]
        for _, key, _ in rows
        for target, _ in targets
        for field in ("r", "low", "high")
    ]
    low = min(0.0, min(spread) - 0.03)
    high = max(spread) + 0.08
    # x carries the correlation, y carries the category rows: three groups of
    # three bars, laid out downwards from 2.55.
    axes = charts.Axes(620, 380, (low, high), (0.0, 3.0), pad=(150, 18, 46, 44))
    axes.frame(
        charts.nice_ticks(low, high, 8),
        [],
        xlabel="correlation between one random half of a season and the other",
    )
    if low < 0.0:
        axes.vline(0.0, color=charts.MUTED, dash="2 3")
    for group, (target, label) in enumerate(targets):
        base = 2.55 - group * 0.95
        axes.text(
            axes.left - 8,
            axes.y(base + 0.30) + 4,
            label,
            size=10.5,
            color=charts.INK,
            anchor="end",
            pixels=True,
            weight="600",
        )
        for index, (name, key, color) in enumerate(rows):
            value = split["metrics"][key][target]
            y = base - index * 0.24
            start, end = sorted((axes.x(0.0), axes.x(value["r"])))
            axes.raw(
                f'<rect x="{start:.2f}" y="{axes.y(y) - 7:.2f}" '
                f'width="{max(end - start, 1):.2f}" '
                f'height="14" fill="{color}" opacity="0.85" rx="2"/>'
            )
            # Clamped, because a bar can be drawn off the end of its axis and
            # still look like a chart.
            whisker = (
                min(max(axes.x(value["low"]), axes.left), axes.right),
                min(max(axes.x(value["high"]), axes.left), axes.right),
            )
            axes.raw(
                f'<line x1="{whisker[0]:.2f}" y1="{axes.y(y):.2f}" '
                f'x2="{whisker[1]:.2f}" y2="{axes.y(y):.2f}" '
                f'stroke="{charts.INK}" stroke-width="1.2" opacity="0.55"/>'
            )
            axes.text(
                axes.x(value["r"]) + 6,
                axes.y(y) + 4,
                f"{value['r']:.3f}",
                size=10,
                color=charts.INK,
                pixels=True,
            )
            if group == 0:
                axes.text(
                    axes.left - 8,
                    axes.y(y) + 4,
                    name,
                    size=10,
                    color=charts.MUTED,
                    anchor="end",
                    pixels=True,
                )
    return axes.render(
        f"{report['league'].upper()} EPA per play beats the box score on all three",
        f"{split['team_seasons']} team-seasons, "
        f"{report['test_seasons'][0]}-{report['test_seasons'][-1]}; "
        "bars are bootstrap means, whiskers 95% intervals",
    )


def _trial_chart(report: dict) -> str:
    """
    The three variants side by side, which is the whole argument in one
    picture: `thinned` sits level with `weighted`, and both sit below
    `unadjusted` by the same amount.
    """
    if "metrics" not in report:
        return ""
    rows = [
        ("no weighting", "unadjusted", charts.GREEN),
        ("no weighting, thinned to the same precision", "thinned", charts.MUTED),
        ("weighted, power 1", "weighted", charts.BLUE),
    ]
    targets = (
        ("reliability", "agrees with its own other half"),
        ("all_scoring", "predicts points per play"),
        ("live_scoring", "predicts points in live games"),
    )
    spread = [
        report["metrics"][key][target][field]
        for _, key, _ in rows
        for target, _ in targets
        for field in ("r", "low", "high")
    ]
    low, high = min(0.0, min(spread) - 0.03), max(spread) + 0.09
    axes = charts.Axes(660, 400, (low, high), (0.0, 3.0), pad=(268, 18, 50, 44))
    axes.frame(
        charts.nice_ticks(low, high, 7),
        [],
        xlabel="correlation between one random half of a season and the other",
    )
    for group, (target, label) in enumerate(targets):
        base = 2.55 - group * 0.95
        axes.text(
            axes.left - 8,
            axes.y(base + 0.30) + 4,
            label,
            size=10.5,
            color=charts.INK,
            anchor="end",
            pixels=True,
            weight="600",
        )
        for index, (name, key, color) in enumerate(rows):
            value = report["metrics"][key][target]
            y = base - index * 0.24
            start, end = sorted((axes.x(0.0), axes.x(value["r"])))
            axes.raw(
                f'<rect x="{start:.2f}" y="{axes.y(y) - 7:.2f}" '
                f'width="{max(end - start, 1):.2f}" height="14" '
                f'fill="{color}" opacity="0.85" rx="2"/>'
            )
            whisker = tuple(
                min(max(axes.x(value[field]), axes.left), axes.right)
                for field in ("low", "high")
            )
            axes.raw(
                f'<line x1="{whisker[0]:.2f}" y1="{axes.y(y):.2f}" '
                f'x2="{whisker[1]:.2f}" y2="{axes.y(y):.2f}" '
                f'stroke="{charts.INK}" stroke-width="1.2" opacity="0.55"/>'
            )
            axes.text(
                axes.x(value["r"]) + 6,
                axes.y(y) + 4,
                f"{value['r']:.3f}",
                size=10,
                color=charts.INK,
                pixels=True,
            )
            if group == 0:
                axes.text(
                    axes.left - 8,
                    axes.y(y) + 4,
                    name,
                    size=10,
                    color=charts.MUTED,
                    anchor="end",
                    pixels=True,
                )
    kept = report["effective_sample_share"]
    return axes.render(
        f"{report['league'].upper()} the weighting costs precision, not accuracy",
        f"{report['team_seasons']} team-seasons, {report['draws']} random deals. "
        f"Weighting leaves {kept:.0%} of the effective sample; the grey bar is "
        f"the unweighted number thinned to that same {kept:.0%}.",
    )


def _description_chart(report: dict) -> str:
    """
    Distance from the live-only mean against the power, which is the shape
    that matters: it falls, bottoms out, and climbs again.
    """
    description = report.get("description", {})
    if "by_power" not in description:
        return ""
    powers = sorted(float(power) for power in description["by_power"])

    def series(rows: dict) -> list[tuple[float, float]]:
        return [(power, rows[str(power)]["mean_abs_error"]) for power in powers]

    overall = series(description["by_power"])
    bands = [
        ("mostly decided games", "mostly decided", charts.RED),
        ("mixed", "mixed", charts.ORANGE),
        ("mostly live games", "mostly live", charts.GREEN),
    ]
    everything = [value for _, value in overall]
    for _, key, _ in bands:
        everything += [
            value for _, value in series(description["by_competitiveness"][key])
        ]
    high = max(everything) * 1.12

    axes = charts.Axes(620, 400, (0, max(powers)), (0, high))
    axes.frame(
        [power for power in powers if power in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)],
        charts.nice_ticks(0, high, 6),
        xlabel="weight_power",
        ylabel="mean distance from the live-only average, points per play",
        yformat="{:.2f}",
    )
    for label, key, color in bands:
        axes.line(
            series(description["by_competitiveness"][key]),
            color=color,
            width=1.6,
            opacity=0.75,
        )
    axes.line(overall, color=charts.BLUE, width=2.6)

    shipped = report["power"]
    axes.vline(shipped, color=charts.INK, dash="3 3")
    axes.text(
        shipped + 0.08,
        high * 0.95,
        f"shipped, power {shipped:g}",
        size=10,
        color=charts.INK,
        weight="600",
    )
    best = description["best_power"]
    best_value = description["by_power"][str(best)]["mean_abs_error"]
    axes.dots([(best, best_value)], color=charts.BLUE, radius=5, edge=charts.GROUND)
    axes.text(
        best,
        best_value - high * 0.055,
        f"best at {best:g}",
        size=10,
        color=charts.BLUE,
        anchor="middle",
    )
    axes.legend(
        [("all team-games", charts.BLUE)]
        + [(label, color) for label, _, color in bands],
        axes.right - 150,
        axes.top + 90,
    )
    plain = description["by_power"]["0.0"]["mean_abs_error"]
    at_shipped = description["by_power"][str(shipped)]["mean_abs_error"]
    return axes.render(
        f"{report['league'].upper()} the weighting halves the gap, then overshoots",
        f"{description['team_games']:,} team-games with {description['min_live_snaps']}+ "
        f"live snaps. Unweighted sits {plain:.3f} from the live-only average; "
        f"at power {shipped:g} that is {at_shipped:.3f}.",
    )


TRIAL_CHARTS = {
    "weighting-trial": _trial_chart,
    "description": _description_chart,
}


def _power_chart(report: dict) -> str:
    """
    Reliability against the power, with the box score as the line to stay
    above. The decision in one picture.
    """
    split = report.get("split_half", {})
    if not split.get("metrics"):
        return ""
    powers = sorted(
        float(name.split()[1]) for name in split["metrics"] if name.startswith("power ")
    )
    if len(powers) < 4:
        return ""
    curve = [
        (power, split["metrics"][f"power {power:g}"]["reliability"]["r"])
        for power in powers
    ]
    band = [
        (
            power,
            split["metrics"][f"power {power:g}"]["reliability"]["low"],
            split["metrics"][f"power {power:g}"]["reliability"]["high"],
        )
        for power in powers
    ]
    box = split["metrics"]["points/play"]["reliability"]["r"]
    low = min(min(value for _, value, _ in band), box) - 0.02
    high = max(max(value for _, _, value in band), box) + 0.02

    axes = charts.Axes(620, 400, (0, max(powers)), (low, high))
    axes.frame(
        [power for power in powers if power in (0.0, 0.5, 1.0, 1.5, 2.0)],
        charts.nice_ticks(low, high, 6),
        xlabel="weight_power",
        ylabel="split-half reliability",
        yformat="{:.2f}",
    )
    axes.band(band, charts.BLUE)
    axes.line(curve, color=charts.BLUE, width=2.4)
    axes.hline(box, color=charts.ORANGE, dash="5 4")
    axes.text(
        max(powers) * 0.98,
        box + (high - low) * 0.018,
        "points per play — the box score",
        size=10,
        color=charts.ORANGE,
        anchor="end",
        weight="600",
    )
    best = max(curve, key=lambda pair: pair[1])
    axes.dots([best], color=charts.BLUE, radius=5, edge=charts.GROUND)
    axes.text(
        best[0],
        best[1] + (high - low) * 0.03,
        f"best at {best[0]:g}",
        size=10,
        color=charts.BLUE,
        anchor="middle",
    )
    shipped = report["weight_power"]
    axes.vline(shipped, color=charts.INK, dash="3 3")
    axes.text(
        shipped - 0.05,
        high - (high - low) * 0.06,
        f"weighted number ships at {shipped:g}",
        size=10,
        color=charts.INK,
        anchor="end",
        weight="600",
    )
    return axes.render(
        f"{report['league'].upper()} where the weighting stops paying",
        f"{split['team_seasons']} team-seasons; band is the 95% bootstrap "
        "interval. The flat number is the value at 0, and it is the one the "
        "package reports alongside the weighted one.",
    )


CHARTS = {
    "power": _power_chart,
    "calibration": _calibration_chart,
    "surface": _surface_chart,
    "distribution": _distribution_chart,
    "weighting": _weighting_chart,
    "split-half": _split_half_chart,
}


def _draw(report: dict, directory: Path, which: dict = CHARTS) -> None:
    """Every chart the report supports, into `directory`."""
    for name, draw in which.items():
        markup = draw(report)
        if not markup:
            continue
        path = directory / f"{name}-{report['league']}.svg"
        path.write_text(markup)
        logger.info("Wrote %s", path)


def redraw(league: str = "nfl", out: str = "docs") -> None:
    """
    Rewrite the charts from a report this already produced.

    The measurement is minutes of fitting and the charts are milliseconds of
    arithmetic, so adjusting a label shouldn't mean refitting two models. The
    JSON is the source of truth for both.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    directory = Path(out)
    payload = directory / f"validation-{league}.json"
    if not payload.is_file():
        raise FileNotFoundError(
            f"No {payload} to redraw from. Run `make validate "
            f"ARGS='--league {league} --root ./plays'` first."
        )
    _draw(json.loads(payload.read_text()), directory)
    # The weighting trial is a separate, slower command, so its report may or
    # may not be there; redraw it when it is rather than making the caller
    # remember which files exist.
    trial = directory / f"weighting-{league}.json"
    if trial.is_file():
        _draw(json.loads(trial.read_text()), directory, TRIAL_CHARTS)


def weighting(
    league: str = "nfl",
    root: str | None = None,
    out: str = "docs",
    fit_seasons: str = "2014-2022",
    test_seasons: str = "2023-2025",
    power: float = DEFAULT_WEIGHT_POWER,
    min_games: int = 10,
    min_live_share: float = 0.5,
    min_live_snaps: int = 20,
    draws: int = 400,
    seed: int = 0,
) -> None:
    """
    Settle the weighting, which `validate` deliberately does not try to.

        make weighting ARGS="--league ncaafb --root ./plays"

    `validate`'s sweep compares a weighted number against an unweighted one
    over different pools of games, because down-weighting a blowout is nearly
    the same as dropping it. This holds the pool still two different ways --
    see `_weighting_trial`. Writes `weighting-{league}.json` next to the rest.

    Fewer draws than `validate` by default: every draw re-thins every
    team-half, so this is the expensive one.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from train import _parse_seasons

    fit_years = _parse_seasons(fit_seasons)
    test_years = _parse_seasons(test_seasons)
    if set(fit_years) & set(test_years):
        raise ValueError("The fit and test seasons overlap; they must not.")

    source = _source(root)
    fit_games = _prepared(source, league, fit_years)
    test_games = _prepared(source, league, test_years)
    logger.info("Fitting on %d games, testing on %d", len(fit_games), len(test_games))
    points, win = _fit_models(fit_games)

    description = _description_trial(
        points,
        win,
        test_games,
        clip=DEFAULT_CLIP,
        powers=DESCRIPTION_POWERS,
        min_live_snaps=min_live_snaps,
    )
    trial = _weighting_trial(
        points,
        win,
        test_games,
        power=power,
        min_games=min_games,
        min_live_share=min_live_share,
        draws=draws,
        seed=seed,
    )
    report = {
        "league": league,
        "fit_seasons": fit_years,
        "test_seasons": test_years,
        "description": description,
        **trial,
    }
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory / f"weighting-{league}.json"
    payload.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Wrote %s", payload)
    _draw(report, directory, TRIAL_CHARTS)

    if "metrics" not in trial:
        logger.info("Not enough team-seasons to say anything.")
        return
    logger.info(
        "\n%s at power %g -- %d team-seasons, %d deals",
        league,
        power,
        trial["team_seasons"],
        trial["draws"],
    )
    logger.info(
        "  weighting leaves %.1f%% of the effective sample",
        100 * trial["effective_sample_share"],
    )
    logger.info("  %-18s %10s %12s %12s", "", "reliability", "scoring", "live")
    for name in ("unadjusted", "thinned", "weighted"):
        row = trial["metrics"][name]
        logger.info(
            "  %-18s %10.3f %12.3f %12.3f",
            name,
            row["reliability"]["r"],
            row["all_scoring"]["r"],
            row["live_scoring"]["r"],
        )
    for label, key in (
        ("weighted - thinned  ", "weighted_vs_thinned"),
        ("thinned - unadjusted", "thinned_vs_unadjusted"),
    ):
        row = trial[key]
        logger.info(
            "  %s %+9.3f %+12.3f %+12.3f   (P %.2f / %.2f / %.2f)",
            label,
            row["reliability"]["difference"],
            row["all_scoring"]["difference"],
            row["live_scoring"]["difference"],
            row["reliability"]["p_better"],
            row["all_scoring"]["p_better"],
            row["live_scoring"]["p_better"],
        )
    if "by_power" in description:
        logger.info(
            "\n  describing one game: how far each power lands from the "
            "live-only mean\n  over %d team-games (median %.0f%% of snaps live)",
            description["team_games"],
            100 * description["live_share"]["median"],
        )
        logger.info(
            "  %8s %14s %11s %14s",
            "power",
            "mean |error|",
            "gap closed",
            "sample kept",
        )
        for power, row in description["by_power"].items():
            mark = "  <-- best" if float(power) == description["best_power"] else ""
            shipped = "  (shipped)" if float(power) == DEFAULT_WEIGHT_POWER else ""
            logger.info(
                "  %8s %14.4f %10.0f%% %13.0f%%%s%s",
                power,
                row["mean_abs_error"],
                100 * row["gap_closed"],
                100 * row["effective_sample_share"],
                shipped,
                mark,
            )
    for label, key in (
        ("same games, vs full ", "live_weighted_vs_live_unadjusted"),
        ("same games, vs thin ", "live_weighted_vs_live_thinned"),
    ):
        row = trial[key]
        logger.info(
            "  %s %+9.3f %+12.3f %+12.3f   (P %.2f / %.2f / %.2f)",
            label,
            row["reliability"]["difference"],
            row["all_scoring"]["difference"],
            row["live_scoring"]["difference"],
            row["reliability"]["p_better"],
            row["all_scoring"]["p_better"],
            row["live_scoring"]["p_better"],
        )


def validate(
    league: str = "nfl",
    root: str | None = None,
    out: str = "docs",
    fit_seasons: str = "2014-2022",
    test_seasons: str = "2023-2025",
    min_games: int = 10,
    min_live_share: float = 0.5,
    draws: int = 1500,
    seed: int = 0,
) -> None:
    """
    Refit on the early seasons, measure on the later ones, write the report.

    Writes `{out}/validation-{league}.json` and the charts beside it. See the
    module docstring for what the four sections are and why the split-half
    one comes with a caveat attached.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from train import _parse_seasons

    fit_years = _parse_seasons(fit_seasons)
    test_years = _parse_seasons(test_seasons)
    overlap = set(fit_years) & set(test_years)
    if overlap:
        raise ValueError(
            f"The fit and test seasons overlap on {sorted(overlap)}. The whole "
            "point of this script is that they don't."
        )

    source = _source(root)
    logger.info("Loading %s", league)
    fit_games = _prepared(source, league, fit_years)
    test_games = _prepared(source, league, test_years)

    logger.info("Fitting on %d games, testing on %d", len(fit_games), len(test_games))
    points, win = _fit_models(fit_games)
    _, fit_labels = _regulation(fit_games)
    test_states, test_labels = _regulation(test_games)

    report = {
        "league": league,
        "fit_seasons": fit_years,
        "test_seasons": test_years,
        "clip": DEFAULT_CLIP,
        "weight_power": DEFAULT_WEIGHT_POWER,
        "n_fit_games": len(fit_games),
        "n_test_games": len(test_games),
        "skill": _skill(points, test_states, test_labels, fit_labels),
        "calibration": _calibration(points, test_states, test_labels),
        "landmarks": _landmarks(points),
        "surface": _surface(points, test_states, test_labels),
        "distribution": _distribution(points, win, test_games, DEFAULT_CLIP),
        "split_half": _split_half(
            points,
            win,
            test_games,
            min_games=min_games,
            min_live_share=min_live_share,
            draws=draws,
            seed=seed,
        ),
    }

    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory / f"validation-{league}.json"
    payload.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Wrote %s", payload)
    _draw(report, directory)

    skill = report["skill"]
    logger.info(
        "\n%s expected points, fit %s-%s, tested on %s-%s (%d snaps)",
        league,
        fit_years[0],
        fit_years[-1],
        test_years[0],
        test_years[-1],
        skill["n_snaps"],
    )
    logger.info(
        "  log loss %.4f against %.4f for the base rates (%.1f%% skill)",
        skill["log_loss"],
        skill["log_loss_base_rates"],
        100 * skill["log_loss_skill"],
    )
    logger.info(
        "  mean absolute error %.3f against %.3f (%.1f%% better)",
        skill["mean_absolute_error"],
        skill["mean_absolute_error_base_rates"],
        100 * skill["mean_absolute_error_skill"],
    )
    logger.info(
        "  bias %+.3f, worst calibration bin off by %.2f points",
        skill["bias"],
        report["calibration"]["worst_gap"],
    )
    split = report["split_half"]
    if split.get("metrics"):
        logger.info("  split-half over %d team-seasons:", split["team_seasons"])
        for key in ("shipped", "unadjusted", "points/play"):
            row = split["metrics"][key]
            logger.info(
                "    %-22s reliability %.3f  scoring %.3f  live %.3f",
                key,
                row["reliability"]["r"],
                row["all_scoring"]["r"],
                row["live_scoring"]["r"],
            )


if __name__ == "__main__":
    fire.Fire({"validate": validate, "weighting": weighting, "redraw": redraw})
