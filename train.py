"""
Fit the models and write them out as releases, and look at what they say.

    make train ARGS="--league nfl --seasons 2022-2025"
    make train-ep ARGS="--league nfl --seasons 2022-2025"

Two fits per league and two files, because they answer different questions:
win probability (`train`) is about the game, expected points (`train_ep`) is
about the snap. `lucky_ones.epa` is the one consumer that needs both.

Two ways to get the plays, and the flag picks between them:

    --root PATH   a local copy of endgame's processed tree, e.g. after
                  `aws s3 sync s3://BUCKET/processed/plays ./plays`. No
                  credentials, reproducible, and fast enough to re-fit while
                  you're changing features.
    (default)     endgame's `ProcessedPlaysStore`, straight out of the
                  bucket. Needs AWS_PROFILE (or a role) and endgame_aws's
                  config -- see its `Config.init_from_file`.

The two are the same `PlaySource` to everything downstream, which is the
point of the protocol: this script is the only place that decides where
plays come from.

The output is a `WinProbabilityRelease` -- coefficients, what they were fit
on, and how they scored on held-out games -- written into the package at
`lucky_ones/releases/{league}.json`, which is the file `MODELS.NFL` serves,
and an `ExpectedPointsRelease` next to it at `{league}-ep.json`. Commit them:
the fits ship with the code, so a retrain is a diff of some coefficients and a
holdout score. See README.md for the consumer end.
"""

import asyncio
import getpass
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import fire
import numpy as np

from lucky_ones import MODELS, GamePlays, group_by_game

# `lucky_ones` exports the six names a consumer needs to score a game. This
# script fits the models behind them, which is the other half of the package,
# so it reaches a layer down for nearly everything -- exactly the split the
# top-level module is drawing.
from lucky_ones.arrow import DatasetPlaySource, StorePlaySource
from lucky_ones.bundled import EP_SUFFIX, RELEASE_DIR
from lucky_ones.curve import game_control
from lucky_ones.epa import (
    DEFAULT_CLIP,
    DEFAULT_WEIGHT_POWER,
    epa_per_play,
    play_epa,
)
from lucky_ones.luck import (
    DEFAULT_RETAINED,
    DEFENDED_MARKERS,
    LuckKind,
    adjusted_curve_from_states,
    find_lucky_plays,
    is_defended_pass,
    lucky_wp_from_states,
    records_defended_passes,
)
from lucky_ones.metrics import (
    brier_score,
    log_loss,
    mean_absolute_error,
    multiclass_log_loss,
)
from lucky_ones.model import LogisticWinProbability, WinProbabilityModel
from lucky_ones.plays import Play, PlaySource
from lucky_ones.points import (
    FIRST_LEGIBLE_SEASON,
    SCORE_VALUES,
    ExpectedPointsModel,
    MultinomialExpectedPoints,
    scoring_plays,
)
from lucky_ones.release import (
    ExpectedPointsMetrics,
    ExpectedPointsRelease,
    Metrics,
    TrainedOn,
    WinProbabilityRelease,
)
from lucky_ones.state import (
    PERIOD_SECONDS,
    REGULATION_SECONDS,
    GameState,
    iter_states,
)
from lucky_ones.training import (
    build_expected_points_set,
    build_training_set,
    split_games,
)

logger = logging.getLogger("train")

# What a season's weeks are, as a range to ask the store for. Weeks that
# haven't been processed come back empty, so asking for too many is free and
# asking for too few silently trains on less than you meant to. NCAAFB runs
# longest -- week 0 through the championship -- so this covers both leagues.
DEFAULT_WEEKS = range(0, 21)


def _parse_seasons(seasons: str | int | Sequence[int]) -> list[int]:
    """
    `2025`, `2022-2025` or `2022,2024` -> a list of years.

    Ranges are inclusive at both ends, because "2022-2025" meaning three
    seasons is the kind of surprise that shows up as a metrics change nobody
    can explain.
    """
    if isinstance(seasons, int):
        return [seasons]
    if not isinstance(seasons, str):
        return [int(season) for season in seasons]
    years: list[int] = []
    for part in seasons.split(","):
        part = part.strip()
        if "-" in part:
            start, _, end = part.partition("-")
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def _parse_weeks(weeks: str | int | None) -> list[int]:
    if weeks is None:
        return list(DEFAULT_WEEKS)
    return _parse_seasons(weeks)


async def _load(
    source: PlaySource, league: str, seasons: Iterable[int], weeks: Sequence[int]
) -> list[Play]:
    plays: list[Play] = []
    for season in seasons:
        season_plays = await source.load_weeks(league, season, weeks)
        logger.info("%s %s: %d plays", league, season, len(season_plays))
        plays.extend(season_plays)
    return plays


def _source(root: str | None) -> PlaySource:
    """
    The local tree if one was named, otherwise the bucket.

    `endgame_aws` is imported here rather than at module scope so that
    `--root` works in an environment that has no AWS anything -- including
    CI, where the tests run this path.

    The store reads its bucket from endgame_aws's own config, which means
    `~/.aws-batch/config.json` has to be there (the devcontainer mounts it).
    A missing one raises FileNotFoundError from inside endgame_aws.
    """
    if root is not None:
        return DatasetPlaySource(root)
    from endgame_aws.pbp_parquet import get_processed_plays_store

    return StorePlaySource(get_processed_plays_store())


def _score(model: LogisticWinProbability, games: Sequence[GamePlays]) -> Metrics:
    """Brier and log loss over every snap of games the fit never saw."""
    holdout = build_training_set(games)
    if not holdout.states:
        raise ValueError("The holdout has no snaps in it; use more seasons")
    predicted = model.predict(holdout.states)
    outcomes = np.asarray(holdout.home_won, dtype=float)
    return Metrics(
        brier_score=brier_score(predicted, outcomes),
        log_loss=log_loss(predicted, outcomes),
        n_games=len({state.game_id for state in holdout.states}),
        n_snaps=holdout.rows,
    )


def train(
    league: str = "nfl",
    seasons: str = "2025",
    weeks: str | None = None,
    root: str | None = None,
    out: str | None = None,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> None:
    """
    Fit a model and write the release.

    Defaults to `lucky_ones/releases/{league}.json` -- the fit `MODELS.NFL`
    serves, inside the package. So a retrain is `make train` and then a commit
    of the JSON it rewrote: the diff is the coefficients and the holdout
    score, which is the review you want on a model change. Pass `--out` to
    write somewhere else without touching the shipped one.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    years, week_numbers = _parse_seasons(seasons), _parse_weeks(weeks)
    plays = asyncio.run(_load(_source(root), league, years, week_numbers))
    games = group_by_game(plays)
    if not games:
        raise ValueError(
            f"No {league} games in {seasons}. With --root, check the path points "
            "at the `processed/plays` directory; without it, check AWS_PROFILE."
        )

    fit_games, holdout_games = split_games(
        games, holdout_fraction=holdout_fraction, seed=seed
    )
    training = build_training_set(fit_games)
    logger.info(
        "Fitting on %d games (%d snaps), holding out %d games",
        len(fit_games),
        training.rows,
        len(holdout_games),
    )
    model = LogisticWinProbability.fit(training.states, training.home_won)
    metrics = _score(model, holdout_games)

    created_at = datetime.now(timezone.utc)
    release = WinProbabilityRelease.from_model(
        model,
        run_id=created_at.strftime("%Y%m%d-%H%M%S"),
        league=league,
        trained_on=TrainedOn(
            league=league,
            seasons=years,
            weeks=week_numbers,
            n_games=len(fit_games),
            n_snaps=training.rows,
        ),
        metrics=metrics,
        created_at=created_at,
        created_by=_whoami(),
    )

    destination = Path(out) if out else RELEASE_DIR / f"{league}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(release.model_dump_json(indent=2) + "\n")

    for name, coefficient in zip(model.feature_names, model.coefficients):
        logger.info("  %-22s %+.4f", name, coefficient)
    logger.info("  %-22s %+.4f", "(intercept)", model.intercept)
    logger.info(
        "Holdout: brier %.4f, log loss %.4f over %d games",
        metrics.brier_score,
        metrics.log_loss,
        metrics.n_games,
    )
    logger.info("Wrote %s", destination)


# The snaps to print an expected points readout for after a fit. Not a test --
# an eyeball check, in the same spirit as printing the win probability
# coefficients: these four are situations whose value everyone already knows,
# so a fit that gets them wrong is wrong in a way you can see from here.
_LANDMARKS: tuple[tuple[str, dict], ...] = (
    ("1st and 10, own 25", dict(down=1, distance=10, yardline=25)),
    ("1st and 10, midfield", dict(down=1, distance=10, yardline=50)),
    ("1st and goal, the 2", dict(down=1, distance=2, yardline=98)),
    ("3rd and 15, own 5", dict(down=3, distance=15, yardline=5)),
)


def _landmark_state(**situation) -> GameState:
    """One of `_LANDMARKS` as a state, early in the first quarter."""
    return GameState(
        game_id="landmark",
        play_id="landmark",
        play_number=1,
        period=1,
        clock_seconds=PERIOD_SECONDS,
        seconds_remaining=REGULATION_SECONDS,
        is_overtime=False,
        home_score=0,
        away_score=0,
        offense_is_home=True,
        **situation,
    )


def _score_ep(
    model: MultinomialExpectedPoints, games: Sequence[GamePlays]
) -> ExpectedPointsMetrics:
    """
    Log loss over the seven classes and mean absolute error in points, over
    every snap of games the fit never saw.

    A holdout class the fit never saw gets a column of zeros rather than
    being dropped: it is a genuine miss and log loss should charge for it,
    and dropping it would flatter a fit that ran short of one class exactly
    when it mattered. Only reachable on a tiny fit -- a real season has all
    seven.
    """
    holdout = build_expected_points_set(games)
    if not holdout.states:
        raise ValueError("The holdout has no snaps in it; use more seasons")
    index_of = {kind: index for index, kind in enumerate(model.kinds)}
    unseen = len(index_of)
    probabilities = model.predict_proba(holdout.states)
    if any(kind not in index_of for kind in holdout.next_score):
        probabilities = np.hstack([probabilities, np.zeros((len(probabilities), 1))])
    actual = np.array(
        [index_of.get(kind, unseen) for kind in holdout.next_score], dtype=int
    )
    return ExpectedPointsMetrics(
        log_loss=multiclass_log_loss(probabilities, actual),
        mean_absolute_error=mean_absolute_error(
            model.predict(holdout.states),
            np.array([SCORE_VALUES[kind] for kind in holdout.next_score]),
        ),
        n_games=len({state.game_id for state in holdout.states}),
        n_snaps=holdout.rows,
    )


def train_ep(
    league: str = "nfl",
    seasons: str = "2025",
    weeks: str | None = None,
    root: str | None = None,
    out: str | None = None,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> None:
    """
    Fit an expected points model and write the release.

    The same shape as `train`, one directory over: it writes
    `lucky_ones/releases/{league}-ep.json`, which is what
    `MODELS.NFL.expected_points` serves and half of what
    `MODELS.NFL.epa_per_play` needs. Commit it for the same reason.

    Held out by game, and that matters more here than it does for `train`:
    every snap of a drive usually carries the same label -- the same next
    score -- so a row-wise split would be scoring the fit on drives it had
    already been told the answer to.

    The seasons to fit it on are not the seasons `train` uses, and the
    difference is in the data rather than in the modelling. A win probability
    label needs only the final score, which every season records correctly.
    This one needs to know *when* the points went up, and before 2014 both
    leagues' score columns jitter badly enough that a tenth of the scoring
    can't be placed -- see `lucky_ones.points.score_events`. The shipped fits
    start at 2014 for that reason, and the class shares printed below are
    where a season that stopped saying would show up.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    years, week_numbers = _parse_seasons(seasons), _parse_weeks(weeks)
    plays = asyncio.run(_load(_source(root), league, years, week_numbers))
    games = group_by_game(plays)
    if not games:
        raise ValueError(
            f"No {league} games in {seasons}. With --root, check the path points "
            "at the `processed/plays` directory; without it, check AWS_PROFILE."
        )

    fit_games, holdout_games = split_games(
        games, holdout_fraction=holdout_fraction, seed=seed
    )
    illegible = [season for season in years if season < FIRST_LEGIBLE_SEASON]
    if illegible:
        # Not fatal -- somebody measuring the damage wants to be able to ask
        # for these -- but silently fitting on them is how a worse model gets
        # shipped without a line in the diff saying so.
        logger.warning(
            "%s: the score columns can't place about one score in ten before "
            "%d, so a tenth of these labels land on the wrong drive. See "
            "lucky_ones.points.score_events; the shipped fits start at %d.",
            ", ".join(str(season) for season in illegible),
            FIRST_LEGIBLE_SEASON,
            FIRST_LEGIBLE_SEASON,
        )

    training = build_expected_points_set(fit_games)
    logger.info(
        "Fitting on %d games (%d snaps), holding out %d games",
        len(fit_games),
        training.rows,
        len(holdout_games),
    )
    model = MultinomialExpectedPoints.fit(training.states, training.next_score)
    metrics = _score_ep(model, holdout_games)

    created_at = datetime.now(timezone.utc)
    release = ExpectedPointsRelease.from_model(
        model,
        run_id=created_at.strftime("%Y%m%d-%H%M%S"),
        league=league,
        trained_on=TrainedOn(
            league=league,
            seasons=years,
            weeks=week_numbers,
            n_games=len(fit_games),
            n_snaps=training.rows,
        ),
        metrics=metrics,
        created_at=created_at,
        created_by=_whoami(),
    )

    destination = Path(out) if out else RELEASE_DIR / f"{league}{EP_SUFFIX}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(release.model_dump_json(indent=2) + "\n")

    shares = Counter(training.next_score)
    for kind in model.kinds:
        logger.info(
            "  %-22s %5.1f%% of snaps",
            kind.value,
            100.0 * shares[kind] / max(training.rows, 1),
        )
    for label, situation in _LANDMARKS:
        logger.info(
            "  %-22s %+.2f points",
            label,
            model.predict([_landmark_state(**situation)])[0],
        )
    logger.info(
        "Holdout: log loss %.4f, mean absolute error %.2f points over %d games",
        metrics.log_loss,
        metrics.mean_absolute_error,
        metrics.n_games,
    )
    logger.info("Wrote %s", destination)


def curve(
    game_id: str | int,
    league: str = "nfl",
    season: int = 2025,
    week: int = 1,
    model: str = "",
    root: str | None = None,
) -> None:
    """
    Print one game's win probability curve as JSON, and its game control.

    The eyeball check on a fresh release, and the same call a backend makes
    to draw the graph -- so if this looks right, the chart will. Uses the
    bundled `MODELS[league]` by default, which is the fit a consumer would
    get; `--model PATH` reads a release from a file instead, for checking one
    before it's committed.

    Both control numbers come out, plus `lucky_wp` -- how much win probability
    the bounces handed each team -- and the plays behind them: `lucky_plays` is
    every fumble and tipped interception the game turned on, priced both ways,
    and each point carries its luck-adjusted probability alongside the real
    one. See `lucky_ones.luck`.

    `game_id` is annotated `str | int` and coerced because fire converts an
    all-digit argument to an int regardless of the annotation, and every ESPN
    game id is all digits. Left alone it reaches the parquet filter as an int
    and pyarrow refuses to compare it to a string column, so the documented
    invocation fails on every real game.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    game_id = str(game_id)
    fit: WinProbabilityModel = (
        WinProbabilityRelease.model_validate_json(Path(model).read_text()).to_model()
        if model
        else MODELS[league]
    )
    plays = asyncio.run(_source(root).load_game(league, season, week, game_id))
    games = group_by_game(plays)
    if not games:
        raise ValueError(f"No plays for {league} {season} week {week} game {game_id}")
    (game,) = games

    states = list(iter_states(game))
    lucky = find_lucky_plays(game.plays)
    # Both walks price the realized curve again. That is one extra matrix
    # multiply on one game, which is nothing here and is why neither function
    # takes the other's output -- see `lucky_ones.luck`.
    adjusted = adjusted_curve_from_states(fit, states, lucky)
    breaks = lucky_wp_from_states(fit, states, lucky)
    control = game_control(adjusted.realized)
    earned = game_control(adjusted.points)
    print(
        json.dumps(
            {
                "game_id": game.game_id,
                "home_team_id": game.home_team_id,
                "away_team_id": game.away_team_id,
                "game_control": None if control is None else control._asdict(),
                "luck_adjusted_game_control": (
                    None if earned is None else earned._asdict()
                ),
                "lucky_wp": {
                    "home": breaks.home,
                    "away": breaks.away,
                    "net": breaks.net,
                },
                # One list, zipped: both walks go through `_priced_branches`,
                # so they report the same plays in the same order.
                "lucky_plays": [
                    {
                        **lucky._asdict(),
                        "realized": swing.realized,
                        "counterfactual": swing.counterfactual,
                        "expected": swing.expected,
                        "home_delta": swing.home_delta,
                    }
                    for lucky, swing in zip(adjusted.lucky_plays, breaks.swings)
                ],
                "points": [
                    {
                        **point._asdict(),
                        "luck_adjusted_win_probability": (
                            adjusted_point.home_win_probability
                        ),
                    }
                    for point, adjusted_point in zip(adjusted.realized, adjusted.points)
                ],
            },
            indent=2,
        )
    )


# Not a classifier -- nothing in `lucky_ones` reads these. Every phrase that
# could mean a defender got a hand on a pass that stayed incomplete, which
# together are the only denominator on offer for
# `LuckKind.TIPPED_INTERCEPTION`.
#
# Kept as families rather than one flat tuple to answer the question that
# comes up every time the share moves: is the annotation going away, or is it
# being written a different way? `rates` reports each family separately.
# NCAAFB's answer is that `broken_up` is the whole union and every other
# family is flat zero, so its swings are annotation genuinely coming and
# going. The NFL's is that none of these phrases is how it says it -- see
# `_defensed_by_syntax`, which is.
DEFENDED_FAMILIES: dict[str, tuple[str, ...]] = {
    "broken_up": ("broken up", "broke up", "break up"),
    "defensed": ("defensed", "pass defended", "defended by", "pbu"),
    "deflected": ("deflect",),
    "tipped": ("tipped", "tip by"),
    "batted": ("batted", "bat down"),
    "knocked": ("knocked down", "knocked away", "knocked loose"),
    "swatted": ("swatted", "punched away", "slapped away"),
}


def rates(
    league: str = "nfl",
    seasons: str = "2025",
    weeks: str | None = None,
    root: str | None = None,
) -> None:
    """
    Measure the rates behind `DEFAULT_RETAINED`, per season, as JSON.

        make rates ARGS="--league ncaafb --seasons 2006-2025 --root ./plays"

    `retained` is `P(the outcome that happened)`, and the two kinds it covers
    are on very different footing. This prints both so the difference is
    visible rather than argued about.

    **Fumbles: measurable, and this is the measurement.** It counts through
    `find_lucky_plays` rather than re-deriving anything, so `lost` is the rate
    over exactly the population the model adjusts, witnessed the way the
    shipped code witnesses it. `lost` is `retained[FUMBLE_LOST]` and its
    complement is `retained[FUMBLE_KEPT]` -- the two are a pair and have to
    sum to 1, which is the constraint 0.5/0.5 satisfies for free and any
    measured pair has to be held to.

    **Tipped interceptions: not measurable, and `coverage` is why.** The
    numerator would be interceptions off a deflection and the denominator
    every pass a defender got to. Neither is in the data. No interception in
    either league says how the ball got there -- ESPN's parenthetical on an
    NFL interception is the tackler on the return -- and the denominator is
    only ever partly written down: `coverage` is how often a season's text
    marks a defended incompletion at all, and `interception_share` moves
    inversely with it across the whole corpus while the interception rate
    barely moves. A share computed against half-recorded breakups is a
    measurement of the annotation, so read `coverage` first and the share
    second, or not at all.

    `by_family` splits the denominator by what caught it, and the split is
    the interesting part, because the two leagues answer in different
    alphabets. NCAAFB says it in words and unreliably: `broken_up` is its
    entire union, no other family ever fires, and coverage swings by a factor
    of four between seasons. The NFL says it in punctuation and steadily --
    `defensed_syntax`, the gamebook parenthetical on an incompletion, runs
    0.093 to 0.109 of pass attempts every season from 2009 on, which is a
    real pass-defensed rate.

    So the NFL share **is** measurable, at 0.20 combined over 2009-2025 and
    0.177-0.229 by season, drifting only as slowly as the interception rate
    itself -- and it is what `DEFAULT_RETAINED[PASS_DEFENDED_INTERCEPTION]` is
    set from. What is *not* measurable is the tipped share specifically: a
    pass defended is any ball a defender got to, and no interception in either
    league records whether one was deflected on the way. The neighbouring
    question turned out to be the answerable one.

    `coverage` is also what `lucky_ones.luck.records_defended_passes` decides
    on, one game at a time. A season whose coverage is low is a season where
    most games will fail that gate and be adjusted for their fumbles only.
    """
    years = _parse_seasons(seasons)
    plays = asyncio.run(_load(_source(root), league, years, _parse_weeks(weeks)))
    if not plays:
        raise ValueError(f"No {league} plays in {seasons}")

    fumbles: dict[int, Counter] = defaultdict(Counter)
    passes: dict[int, Counter] = defaultdict(Counter)
    for game in group_by_game(plays):
        for lucky in find_lucky_plays(game.plays):
            fumbles[game.season][lucky.kind.value] += 1
        # Every count below is kept twice: once over the whole season, which
        # is the feed profile, and once over only the games whose feed records
        # defended passes, which is the measurement. Mixing them is the error
        # this is split to avoid -- a game that records none of its breakups
        # still records all of its interceptions, so it would put a numerator
        # into the share and nothing into the denominator. In NCAAFB 2023 that
        # is four games in five.
        recording = records_defended_passes(game.plays)
        passes[game.season]["games"] += 1
        passes[game.season]["games_recording"] += recording
        for play in game.plays:
            text = (play.text or "").lower()
            if not text or "no play" in text:
                continue
            intercepted = "intercept" in text
            kind = (play.play_type or "").lower()
            # Sacks carry "pass" in the text often enough to matter and are
            # not attempts; an interception is one however it is typed.
            if not (intercepted or ("pass" in kind and "sack" not in kind)):
                continue
            passes[game.season]["attempts"] += 1
            if recording:
                passes[game.season]["gated_attempts"] += 1
            if intercepted:
                passes[game.season]["interceptions"] += 1
                if recording:
                    passes[game.season]["gated_interceptions"] += 1
                continue
            if "incomplete" not in text:
                continue
            for family, markers in DEFENDED_FAMILIES.items():
                if any(marker in text for marker in markers):
                    passes[game.season][f"family:{family}"] += 1
            if not any(marker in text for marker in DEFENDED_MARKERS):
                # Whatever caught it wasn't a word, so it was the punctuation.
                if is_defended_pass(play.text):
                    passes[game.season]["family:defensed_syntax"] += 1
            # The total comes from the classifier's own predicate rather than
            # from these families, so a season's reported coverage is the
            # coverage `records_defended_passes` will see. The family keys are
            # prefixed because one of them is `broken_up` too, and a total
            # sharing a key with a family counts twice.
            if is_defended_pass(play.text):
                passes[game.season]["defended"] += 1
                if recording:
                    passes[game.season]["gated_defended"] += 1

    print(
        json.dumps(
            {
                "league": league,
                "seasons": years,
                "retained_in_use": {
                    kind.value: weight for kind, weight in DEFAULT_RETAINED.items()
                },
                "fumbles": _fumble_rates(fumbles),
                "defended_passes": _defended_rates(passes),
            },
            indent=2,
        )
    )


def _fumble_rates(per_season: dict[int, Counter]) -> dict:
    """`retained[FUMBLE_LOST]` and its complement, overall and by season."""

    def summarise(counts: Counter) -> dict:
        lost = counts[LuckKind.FUMBLE_LOST.value]
        classified = lost + counts[LuckKind.FUMBLE_KEPT.value]
        return {
            "classified": classified,
            "lost": round(lost / classified, 4) if classified else None,
            "kept": round(1 - lost / classified, 4) if classified else None,
        }

    total = Counter()
    for counts in per_season.values():
        total.update(counts)
    return {
        **summarise(total),
        "by_season": {
            str(season): summarise(per_season[season]) for season in sorted(per_season)
        },
    }


def _defended_rates(per_season: dict[int, Counter]) -> dict:
    """
        Interceptions over the passes a defender is recorded as having reached.

    Two sets of counts, and the difference between them is the point.

        The bare ones -- `attempts`, `interceptions`, `defended`, `coverage`,
        `by_family` -- run over every game and describe **the feed**: how much of
        the denominator a season's text writes down, and in which words. Read
        `coverage` to know how much of a season is legible at all.

        The `gated_` ones and `interception_share` run only over the games where
        `records_defended_passes` says both sides are recorded, and describe **the
        football**. That split is the whole correctness of the number: a game that
        records none of its breakups still records all of its interceptions, so
        leaving it in puts a numerator into the share with nothing underneath it.
        In NCAAFB 2023 that is four games in five, which is most of why the
        ungated ratio there reads 0.53 against a gated 0.19.

        `by_family` splits the feed-wide total by the phrase or the punctuation
        that caught it, and omits the families that caught nothing. A play
        matching two families counts once in `defended` and once in each, so
        `by_family` sums to at least the total rather than exactly it.
    """

    def summarise(counts: Counter) -> dict:
        attempts = counts["attempts"]
        gated_attempts = counts["gated_attempts"]
        gated_interceptions = counts["gated_interceptions"]
        gated_defended = counts["gated_defended"]
        gated_total = gated_interceptions + gated_defended
        return {
            "games": counts["games"],
            "games_recording": counts["games_recording"],
            "attempts": attempts,
            "interceptions": counts["interceptions"],
            "defended": counts["defended"],
            "coverage": round(counts["defended"] / attempts, 4) if attempts else None,
            "by_family": {
                family: counts[f"family:{family}"]
                for family in (*DEFENDED_FAMILIES, "defensed_syntax")
                if counts[f"family:{family}"]
            },
            # Only over the games that record both sides. This is the one to
            # read -- the counts above describe the feed, this describes the
            # football.
            "gated_attempts": gated_attempts,
            "gated_interceptions": gated_interceptions,
            "gated_defended": gated_defended,
            "gated_coverage": (
                round(gated_defended / gated_attempts, 4) if gated_attempts else None
            ),
            "interception_share": (
                round(gated_interceptions / gated_total, 4) if gated_total else None
            ),
        }

    total = Counter()
    for counts in per_season.values():
        total.update(counts)
    return {
        **summarise(total),
        "by_season": {
            str(season): summarise(per_season[season]) for season in sorted(per_season)
        },
    }


def _whoami() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - depends on the environment
        # No passwd entry, which is normal in a container running as a bare
        # uid. A release with an unknown author is better than a failed run.
        return "unknown"


def _fits(
    league: str, model: str, ep_model: str
) -> tuple[WinProbabilityModel, ExpectedPointsModel]:
    """
    The pair of fits an EPA needs: the bundled ones for `league` unless a
    path was given for either.

    Two flags rather than one because the two fits are retrained separately,
    and checking a fresh expected points model against the shipped win
    probability one is the normal thing to want.
    """
    win = (
        WinProbabilityRelease.model_validate_json(Path(model).read_text()).to_model()
        if model
        else MODELS[league]
    )
    points = (
        ExpectedPointsRelease.model_validate_json(Path(ep_model).read_text()).to_model()
        if ep_model
        else MODELS[league].expected_points
    )
    return win, points


def epa(
    game_id: str | int,
    league: str = "nfl",
    season: int = 2025,
    week: int = 1,
    model: str = "",
    ep_model: str = "",
    root: str | None = None,
    clip: float = DEFAULT_CLIP,
    weight_power: float = DEFAULT_WEIGHT_POWER,
) -> None:
    """
    Print one game's EPA per play as JSON, and every snap behind it.

    The eyeball check on an expected points release, and the same call a
    backend makes: `make epa ARGS="401671789 --week 3"`. Both offenses' number
    comes out, with the effective sample behind each -- and `plays` carries
    every snap with what it was worth, what the bound did to it and how much
    the game was still in doubt when it happened, so "which plays is this
    number made of" is a sort rather than a second run.

    `--clip inf --weight_power 0` prints the plain unweighted mean of raw EPA,
    which is the number to compare against when you want to see what the two
    adjustments are actually doing.

    `game_id` is coerced for the reason `curve`'s is: fire makes an int of an
    all-digit argument, and pyarrow won't compare one to a string column.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    game_id = str(game_id)
    win, points = _fits(league, model, ep_model)
    plays = asyncio.run(_source(root).load_game(league, season, week, game_id))
    games = group_by_game(plays)
    if not games:
        raise ValueError(f"No plays for {league} {season} week {week} game {game_id}")
    (game,) = games

    result = epa_per_play(points, win, game, clip=clip, weight_power=weight_power)
    print(
        json.dumps(
            {
                "game_id": game.game_id,
                "home_team_id": game.home_team_id,
                "away_team_id": game.away_team_id,
                "clip": clip,
                "weight_power": weight_power,
                "epa_per_play": {
                    "home": result.home,
                    "away": result.away,
                    "net": result.net,
                    "home_plays": result.home_plays,
                    "away_plays": result.away_plays,
                    "home_weight": round(result.home_weight, 3),
                    "away_weight": round(result.away_weight, 3),
                },
                "plays": [play._asdict() for play in result.plays],
            },
            indent=2,
        )
    )


# The quantiles `bounds` reports. Weighted towards the tails on purpose: the
# middle of an EPA distribution is not in question and the bound is entirely
# a decision about the ends, so the interesting rows are p1/p99 (where
# DEFAULT_CLIP sits) and p0.1/p99.9 (the plays it is bounding).
_QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)

# And of |EPA|, which is what a symmetric bound is actually a quantile of.
_ABS_QUANTILES = (0.9, 0.95, 0.975, 0.99, 0.995, 0.999)


def bounds(
    league: str = "nfl",
    seasons: str = "2025",
    weeks: str | None = None,
    root: str | None = None,
    model: str = "",
    ep_model: str = "",
    clip: float = DEFAULT_CLIP,
    weight_power: float = DEFAULT_WEIGHT_POWER,
) -> None:
    """
    Measure the distribution behind `DEFAULT_CLIP` and `DEFAULT_WEIGHT_POWER`,
    per season, as JSON.

        make bounds ARGS="--league nfl --seasons 2009-2025 --root ./plays"

    What `rates` is to `DEFAULT_RETAINED`, this is to the two knobs in
    `lucky_ones.epa`, and it runs through `play_epa` for the same reason: the
    numbers come off exactly the code path that averages them, so this is the
    distribution the metric sees rather than a second opinion about it.

    Three things per season, and they answer three different questions.

    **`quantiles` and `beyond_clip`: where is the tail, and how much of it is
    the bound touching?** A clip belongs at the point where a play stops being
    football and starts being an accident of the sample, which is somewhere
    past p99 -- and `beyond_clip` says what fraction of snaps that turns out
    to be. If it is more than a percent or two the bound is doing more than
    bounding.

    **`weight`: how much of a season survives the weighting?**
    `effective_play_share` is the mean play weight, so 0.6 says that after
    down-weighting the decided stretches, a season's snaps are worth about
    three fifths of their count. Read it as what the number is averaged over.

    **`game_shift`: does any of this change the answer?** The per-game,
    per-team gap between the number this module reports and the plain
    unweighted mean of raw EPA. If that were near zero the two knobs would be
    ceremony; it is not, and its size is the honest statement of how much of
    this metric is a modelling choice.
    """
    years = _parse_seasons(seasons)
    win, points = _fits(league, model, ep_model)
    plays = asyncio.run(_load(_source(root), league, years, _parse_weeks(weeks)))
    if not plays:
        raise ValueError(f"No {league} plays in {seasons}")

    per_season: dict[int, list[np.ndarray]] = defaultdict(list)
    shifts: dict[int, list[float]] = defaultdict(list)
    for game in group_by_game(plays):
        scored = scoring_plays(list(game.plays))
        priced = play_epa(
            points,
            win,
            list(iter_states(game)),
            scored,
            clip=clip,
            weight_power=weight_power,
        )
        if not priced:
            continue
        per_season[game.season].append(
            np.array([[play.epa, play.bounded, play.weight] for play in priced])
        )
        for home in (True, False):
            side = [play for play in priced if play.offense_is_home == home]
            weight = sum(play.weight for play in side)
            if not side or weight <= 0.0:
                continue
            adjusted = sum(play.weight * play.bounded for play in side) / weight
            raw = sum(play.epa for play in side) / len(side)
            shifts[game.season].append(abs(adjusted - raw))

    print(
        json.dumps(
            {
                "league": league,
                "seasons": years,
                "clip_in_use": clip,
                "weight_power_in_use": weight_power,
                "by_season": {
                    str(season): _bound_summary(
                        np.vstack(per_season[season]), shifts[season], clip
                    )
                    for season in sorted(per_season)
                },
            },
            indent=2,
        )
    )


def _bound_summary(priced: np.ndarray, shifts: Sequence[float], clip: float) -> dict:
    """One season's rows of `[epa, bounded, weight]`, summarised."""
    raw, weight = priced[:, 0], priced[:, 2]
    return {
        "plays": len(raw),
        "mean_abs_epa": round(float(np.mean(np.abs(raw))), 4),
        "quantiles": {
            f"p{quantile * 100:g}": round(float(value), 3)
            for quantile, value in zip(_QUANTILES, np.quantile(raw, _QUANTILES))
        },
        # The signed distribution above is lopsided -- the downside tail is
        # turnovers and the upside is only touchdowns -- so a symmetric bound
        # is a statement about this one, not about that one.
        "abs_quantiles": {
            f"p{quantile * 100:g}": round(float(value), 3)
            for quantile, value in zip(
                _ABS_QUANTILES, np.quantile(np.abs(raw), _ABS_QUANTILES)
            )
        },
        "beyond_clip": round(float(np.mean(np.abs(raw) > clip)), 4),
        "effective_play_share": round(float(np.mean(weight)), 4),
        "game_shift": {
            "team_games": len(shifts),
            "median_abs": (round(float(np.median(shifts)), 4) if len(shifts) else None),
            "p95_abs": (
                round(float(np.quantile(shifts, 0.95)), 4) if len(shifts) else None
            ),
        },
    }


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "train_ep": train_ep,
            "curve": curve,
            "epa": epa,
            "rates": rates,
            "bounds": bounds,
        }
    )
