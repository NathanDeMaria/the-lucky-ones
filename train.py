"""
Fit a win probability model and write it out as a release.

    make train ARGS="--league nfl --seasons 2022-2025"

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
`lucky_ones/releases/{league}.json`, which is the file `MODELS.NFL` serves.
Commit it: the fit ships with the code, so a retrain is a diff of eight
coefficients and a holdout score. See README.md for the consumer end.
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

# `lucky_ones` exports the five names a consumer needs to score a game. This
# script fits one, which is the other half of the package, so it reaches a
# layer down for nearly everything -- exactly the split the top-level module
# is drawing.
from lucky_ones.arrow import DatasetPlaySource, StorePlaySource
from lucky_ones.bundled import RELEASE_DIR
from lucky_ones.curve import game_control
from lucky_ones.luck import (
    DEFAULT_RETAINED,
    DEFENDED_MARKERS,
    LuckKind,
    adjusted_curve_from_states,
    find_lucky_plays,
    is_defended_pass,
    lucky_wp_from_states,
)
from lucky_ones.metrics import brier_score, log_loss
from lucky_ones.model import LogisticWinProbability, WinProbabilityModel
from lucky_ones.plays import Play, PlaySource
from lucky_ones.release import Metrics, TrainedOn, WinProbabilityRelease
from lucky_ones.state import iter_states
from lucky_ones.training import build_training_set, split_games

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


# What a complete feed records: defended passes per pass attempt. Two feeds
# that share no code agree on it -- the NFL's gamebook parenthetical runs
# 0.093-0.109 every season from 2009, and NCAAFB's "broken up by" hits 0.107
# in 2026, the season it switched to a gamebook-shaped format. Every NCAAFB
# season before that is below it, and `implied_share` is what the season would
# have said had it not been.
REFERENCE_COVERAGE = 0.10


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
            if intercepted:
                passes[game.season]["interceptions"] += 1
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

    `coverage` is `defended / attempts` -- how much of the denominator the
    season's text writes down at all, and the number to read first, because
    `interception_share` is only ever as good as it.

    `implied_share` is the same share with the denominator taken at
    `REFERENCE_COVERAGE` instead of the season's own: what the season would
    have said had its feed recorded every defended pass. It is the one that
    can be compared across seasons and across leagues, and doing so is what
    turns this from a shrug into a measurement -- every NCAAFB season from
    2006 on lands between 0.19 and 0.22 once corrected, which is the NFL's
    own range, from a feed that shares no code with it. A season whose
    `coverage` is already at or above the reference is its own best evidence
    and doesn't need the correction; read `interception_share` there.

    `by_family` splits the total by the phrase or the punctuation that caught
    it, and omits the families that caught nothing. A play matching two
    families counts once in `defended` and once in each, so `by_family` sums
    to at least the total rather than exactly it.
    """

    def summarise(counts: Counter) -> dict:
        attempts = counts["attempts"]
        interceptions = counts["interceptions"]
        defended_passes = counts["defended"]
        defended = interceptions + defended_passes
        implied = interceptions + REFERENCE_COVERAGE * attempts
        return {
            "attempts": attempts,
            "interceptions": interceptions,
            "defended": defended_passes,
            "by_family": {
                family: counts[f"family:{family}"]
                for family in (*DEFENDED_FAMILIES, "defensed_syntax")
                if counts[f"family:{family}"]
            },
            "coverage": round(defended_passes / attempts, 4) if attempts else None,
            "interception_share": (
                round(interceptions / defended, 4) if defended else None
            ),
            "implied_share": (round(interceptions / implied, 4) if implied else None),
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


if __name__ == "__main__":
    fire.Fire({"train": train, "curve": curve, "rates": rates})
