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
from lucky_ones.luck import adjusted_curve_from_states, find_lucky_plays
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

    Both control numbers come out, plus the plays between them: `lucky_plays`
    is every fumble and tipped interception the game turned on, and each point
    carries its luck-adjusted probability alongside the real one. See
    `lucky_ones.luck`.

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
    adjusted = adjusted_curve_from_states(fit, states, find_lucky_plays(game.plays))
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
                "lucky_plays": [lucky._asdict() for lucky in adjusted.lucky_plays],
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


def _whoami() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - depends on the environment
        # No passwd entry, which is normal in a container running as a bare
        # uid. A release with an unknown author is better than a failed run.
        return "unknown"


if __name__ == "__main__":
    fire.Fire({"train": train, "curve": curve})
