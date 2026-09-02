"""
Football-shaped play-by-play, generated rather than downloaded.

Real plays need AWS and a season of them takes a while, which makes the
training script the one part of this repo nobody would exercise before
running it for real. This builds a tree in endgame's processed layout
(`league={}/season={}/week={}/data.parquet`) out of games with a plausible
clock, drives that alternate, and scores that follow a per-game team
strength -- enough for `train.py --root` to fit something and for a test to
assert it did.

It is a fixture, not a simulator. The plays are not football; the *shape* is.
Never fit a model you intend to use on this.
"""

import random
from pathlib import Path

import pyarrow.parquet as pq

from lucky_ones.conftest import make_play, make_table

PERIOD_SECONDS = 15 * 60
SNAP_SECONDS = 30


# Roughly a fumble every other drive, which is more than football has -- a
# fixture with one bounce in it exercises `lucky_ones.luck` about as well as a
# fixture with none. Both branches appear, because the module reads
# `is_turnover` to tell them apart and a tree with only lost fumbles in it
# would never exercise the other one.
FUMBLE_CHANCE = 0.02


def game_rows(game_id: str, home: str, away: str, seed: int) -> list[dict]:
    """
    One game: four periods of alternating drives, with the stronger team more
    likely to score. Ends with an administrative play carrying the final
    score and no possession, the way a real game does.

    A few plays carry fumble text, so `make curve` on this tree has something
    for the luck adjustment to find. The text is the shape ESPN writes, not a
    quote of it -- like everything else here, it's the shape that's real.
    """
    rng = random.Random(seed)
    strength = rng.gauss(0, 10)
    rows: list[dict] = []
    home_score = away_score = 0
    play_number = 0
    for period in (1, 2, 3, 4):
        for clock in range(PERIOD_SECONDS, 0, -SNAP_SECONDS):
            play_number += 1
            offense = home if (play_number // 6) % 2 == 0 else away
            # Sharper than football, on purpose: a fixture whose scores don't
            # track the strength teaches the fit nothing.
            edge = strength if offense == home else -strength
            if rng.random() < 0.04 + max(0.0, edge) / 400:
                points = rng.choice([3, 7])
                if offense == home:
                    home_score += points
                else:
                    away_score += points
            fumbled = rng.random() < FUMBLE_CHANCE
            lost = fumbled and rng.random() < 0.5
            recovered = away if offense == home else home
            rows.append(
                make_play(
                    game_id=game_id,
                    play_id=f"{game_id}-{play_number}",
                    play_number=play_number,
                    period=period,
                    clock_seconds=clock,
                    offense_team_id=offense,
                    defense_team_id=away if offense == home else home,
                    home_score=home_score,
                    away_score=away_score,
                    down=rng.randint(1, 4),
                    distance=rng.randint(1, 15),
                    yardline=rng.randint(1, 99),
                    drive_number=1 + play_number // 6,
                    text=(
                        "Rush for 3 yards. FUMBLE, recovered by "
                        f"{recovered if lost else offense}."
                        if fumbled
                        else "Rush for 3 yards"
                    ),
                    is_turnover=lost,
                )
            )
    if home_score == away_score:
        # A tie has no label, and `build_training_set` drops it. Fine in real
        # data, a waste of a fixture game here.
        home_score += 3
        rows[-1]["home_score"] = home_score
    rows.append(
        make_play(
            game_id=game_id,
            play_id=f"{game_id}-end",
            play_number=play_number + 1,
            period=4,
            clock_seconds=0,
            offense_team_id=None,
            defense_team_id=None,
            down=None,
            yardline=None,
            home_score=home_score,
            away_score=away_score,
            play_type="End of Game",
            drive_number=1 + play_number // 6,
        )
    )
    return rows


def write_tree(
    root: Path,
    league: str = "nfl",
    season: int = 2025,
    weeks: int = 4,
    games_per_week: int = 8,
    seed: int = 0,
) -> Path:
    """
    Write a whole synthetic season under `root`, and return it.

    Layout matches what `aws s3 sync s3://BUCKET/processed/plays ./plays`
    produces, so `train.py --root` can't tell the difference.
    """
    root = Path(root)
    for week in range(1, weeks + 1):
        rows: list[dict] = []
        for game in range(games_per_week):
            number = week * 100 + game
            rows.extend(
                game_rows(
                    f"g{number}",
                    home=f"home{number}",
                    away=f"away{number}",
                    seed=seed + number,
                )
            )
        for row in rows:
            row["league"], row["season"], row["week"] = league, season, week
        directory = root / f"league={league}" / f"season={season}" / f"week={week}"
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(make_table(rows), directory / "data.parquet")
    return root


if __name__ == "__main__":
    import fire

    fire.Fire(write_tree)
