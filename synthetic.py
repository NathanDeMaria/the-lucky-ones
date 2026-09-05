"""
Football-shaped play-by-play, generated rather than downloaded.

Real plays need AWS and a season of them takes a while, which makes the
training script the one part of this repo nobody would exercise before
running it for real. This builds a tree in endgame's processed layout
(`league={}/season={}/week={}/data.parquet`) out of games with a plausible
clock, drives that alternate, and scores that follow a per-game team
strength -- enough for `train.py --root` to fit something and for a test to
assert it did. Scores move the score columns *and* set `scoring_play`,
because both models read them and `lucky_ones.points` insists on both.

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
# fixture with none. Both branches appear, because the two are classified
# differently and a tree with only lost fumbles in it would never exercise
# the other one.
FUMBLE_CHANCE = 0.02

# Enough passes, and enough of them incomplete, for both halves of the pass
# coin to appear: an interception, and a defended incompletion in the shape
# NCAAFB writes it ("broken up by"). The rates are chosen so a game clears
# `luck.records_defended_passes` -- roughly 19 incompletions with a third of
# them defended, against a floor of ten and 15%. A fixture that fell below the
# gate would exercise only the path where the gate is shut.
PASS_CHANCE = 0.45
INCOMPLETION_CHANCE = 0.36
BROKEN_UP_CHANCE = 0.30
INTERCEPTION_CHANCE = 0.03

# A drive is six snaps unless the ball changes hands first.
DRIVE_SNAPS = 6


def game_rows(game_id: str, home: str, away: str, seed: int) -> list[dict]:
    """
    One game: four periods of alternating drives, with the stronger team more
    likely to score. Ends with an administrative play carrying the final
    score and no possession, the way a real game does.

    A few plays carry fumble text, so `make curve` on this tree has something
    for the luck adjustment to find, and about half are passes so `make rates`
    has a denominator. The text is the shape ESPN writes, not a quote of it --
    like everything else here, it's the shape that's real.

    Possession follows the text rather than a fixed cadence, which matters
    since `lucky_ones.luck` reads the next snap's offense to decide whether a
    fumble was lost: a tree where the two disagree would be a fixture arguing
    with the code it exists to exercise.
    """
    rng = random.Random(seed)
    strength = rng.gauss(0, 10)
    rows: list[dict] = []
    home_score = away_score = 0
    play_number = 0
    offense = home
    drive_number = 1
    drive_snaps = 0
    for period in (1, 2, 3, 4):
        for clock in range(PERIOD_SECONDS, 0, -SNAP_SECONDS):
            play_number += 1
            drive_snaps += 1
            defense = away if offense == home else home
            # Sharper than football, on purpose: a fixture whose scores don't
            # track the strength teaches the fit nothing.
            edge = strength if offense == home else -strength
            scored = rng.random() < 0.04 + max(0.0, edge) / 400
            if scored:
                points = rng.choice([3, 7])
                if offense == home:
                    home_score += points
                else:
                    away_score += points
            passed = rng.random() < PASS_CHANCE
            fumbled = rng.random() < FUMBLE_CHANCE
            lost = fumbled and rng.random() < 0.5
            live_pass = passed and not fumbled
            intercepted = live_pass and rng.random() < INTERCEPTION_CHANCE
            incomplete = (
                live_pass and not intercepted and rng.random() < INCOMPLETION_CHANCE
            )
            broken_up = incomplete and rng.random() < BROKEN_UP_CHANCE
            if fumbled:
                text = (
                    f"{'Pass complete' if passed else 'Rush'} for 3 yards. "
                    f"FUMBLE, recovered by {defense if lost else offense}."
                )
                play_type = "Pass Reception" if passed else "Rush"
            elif intercepted:
                text = "Pass intercepted."
                play_type = "Pass Interception"
            elif broken_up:
                text = "Pass incomplete short right to Smith, broken up by Jones."
                play_type = "Pass Incompletion"
            elif incomplete:
                text = "Pass incomplete short right to Smith."
                play_type = "Pass Incompletion"
            elif passed:
                text = "Pass complete for 6 yards"
                play_type = "Pass Reception"
            else:
                text = "Rush for 3 yards"
                play_type = "Rush"
            rows.append(
                make_play(
                    game_id=game_id,
                    play_id=f"{game_id}-{play_number}",
                    play_number=play_number,
                    period=period,
                    clock_seconds=clock,
                    offense_team_id=offense,
                    defense_team_id=defense,
                    home_score=home_score,
                    away_score=away_score,
                    down=rng.randint(1, 4),
                    distance=rng.randint(1, 15),
                    yardline=rng.randint(1, 99),
                    play_type=play_type,
                    drive_number=drive_number,
                    text=text,
                    is_turnover=lost or intercepted,
                    # The second witness `lucky_ones.points.score_events`
                    # needs. A tree that moved the score without flagging it
                    # would be a fixture of the pre-2014 feed's bug, and
                    # expected points would find nothing in it to fit.
                    scoring_play=scored,
                )
            )
            if lost or intercepted or drive_snaps >= DRIVE_SNAPS:
                offense, drive_snaps = defense, 0
                drive_number += 1
    if home_score == away_score:
        # A tie has no label, and `build_training_set` drops it. Fine in real
        # data, a waste of a fixture game here -- so the home team kicks a
        # late field goal on the last snap that didn't already score. Three
        # points onto that row and every row after it, since the column is
        # cumulative, and the flag onto the row itself, since every jump in it
        # carries one.
        kicked = max(index for index, row in enumerate(rows) if not row["scoring_play"])
        for row in rows[kicked:]:
            row["home_score"] += 3
        rows[kicked]["scoring_play"] = True
        home_score += 3
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
            drive_number=drive_number,
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
    produces, so `train.py --root` can't tell the difference. Called twice
    with two seasons it writes both, which is what `validate.py` needs -- its
    whole design is fitting on one and measuring on another.

    The teams are a fixed pool that plays every week rather than a fresh pair
    per game, because anything measuring a *team* needs a team with more than
    one game. `validate.py`'s split-half section is the caller that cares:
    against a tree where nobody plays twice it would have had nothing to deal
    and would have reported an empty section rather than failing.

    The pairing rotates by week, so a team alternates home and away. That is
    football-shaped and it is also the trap worth having in a fixture: a
    split-half design that took every *other* game would put a team's home
    games in one set and its away games in the other.
    """
    root = Path(root)
    names = [f"t{index}" for index in range(games_per_week * 2)]
    for week in range(1, weeks + 1):
        rows: list[dict] = []
        for game in range(games_per_week):
            number = week * 100 + game
            home = names[(2 * game + week) % len(names)]
            away = names[(2 * game + 1 + week) % len(names)]
            rows.extend(
                game_rows(
                    f"g{season}{number}",
                    home=home,
                    away=away,
                    # The season is in the seed so two of them aren't the same
                    # games under different labels.
                    seed=seed + number + season * 1000,
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
