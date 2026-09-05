from .arrow import table_to_plays
from .conftest import make_play, make_table
from .game import GamePlays
from .points import ScoreKind
from .training import build_expected_points_set, build_training_set, split_games


def _game(game_id: str, home_final: int, away_final: int, snaps: int = 3) -> GamePlays:
    rows = [
        make_play(
            game_id=game_id,
            play_id=f"{game_id}-{n}",
            play_number=n,
            home_score=home_final if n == snaps else 0,
            away_score=away_final if n == snaps else 0,
        )
        for n in range(1, snaps + 1)
    ]
    return GamePlays(
        game_id=game_id,
        league="nfl",
        season=2025,
        week=1,
        home_team_id="home",
        away_team_id="away",
        plays=table_to_plays(make_table(rows)),
    )


def test_every_snap_carries_its_game_s_result():
    training = build_training_set([_game("g1", 24, 17), _game("g2", 10, 31)])

    assert training.rows == 6
    assert training.home_won == [True] * 3 + [False] * 3


def test_a_tie_is_dropped():
    """
    Rare but real in the NFL, and there's no binary label for it.
    """
    training = build_training_set([_game("tie", 17, 17), _game("g1", 24, 17)])

    assert {state.game_id for state in training.states} == {"g1"}


def test_a_game_with_no_scores_at_all_is_dropped():
    game = _game("empty", 0, 0)
    game = game._replace(
        plays=table_to_plays(
            make_table([make_play(game_id="empty", home_score=None, away_score=None)])
        )
    )

    assert build_training_set([game]).rows == 0


def test_the_split_is_by_game_not_by_row():
    """
    Splitting rows would put snaps of the same game on both sides, and they
    all share a label -- so the holdout would be scoring the model on games
    it had already been told the answer to.
    """
    games = [_game(f"g{n}", 21, 14) for n in range(10)]

    train, holdout = split_games(games, holdout_fraction=0.2)

    assert len(train) == 8 and len(holdout) == 2
    assert not {game.game_id for game in train} & {game.game_id for game in holdout}


def test_the_split_is_deterministic():
    games = [_game(f"g{n}", 21, 14) for n in range(10)]

    first, _ = split_games(games, seed=7)
    again, _ = split_games(list(reversed(games)), seed=7)

    assert [game.game_id for game in first] == [game.game_id for game in again]


def _scoring_game(game_id: str, snaps: int = 3) -> GamePlays:
    """
    A game whose last snap scores a touchdown, so every snap of it has a real
    expected points label rather than `NO_SCORE`.
    """
    rows = [
        make_play(
            game_id=game_id,
            play_id=f"{game_id}-{n}",
            play_number=n,
            home_score=7 if n == snaps else 0,
            scoring_play=n == snaps,
        )
        for n in range(1, snaps + 1)
    ]
    return GamePlays(
        game_id=game_id,
        league="nfl",
        season=2025,
        week=1,
        home_team_id="home",
        away_team_id="away",
        plays=table_to_plays(make_table(rows)),
    )


def test_every_snap_carries_what_scored_next_in_its_half():
    training = build_expected_points_set([_scoring_game("g1")])

    assert training.rows == 3
    assert training.next_score == [ScoreKind.OFFENSE_TOUCHDOWN] * 3


def test_a_tie_is_kept_for_expected_points():
    """
    Unlike a win probability label, this one doesn't need the game to have
    had a winner -- every snap of a tie has a real next score.
    """
    training = build_expected_points_set([_game("tie", 17, 17)])

    assert training.rows == 3


def test_overtime_is_dropped_for_expected_points():
    """
    `half_seconds_remaining` is pinned to zero there, so an overtime snap
    would arrive at the fit looking like the last play of a half -- and
    college overtime is not a situation regulation football has a value for.
    """
    rows = [
        make_play(play_id="p1", play_number=1, period=4, clock_seconds=100),
        make_play(play_id="p2", play_number=2, period=5, clock_seconds=0),
    ]
    game = GamePlays(
        game_id="ot",
        league="nfl",
        season=2025,
        week=1,
        home_team_id="home",
        away_team_id="away",
        plays=table_to_plays(make_table(rows)),
    )

    training = build_expected_points_set([game])

    assert [state.play_id for state in training.states] == ["p1"]
