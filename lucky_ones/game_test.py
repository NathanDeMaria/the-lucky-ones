from .arrow import table_to_plays
from .conftest import make_play, make_table
from .game import group_by_game, infer_home_team_id


def _scoring_drive(play_number: int, team: str, home: int, away: int, **overrides):
    """A play that put points on the board for `team`, leaving the score at
    `home`-`away` afterwards."""
    return make_play(
        play_id=f"p{play_number}",
        play_number=play_number,
        offense_team_id=team,
        defense_team_id="away" if team == "home" else "home",
        home_score=home,
        away_score=away,
        scoring_play=True,
        drive_is_score=True,
        **overrides,
    )


def _plays(rows):
    return table_to_plays(make_table(rows))


def test_infers_home_from_scoring_drives():
    """
    Team "a" scores twice and the home score moves both times, so "a" is
    home -- which is the only way to know, since nothing in the schema says.
    """
    plays = _plays(
        [
            make_play(play_id="p1", play_number=1, offense_team_id="a"),
            _scoring_drive(2, "a", 7, 0),
            make_play(play_id="p3", play_number=3, offense_team_id="b"),
            _scoring_drive(4, "b", 7, 3),
            _scoring_drive(5, "a", 14, 3),
        ]
    )

    assert infer_home_team_id(plays) == "a"


def test_a_defensive_score_does_not_flip_the_call():
    """
    A pick-six scores for the team that didn't have the ball, so it votes the
    wrong way. One of those against three ordinary scores has to lose.
    """
    plays = _plays(
        [
            _scoring_drive(1, "a", 7, 0),
            make_play(play_id="p2", play_number=2, offense_team_id="b"),
            _scoring_drive(3, "a", 14, 0),
            make_play(play_id="p4", play_number=4, offense_team_id="b"),
            _scoring_drive(5, "a", 21, 0),
            # "a" had the ball; "b" returned it for a touchdown.
            _scoring_drive(6, "a", 21, 7, is_turnover=True),
        ]
    )

    assert infer_home_team_id(plays) == "a"


def test_no_scoring_plays_is_unknowable():
    plays = _plays(
        [
            make_play(play_id="p1", play_number=1, offense_team_id="a"),
            make_play(play_id="p2", play_number=2, offense_team_id="b"),
        ]
    )

    assert infer_home_team_id(plays) is None


def test_an_even_split_is_unknowable_rather_than_a_coin_flip():
    plays = _plays(
        [
            _scoring_drive(1, "a", 7, 0),
            _scoring_drive(2, "b", 7, 7),
            # "a" scored but the away side moved, and vice versa: the two
            # assignments now explain exactly two votes each.
            _scoring_drive(3, "a", 7, 14),
            _scoring_drive(4, "b", 14, 14),
        ]
    )

    assert infer_home_team_id(plays) is None


def test_group_by_game_splits_and_labels_sides():
    plays = _plays(
        [
            _scoring_drive(1, "a", 7, 0, game_id="g1"),
            make_play(game_id="g1", play_id="g1p2", play_number=2, offense_team_id="b"),
            _scoring_drive(1, "c", 0, 7, game_id="g2"),
            make_play(game_id="g2", play_id="g2p2", play_number=2, offense_team_id="d"),
        ]
    )

    games = group_by_game(plays)

    assert [(game.game_id, game.home_team_id) for game in games] == [
        ("g1", "a"),
        ("g2", "d"),
    ]
    assert games[0].away_team_id == "b"
    assert (games[0].league, games[0].season, games[0].week) == ("nfl", 2025, 1)


def test_an_explicit_mapping_beats_inference():
    """
    Inference is the fallback. When the home side is known from somewhere
    authoritative, that wins -- including where it contradicts the votes.
    """
    plays = _plays(
        [
            _scoring_drive(1, "a", 7, 0),
            make_play(play_id="p2", play_number=2, offense_team_id="b"),
        ]
    )

    (game,) = group_by_game(plays, home_team_ids={"g1": "b"})

    assert (game.home_team_id, game.away_team_id) == ("b", "a")


def test_a_game_with_no_home_side_is_dropped():
    plays = _plays(
        [
            make_play(play_id="p1", play_number=1, offense_team_id="a"),
            make_play(play_id="p2", play_number=2, offense_team_id="b"),
        ]
    )

    assert group_by_game(plays) == []
