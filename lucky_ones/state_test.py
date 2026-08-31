from .arrow import table_to_plays
from .conftest import make_play, make_table
from .game import GamePlays
from .state import REGULATION_SECONDS, final_outcome, iter_states


def _game(rows, home_team_id="home"):
    plays = table_to_plays(make_table(rows))
    return GamePlays(
        game_id="g1",
        league="nfl",
        season=2025,
        week=1,
        home_team_id=home_team_id,
        away_team_id="away",
        plays=plays,
    )


def test_the_score_is_the_one_before_the_play():
    """
    The load-bearing one. endgame's score columns are cumulative *after* the
    play, so the touchdown play itself already carries its own points --
    training on that leaks the result into the features.
    """
    game = _game(
        [
            make_play(play_id="p1", play_number=1, home_score=0, away_score=0),
            # The 7 points are on the play that scored them.
            make_play(
                play_id="p2",
                play_number=2,
                home_score=7,
                away_score=0,
                scoring_play=True,
            ),
            make_play(play_id="p3", play_number=3, home_score=7, away_score=0),
        ]
    )

    margins = [state.score_margin for state in iter_states(game)]

    assert margins == [0, 0, 7]


def test_non_snaps_are_skipped():
    """
    Kickoffs have no down and END QUARTER has no possession team. Both are
    ordinary rows in the source data, not errors.
    """
    game = _game(
        [
            make_play(play_id="k", play_number=1, down=None, play_type="Kickoff"),
            make_play(play_id="snap", play_number=2),
            make_play(
                play_id="end",
                play_number=3,
                offense_team_id=None,
                down=None,
                yardline=None,
                play_type="End of Quarter",
            ),
        ]
    )

    assert [state.play_id for state in iter_states(game)] == ["snap"]


def test_seconds_remaining_counts_down_the_whole_game():
    game = _game(
        [
            make_play(play_id="p1", play_number=1, period=1, clock_seconds=900),
            make_play(play_id="p2", play_number=2, period=3, clock_seconds=600),
            make_play(play_id="p3", play_number=3, period=4, clock_seconds=0),
        ]
    )

    remaining = [state.seconds_remaining for state in iter_states(game)]

    assert remaining == [REGULATION_SECONDS, 600 + 900, 0]


def test_overtime_has_no_regulation_clock_left():
    game = _game([make_play(period=5, clock_seconds=540)])

    (state,) = iter_states(game)

    assert state.is_overtime
    assert state.seconds_remaining == 0


def test_offense_is_home_follows_the_ball():
    game = _game(
        [
            make_play(play_id="p1", play_number=1, offense_team_id="home"),
            make_play(play_id="p2", play_number=2, offense_team_id="away"),
        ]
    )

    assert [state.offense_is_home for state in iter_states(game)] == [True, False]


def test_final_outcome_reads_past_the_administrative_last_play():
    game = _game(
        [
            make_play(play_id="p1", play_number=1, home_score=21, away_score=17),
            make_play(
                play_id="end",
                play_number=2,
                offense_team_id=None,
                down=None,
                home_score=21,
                away_score=17,
            ),
        ]
    )

    outcome = final_outcome(game)

    assert outcome is not None
    assert (outcome.home_score, outcome.away_score) == (21, 17)
    assert outcome.home_won
    assert not outcome.tied


def test_final_outcome_of_a_game_with_no_scores_at_all():
    game = _game([make_play(home_score=None, away_score=None)])

    assert final_outcome(game) is None
