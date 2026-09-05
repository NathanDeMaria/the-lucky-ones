import pytest

from .arrow import table_to_plays
from .conftest import make_play, make_state, make_table
from .game import GamePlays
from .state import (
    REGULATION_SECONDS,
    final_outcome,
    half_of,
    is_scrimmage_play,
    iter_states,
)


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


def test_the_halves_are_where_the_periods_say():
    assert [half_of(period) for period in (1, 2, 3, 4)] == [1, 1, 2, 2]


def test_every_overtime_period_is_its_own_half():
    """
    Which is what the boundary is for: the next scoring drive doesn't carry
    across a kickoff that resets the situation, and college overtime resets
    it every possession pair.
    """
    assert half_of(5) == 3
    assert half_of(6) == 4


def test_the_half_clock_counts_the_rest_of_the_half():
    assert make_state(period=1, clock_seconds=900).half_seconds_remaining == 1800
    assert make_state(period=2, clock_seconds=900).half_seconds_remaining == 900
    assert make_state(period=3, clock_seconds=120).half_seconds_remaining == 1020
    assert make_state(period=4, clock_seconds=0).half_seconds_remaining == 0


def test_the_half_clock_is_zero_in_overtime():
    """
    Same reason `seconds_remaining` is: there's no league-agnostic length to
    count down from, and `lucky_ones.epa` measures regulation only.
    """
    state = make_state(period=5, clock_seconds=600, is_overtime=True)

    assert state.half_seconds_remaining == 0


def test_a_timeout_is_not_a_snap():
    """
    The filter nothing else can make: a timeout arrives with a down, a
    distance, a yardline, a clock and a possession team, so every column test
    passes and only `play_type` says it wasn't a play.
    """
    game = _game(
        [
            make_play(play_id="p1", play_number=1, clock_seconds=900),
            make_play(
                play_id="p2",
                play_number=2,
                clock_seconds=880,
                play_type="Official Timeout",
            ),
            make_play(play_id="p3", play_number=3, clock_seconds=860),
        ]
    )

    assert [state.play_id for state in iter_states(game)] == ["p1", "p3"]


@pytest.mark.parametrize(
    "play_type",
    [
        "Timeout",
        "Official Timeout",
        "Two-minute warning",
        "Kickoff",
        "Kickoff Return (Offense)",
        "End Period",
        "End of Game",
        "Coin Toss",
    ],
)
def test_the_administrative_rows_are_all_refused(play_type):
    assert not is_scrimmage_play(play_type)


@pytest.mark.parametrize(
    "play_type",
    ["Rush", "Pass Reception", "Punt", "Penalty", "Field Goal Good", "Sack", None],
)
def test_a_play_run_from_scrimmage_is_kept(play_type):
    """
    Punts and field goals are downs and are priced as downs; a penalty is a
    snap the flag came out on. And a null type is kept -- absence of evidence
    isn't evidence of a timeout.
    """
    assert is_scrimmage_play(play_type)


def test_a_college_kickoff_that_carries_a_down_is_still_a_kickoff():
    """
    NCAAFB's kickoffs usually do have a down, so the column test misses them
    entirely -- they were 5.5% of that league's states.
    """
    game = _game(
        [
            make_play(
                play_id="p1", play_number=1, play_type="Kickoff", down=1, distance=10
            ),
            make_play(play_id="p2", play_number=2, clock_seconds=880),
        ]
    )

    assert [state.play_id for state in iter_states(game)] == ["p2"]
