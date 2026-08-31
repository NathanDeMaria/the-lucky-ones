import numpy as np
import pytest

from .arrow import table_to_plays
from .conftest import make_play, make_table
from .curve import CurvePoint, game_control, win_probability_curve
from .game import GamePlays


class _FixedModel:
    """A model that says what it was told to, so the curve maths is testable
    without a fit in the way."""

    def __init__(self, probabilities) -> None:
        self._probabilities = list(probabilities)

    def predict(self, states) -> np.ndarray:
        return np.array(self._probabilities[: len(states)], dtype=float)


def _game(rows) -> GamePlays:
    return GamePlays(
        game_id="g1",
        league="nfl",
        season=2025,
        week=1,
        home_team_id="home",
        away_team_id="away",
        plays=table_to_plays(make_table(rows)),
    )


def _point(seconds_remaining: int, probability: float) -> CurvePoint:
    return CurvePoint(
        play_id=f"p{seconds_remaining}",
        play_number=1,
        period=1,
        clock_seconds=0,
        seconds_remaining=seconds_remaining,
        home_score=0,
        away_score=0,
        home_win_probability=probability,
    )


def test_the_curve_carries_what_a_chart_needs():
    game = _game(
        [
            make_play(
                play_id="p1",
                play_number=1,
                period=3,
                clock_seconds=442,
                home_score=14,
                away_score=10,
            )
        ]
    )

    (point,) = win_probability_curve(_FixedModel([0.7]), game)

    assert point.play_id == "p1"
    assert (point.period, point.clock_seconds) == (3, 442)
    assert point.home_win_probability == pytest.approx(0.7)
    # The score before the play, so a tooltip and the model agree
    assert (point.home_score, point.away_score) == (0, 0)


def test_a_game_with_no_scrimmage_plays_has_no_curve():
    game = _game([make_play(down=None, offense_team_id=None, yardline=None)])

    assert win_probability_curve(_FixedModel([0.5]), game) == []


def test_game_control_weights_by_time_not_by_play():
    """
    The whole reason for the weighting: three frantic snaps at the end of a
    game can't outvote the fifty-five minutes before them.
    """
    points = [
        _point(3600, 0.0),  # stood for 3595 seconds
        _point(5, 1.0),  # stood for 3
        _point(2, 1.0),  # stood for 1
        _point(1, 1.0),  # stood for 1
    ]

    control = game_control(points)

    assert control is not None
    # An unweighted mean would be 0.75
    assert control.home == pytest.approx(5 / 3600, abs=1e-6)


def test_control_sides_sum_to_one():
    control = game_control([_point(3600, 0.8), _point(1800, 0.8)])

    assert control is not None
    assert control.home + control.away == pytest.approx(1.0)


def test_a_wire_to_wire_win_approaches_one():
    control = game_control([_point(3600, 0.95), _point(1800, 0.97)])

    assert control is not None
    assert control.home > 0.9


def test_control_reports_the_time_it_covers():
    control = game_control([_point(3600, 0.5), _point(1800, 0.5)])

    assert control is not None
    assert control.seconds == 3600


def test_a_clock_that_goes_backwards_contributes_nothing():
    """
    Two snaps can share a clock reading, and ESPN's occasionally runs
    backwards. Either way that's zero seconds of game time, not negative
    ones -- a negative weight would pull the average the wrong way.
    """
    control = game_control([_point(1800, 1.0), _point(1900, 0.0), _point(900, 0.0)])

    assert control is not None
    # The 1.0 is the one whose clock went backwards, so it weighs nothing
    assert control.home == pytest.approx(0.0)


def test_overtime_only_has_nothing_to_weight():
    """
    `seconds_remaining` is pinned to zero in overtime, so an overtime-only
    curve has no elapsed time -- None rather than 0.5, which would read as a
    genuinely even game.
    """
    assert game_control([_point(0, 0.9), _point(0, 0.1)]) is None


def test_an_empty_curve_has_no_control():
    assert game_control([]) is None
