import numpy as np

from .conftest import make_state as _state
from .features import FEATURE_NAMES, feature_matrix, to_features
from .state import GameState


def _feature(state: GameState, name: str) -> float:
    return float(to_features(state)[FEATURE_NAMES.index(name)])


def test_a_row_per_feature_name():
    assert to_features(_state()).shape == (len(FEATURE_NAMES),)


def test_the_same_lead_is_worth_more_late():
    """
    The whole point of margin-over-root-time: a linear term in margin can't
    tell the first quarter from the last minute, and this has to.
    """
    early = _feature(
        _state(score_margin=7, seconds_remaining=3000), "margin_per_root_time"
    )
    late = _feature(
        _state(score_margin=7, seconds_remaining=10), "margin_per_root_time"
    )

    assert late > early


def test_no_division_by_zero_at_the_end_of_the_game():
    value = _feature(
        _state(score_margin=7, seconds_remaining=0), "margin_per_root_time"
    )

    assert np.isfinite(value)


def test_possession_is_signed():
    assert _feature(_state(offense_is_home=True), "offense_is_home") == 1.0
    assert _feature(_state(offense_is_home=False), "offense_is_home") == -1.0


def test_field_position_is_read_from_the_home_team_s_side():
    """
    Deep in your own end is bad for *you*, so the same yardline has to point
    opposite ways depending on who has the ball.
    """
    home_has_it = _feature(_state(offense_is_home=True, yardline=10), "yardline")
    away_has_it = _feature(_state(offense_is_home=False, yardline=10), "yardline")

    assert home_has_it == -away_has_it


def test_a_negative_distance_does_not_produce_a_nan():
    """
    ESPN sends a negative distance for a penalty enforced from behind the
    spot, which is why the source column is an int16 rather than an int8.
    """
    value = _feature(_state(distance=-5), "log_distance")

    assert np.isfinite(value)


def test_feature_matrix_keeps_its_width_when_empty():
    assert feature_matrix([]).shape == (0, len(FEATURE_NAMES))


def test_feature_matrix_stacks_rows():
    assert feature_matrix([_state(), _state(down=2)]).shape == (2, len(FEATURE_NAMES))
