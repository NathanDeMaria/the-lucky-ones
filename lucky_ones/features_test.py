import numpy as np
import pytest

from .conftest import make_state as _state
from .features import (
    EP_FEATURE_NAMES,
    FEATURE_NAMES,
    ep_feature_matrix,
    feature_matrix,
    to_ep_features,
    to_features,
)
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


# --- Expected points features ------------------------------------------


def _ep_feature(state: GameState, name: str) -> float:
    return float(to_ep_features(state)[EP_FEATURE_NAMES.index(name)])


def test_an_ep_row_per_ep_feature_name():
    assert to_ep_features(_state()).shape == (len(EP_FEATURE_NAMES),)


def test_the_two_feature_lists_share_only_what_means_the_same_thing():
    """
    Two models, two questions, and almost no overlap: win probability is
    oriented to the home team and reads the score, expected points is
    oriented to the offense and doesn't. `log_distance` is the one name in
    both, and it is the same transform of the same column in each.
    """
    assert set(FEATURE_NAMES) & set(EP_FEATURE_NAMES) == {"log_distance"}


def test_expected_points_does_not_see_the_score():
    """
    The decision the module docstring argues for: a blowout genuinely
    predicts the next points, which is why it stays out of the price of a
    situation. `lucky_ones.epa` handles it in the weighting instead.
    """
    level = to_ep_features(_state(score_margin=0))
    routed = to_ep_features(_state(score_margin=-35))

    assert np.array_equal(level, routed)


def test_expected_points_does_not_see_which_team_is_home():
    """
    Otherwise the same snap is worth more to one side, and the two teams'
    EPA per play stop being on one scale.
    """
    home = to_ep_features(_state(offense_is_home=True))
    away = to_ep_features(_state(offense_is_home=False))

    assert np.array_equal(home, away)


def test_down_is_indicators_rather_than_a_number():
    """
    The drop from third to fourth is nothing like the drop from first to
    second, and a line through four points can't say both.
    """
    downs = [
        [
            _ep_feature(_state(down=down), name)
            for name in ("down_2", "down_3", "down_4")
        ]
        for down in (1, 2, 3, 4)
    ]

    assert downs == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_the_field_is_measured_from_the_end_zone_being_attacked():
    assert _ep_feature(_state(yardline=1), "yards_to_goal") == pytest.approx(0.99)
    assert _ep_feature(_state(yardline=99), "yards_to_goal") == pytest.approx(0.01)


def test_the_spline_bends_where_the_football_does():
    """
    Each `yards_past_*` term is zero until the ball is farther out than its
    landmark, which is what lets the curve change slope there instead of
    spending one bend on the whole field.
    """
    red_zone = _state(yardline=85)  # 15 yards out: past the 10, not the 20
    backed_up = _state(yardline=5)  # 95 yards out: past all of them

    assert _ep_feature(red_zone, "yards_past_opponent_10") == pytest.approx(0.05)
    assert _ep_feature(red_zone, "yards_past_opponent_20") == 0.0
    assert _ep_feature(red_zone, "yards_past_own_10") == 0.0
    assert _ep_feature(backed_up, "yards_past_own_10") == pytest.approx(0.05)


def test_goal_to_go_is_read_off_the_yardline():
    """ESPN has no goal-to-go column, and it doesn't need one."""
    assert _ep_feature(_state(yardline=95, distance=8), "goal_to_go") == 1.0
    assert _ep_feature(_state(yardline=50, distance=8), "goal_to_go") == 0.0


def test_the_clock_is_the_half_s_not_the_game_s():
    """
    Expected points asks how much time this drive has, and that runs out at
    the half. A first down at midfield with eight seconds left in the second
    quarter is nearly worthless, and `seconds_remaining` would call it a
    normal snap with a quarter and a half to go.
    """
    fresh = _state(period=1, clock_seconds=900)
    expiring = _state(period=2, clock_seconds=8, seconds_remaining=908)

    assert _ep_feature(fresh, "fraction_half_remaining") == pytest.approx(1.0)
    assert _ep_feature(expiring, "fraction_half_remaining") == pytest.approx(8 / 1800)


def test_the_clock_and_the_field_interact():
    """
    A minute left is worth nothing on your own 10 and a field goal on the
    opponent's 15, so the clock term can't be a level on its own.
    """
    deep = _ep_feature(
        _state(period=2, clock_seconds=60, yardline=10),
        "half_remaining_x_yards_to_goal",
    )
    close = _ep_feature(
        _state(period=2, clock_seconds=60, yardline=85),
        "half_remaining_x_yards_to_goal",
    )

    assert deep > close


def test_ep_feature_matrix_keeps_its_width_when_empty():
    assert ep_feature_matrix([]).shape == (0, len(EP_FEATURE_NAMES))
