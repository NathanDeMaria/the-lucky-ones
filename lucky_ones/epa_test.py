import numpy as np
import pytest

from .arrow import table_to_plays
from .conftest import make_play, make_table
from .epa import (
    DEFAULT_CLIP,
    competitiveness,
    epa_per_play,
    epa_per_play_from_states,
    play_epa,
)
from .game import GamePlays
from .points import scoring_plays
from .state import iter_states

INFINITE = float("inf")
"""`clip=INFINITE, weight_power=0.0` is the identity -- the plain mean."""


class _FixedPoints:
    """An expected points model that says what it was told to, one per state."""

    def __init__(self, points) -> None:
        self._points = list(points)

    def predict(self, states) -> np.ndarray:
        return np.array(self._points[: len(states)], dtype=float)


class _FixedWinProbability:
    """The same for win probability, so the weighting is testable on its own."""

    def __init__(self, probabilities=None) -> None:
        self._probabilities = probabilities

    def predict(self, states) -> np.ndarray:
        if self._probabilities is None:
            # Dead even, so every play weighs exactly 1 and the weighting is
            # out of the way of a test that isn't about it.
            return np.full(len(states), 0.5)
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


def _snaps(*snaps) -> GamePlays:
    """
    A game from `(offense, clock, overrides)` triples, one snap each, all in
    the first quarter unless a snap says otherwise.
    """
    return _game(
        [
            make_play(
                play_id=f"p{number}",
                play_number=number,
                clock_seconds=clock,
                offense_team_id=offense,
                defense_team_id="away" if offense == "home" else "home",
                **overrides,
            )
            for number, (offense, clock, overrides) in enumerate(snaps, start=1)
        ]
    )


def _epa(game: GamePlays, points, probabilities=None, **kwargs):
    return epa_per_play_from_states(
        _FixedPoints(points),
        _FixedWinProbability(probabilities),
        list(iter_states(game)),
        scoring_plays(list(game.plays)),
        **kwargs,
    )


# --- What a play is worth ----------------------------------------------


def test_a_play_is_worth_the_change_in_expected_points():
    """The definition, on the ordinary case: same offense, next snap."""
    game = _snaps(("home", 900, {}), ("home", 870, {}))

    plays = play_epa(
        _FixedPoints([1.0, 2.5]),
        _FixedWinProbability(),
        list(iter_states(game)),
        {},
    )

    assert plays[0].epa == pytest.approx(1.5)


def test_a_change_of_possession_flips_the_sign():
    """
    +2.1 to the team that just took it over is -2.1 to the team that just
    gave it up. This is where the sign convention lives, and getting it
    backwards would make every turnover look good.
    """
    game = _snaps(("home", 900, {}), ("away", 870, {}))

    plays = play_epa(
        _FixedPoints([1.0, 2.0]),
        _FixedWinProbability(),
        list(iter_states(game)),
        {},
    )

    assert plays[0].epa == pytest.approx(-3.0)


def test_a_scoring_play_is_worth_the_points_not_the_next_snap():
    """
    The drive is over, so there is no next snap to price it from -- and the
    kickoff that follows is not this play's doing.
    """
    game = _snaps(
        ("home", 900, {}),
        ("home", 870, dict(home_score=7, scoring_play=True)),
        ("away", 800, {}),
    )

    plays = _epa(game, [1.0, 5.0, 1.0], clip=INFINITE, weight_power=0.0).plays

    assert plays[1].epa == pytest.approx(7.0 - 5.0)


def test_a_pick_six_is_charged_to_the_offense_that_threw_it():
    """
    The points went up on the other side of the scoreboard, so the same seven
    is minus seven here. `scoring_plays` is signed to the home team and the
    state says who had the ball; this is where the two are reconciled.
    """
    game = _snaps(
        ("home", 900, {}),
        ("home", 870, dict(away_score=7, scoring_play=True)),
        ("home", 800, {}),
    )

    plays = _epa(game, [1.0, 1.0, 1.0], clip=INFINITE, weight_power=0.0).plays

    assert plays[1].epa == pytest.approx(-8.0)


def test_the_half_running_out_is_worth_zero():
    """
    The same absorbing state the fit was trained against: a drive that ends
    with the clock is worth nothing, not whatever the next half produces.
    """
    game = _snaps(
        ("home", 30, dict(period=2)),
        ("home", 800, dict(period=3)),
    )

    plays = play_epa(
        _FixedPoints([2.0, 2.0]),
        _FixedWinProbability(),
        list(iter_states(game)),
        {},
    )

    assert plays[0].epa == pytest.approx(-2.0)


def test_the_last_snap_of_the_game_is_worth_zero():
    game = _snaps(("home", 900, {}))

    plays = play_epa(
        _FixedPoints([1.5]), _FixedWinProbability(), list(iter_states(game)), {}
    )

    assert plays[0].epa == pytest.approx(-1.5)


def test_overtime_is_left_out():
    """
    Regulation only, matching `game_control` and for the same reason: an
    untimed college overtime possession is not a situation regulation
    football has an expected value for, and the fit wasn't given one.
    """
    game = _snaps(
        ("home", 900, {}),
        ("home", 870, {}),
        ("home", 0, dict(period=5)),
    )

    result = _epa(game, [1.0, 2.0, 3.0])

    assert [play.play_id for play in result.plays] == ["p1", "p2"]


# --- The bound ---------------------------------------------------------


def test_the_bound_is_the_only_thing_the_average_sees():
    game = _snaps(("home", 900, {}), ("home", 870, {}), ("home", 840, {}))

    result = _epa(game, [0.0, 9.0, 1.0], clip=5.0, weight_power=0.0)
    unbounded = _epa(game, [0.0, 9.0, 1.0], clip=INFINITE, weight_power=0.0)

    assert result.plays[0].epa == pytest.approx(9.0)
    assert result.plays[0].bounded == pytest.approx(5.0)
    assert result.plays[0].clipped
    assert result.home == pytest.approx(
        float(np.mean([play.bounded for play in result.plays]))
    )
    assert result.home != pytest.approx(unbounded.home)


def test_the_bound_is_symmetric():
    game = _snaps(("home", 900, {}), ("home", 870, {}))

    (big, _) = _epa(game, [9.0, 0.0], clip=5.0, weight_power=0.0).plays

    assert big.bounded == pytest.approx(-5.0)


def test_an_ordinary_play_is_left_exactly_alone():
    game = _snaps(("home", 900, {}), ("home", 870, {}))

    (play, _) = _epa(game, [1.0, 2.2], clip=DEFAULT_CLIP).plays

    assert play.bounded == play.epa
    assert not play.clipped


# --- The weighting -----------------------------------------------------


def test_a_coin_flip_game_weighs_one_and_a_decided_one_weighs_nothing():
    assert competitiveness(0.5) == pytest.approx(1.0)
    assert competitiveness(0.9) == pytest.approx(0.1296)
    assert competitiveness(0.99) == pytest.approx(0.0016, abs=1e-4)
    # Symmetric: it is about the game being in doubt, not about who is ahead.
    assert competitiveness(0.2) == pytest.approx(competitiveness(0.8))


def test_the_weight_falls_away_faster_at_a_higher_power():
    """
    Which is the whole content of the exponent, and why it has a measured
    value rather than an obvious one: it decides how much of a decided game
    still counts.
    """
    three_scores = [competitiveness(0.9, power) for power in (0.5, 1.0, 2.0, 3.0)]

    assert three_scores == sorted(three_scores, reverse=True)
    # A coin flip is the fixed point -- every power leaves it at 1.
    assert {competitiveness(0.5, power) for power in (0.0, 0.5, 1.0, 2.0)} == {1.0}


def test_the_power_turns_the_weighting_off():
    assert competitiveness(0.99, power=0.0) == 1.0
    assert competitiveness(0.0, power=0.0) == 1.0


def test_a_blowout_snap_barely_counts():
    """
    The whole reason for the weighting: a garbage-time drive can't outvote
    the football played while the game was live.
    """
    game = _snaps(("home", 900, {}), ("home", 870, {}), ("home", 840, {}))

    result = _epa(game, [0.0, 0.0, 0.0], probabilities=[0.5, 0.999, 0.999])

    # Two enormous garbage-time plays against one ordinary live one, and the
    # live one still owns the number.
    assert result.plays[0].weight > 50 * result.plays[1].weight


def test_a_kneel_down_needs_no_naming():
    """
    Clock-killing plays are hugely negative in EPA and happen at p above
    0.97, so the weighting disposes of them without a list of play types.
    """
    game = _snaps(("home", 900, {}), ("home", 60, {}), ("home", 30, {}))

    weighted = _epa(game, [1.0, 1.0, -1.5], probabilities=[0.5, 0.99, 0.99])
    unweighted = _epa(
        game, [1.0, 1.0, -1.5], probabilities=[0.5, 0.99, 0.99], weight_power=0.0
    )

    assert weighted.home is not None and unweighted.home is not None
    assert weighted.home > unweighted.home


# --- The identity ------------------------------------------------------


def test_no_clip_and_no_weighting_is_the_plain_mean():
    """
    The escape hatch, and what makes the two adjustments arguable rather than
    baked in: at their identity values this is the unweighted mean of raw
    EPA, exactly.
    """
    game = _snaps(("home", 900, {}), ("home", 870, {}), ("home", 840, {}))

    result = _epa(
        game,
        [0.0, 9.0, 1.0],
        probabilities=[0.99, 0.01, 0.5],
        clip=INFINITE,
        weight_power=0.0,
    )

    raw = [play.epa for play in result.plays]
    assert result.home == pytest.approx(sum(raw) / len(raw))


# --- The two sides -----------------------------------------------------


def test_each_offense_gets_its_own_number():
    game = _snaps(
        ("home", 900, {}),
        ("home", 870, {}),
        ("away", 840, {}),
        ("away", 810, {}),
    )

    result = _epa(game, [1.0, 2.0, 1.0, 2.0], clip=INFINITE, weight_power=0.0)

    assert result.home_plays == 2
    assert result.away_plays == 2
    assert [play.offense_is_home for play in result.plays] == [
        True,
        True,
        False,
        False,
    ]


def test_the_two_sides_do_not_sum_to_anything():
    """
    Where this differs most from `GameControl`: two separate averages over
    two disjoint sets of snaps, in points. Both teams moving the ball is a
    game where both numbers are positive.
    """
    # Both teams drive the length of the field and score, which is the game
    # where "who moved the ball" has the same answer twice.
    game = _snaps(
        ("home", 900, {}),
        ("home", 870, {}),
        ("home", 840, dict(home_score=7, scoring_play=True)),
        ("away", 800, dict(home_score=7)),
        ("away", 770, dict(home_score=7)),
        ("away", 740, dict(home_score=7, away_score=7, scoring_play=True)),
    )

    result = _epa(game, [0.0, 3.0, 5.0, 0.0, 3.0, 5.0], clip=INFINITE, weight_power=0.0)

    assert result.home is not None and result.away is not None
    assert result.home > 0.0 and result.away > 0.0
    assert result.net == pytest.approx(result.home - result.away)


def test_the_effective_sample_is_reported():
    """
    The honest part of the number, in the way `GameControl.seconds` is: an
    average over a rout covers much less than its play count says.
    """
    game = _snaps(("home", 900, {}), ("home", 870, {}))

    live = _epa(game, [1.0, 1.0], probabilities=[0.5, 0.5])
    rout = _epa(game, [1.0, 1.0], probabilities=[0.99, 0.99])

    assert live.home_plays == rout.home_plays == 2
    assert live.home_weight == pytest.approx(2.0)
    assert rout.home_weight < 0.1


def test_a_team_with_no_snaps_has_no_number():
    """
    None rather than 0.0, which would read as a team that played exactly to
    expectation rather than one that never had the ball.
    """
    game = _snaps(("home", 900, {}), ("home", 870, {}))

    result = _epa(game, [1.0, 1.0])

    assert result.away is None
    assert result.away_plays == 0
    assert result.net is None


def test_a_game_with_no_snaps_has_nothing():
    game = _game([make_play(down=None, offense_team_id=None, yardline=None)])

    result = _epa(game, [])

    assert (result.home, result.away, result.plays) == (None, None, [])


def test_a_season_adds_up_as_a_weighted_sum():
    """
    The rule `EpaPerPlay` documents for combining games: weight each game's
    number by its own `home_weight`. Averaging the two numbers instead would
    let a rout, whose weight is nearly all gone, count as much as a game that
    was live throughout.
    """
    live = _epa(
        _snaps(
            ("home", 900, {}),
            ("home", 870, {}),
            ("home", 840, dict(home_score=7, scoring_play=True)),
        ),
        [0.0, 1.0, 3.0],
        probabilities=[0.5, 0.5, 0.5],
    )
    rout = _epa(
        _snaps(("home", 900, {}), ("home", 870, {})),
        [0.0, 4.0],
        probabilities=[0.99, 0.99],
    )

    combined = (live.home * live.home_weight + rout.home * rout.home_weight) / (
        live.home_weight + rout.home_weight
    )

    pooled = live.plays + rout.plays
    weight = sum(play.weight for play in pooled)
    assert combined == pytest.approx(
        sum(play.weight * play.bounded for play in pooled) / weight
    )
    # And the mean of the two numbers is a different, worse answer.
    assert combined != pytest.approx((live.home + rout.home) / 2)


def test_epa_per_play_walks_the_game_itself():
    """`epa_per_play` and the `_from_states` half agree, which is what lets a
    backend use the one that doesn't walk the game twice."""
    game = _snaps(("home", 900, {}), ("home", 870, {}), ("away", 840, {}))
    points, probabilities = [1.0, 2.0, 1.0], [0.5, 0.6, 0.4]

    walked = epa_per_play(
        _FixedPoints(points), _FixedWinProbability(probabilities), game
    )

    assert walked == _epa(game, points, probabilities=probabilities)
