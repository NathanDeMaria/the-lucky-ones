"""
`GameState` to numbers.

One module, on purpose: the feature list is the modelling decision in this
package, and a fitted model is only meaningful next to the exact
transformation that produced its inputs. `FEATURE_NAMES` is the contract --
`LogisticWinProbability` saves it alongside its coefficients and refuses to
load a fit whose features don't match the code doing the loading.

Two lists, because there are two models and they are asking different
questions of the same snap:

- `FEATURE_NAMES` is win probability's. Everything in it is oriented to the
  *home* team, because that is what the model predicts, and the score is in
  it twice -- a lead, and a lead against the time left to undo it.
- `EP_FEATURE_NAMES` is expected points'. Everything in it is oriented to
  whoever has the ball, and the score is not in it at all. See
  `to_ep_features` for why that absence is a decision rather than an
  oversight.

They live in one module so that "what does the model see" has one place to
read, and each fit stores the list it was made against so the two can never
be crossed.
"""

from typing import Iterable, Sequence

import numpy as np

from .state import HALF_SECONDS, REGULATION_SECONDS, GameState

FEATURE_NAMES: tuple[str, ...] = (
    "score_margin",
    "margin_per_root_time",
    "fraction_remaining",
    "offense_is_home",
    "down",
    "log_distance",
    "yardline",
    "is_overtime",
)


def to_features(state: GameState) -> np.ndarray:
    """
    One state as a row of `FEATURE_NAMES`, float64.

    What each of them is for:

    `score_margin` alone can't carry the game. A 7-point lead means one thing
    in the first quarter and nearly everything in the last minute, and a
    linear term in margin can only say one of those. `margin_per_root_time`
    is the standard fix -- lead over the square root of the time left, which
    is the shape a random walk's tail probability actually has -- and it's
    what lets a logistic model track a game to its end instead of flattening
    out. The two together let the fit weigh a lead against how much game
    remains to undo it.

    `fraction_remaining` gives the model a level for the clock, so the
    interaction term above isn't also doing duty as the time variable.

    `offense_is_home` is signed (+1 home, -1 away) rather than 0/1, because
    the field-position features below are the offense's and the target is the
    home team's. Signed, the fit can express "having the ball is worth
    something to whoever has it" in one coefficient per feature instead of
    needing an interaction with each.

    `down`, `log_distance` and `yardline` are the drive's situation. Distance
    is logged because the difference between 1 and 5 yards to go matters far
    more than the difference between 20 and 24; `log1p` of a distance floored
    at zero, since the source column can be negative after a penalty enforced
    from behind the spot.

    `is_overtime` is a flag rather than a clock, because `seconds_remaining`
    is pinned to zero there (see `state._seconds_remaining`) and every
    overtime snap would otherwise look like the last play of regulation.
    """
    possession = 1.0 if state.offense_is_home else -1.0
    remaining = float(state.seconds_remaining)
    return np.array(
        [
            float(state.score_margin),
            # +1 so the last second of the game is a large number rather than
            # a division by zero.
            state.score_margin / np.sqrt(remaining + 1.0),
            remaining / REGULATION_SECONDS,
            possession,
            float(state.down),
            np.log1p(max(0, state.distance)),
            # Signed for the same reason as `offense_is_home`: 80 yards from
            # the end zone is good news for the home team or bad, and which
            # one depends on who has the ball.
            possession * state.yardline / 100.0,
            1.0 if state.is_overtime else 0.0,
        ]
    )


def feature_matrix(states: Iterable[GameState]) -> np.ndarray:
    """
    A `(len(states), len(FEATURE_NAMES))` matrix.

    Comes back with the right width even when `states` is empty, so a caller
    that concatenates weeks doesn't have to special-case the week where every
    game was postponed.
    """
    rows = [to_features(state) for state in states]
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)))
    return np.vstack(rows)


def labelled_matrix(
    states: Sequence[GameState], home_won: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray]:
    """
    `(X, y)` for fitting: every state in `states` against the result of the
    game it came from.

    `home_won` is per-state, not per-game -- the caller has already fanned the
    outcome out over the game's states (see `lucky_ones.training`). Keeping
    it that way means this function never has to know how games map to rows,
    which is the part that differs between a training run and a backtest.
    """
    if len(states) != len(home_won):
        raise ValueError(
            f"Got {len(states)} states and {len(home_won)} labels; they have to "
            "line up row for row"
        )
    return feature_matrix(states), np.asarray(home_won, dtype=float)


# Where the field-position spline below is allowed to change slope, as a
# fraction of the field from the offense's target end zone. Five landmarks
# rather than five tuned knots: the red zone's mouth and the goal line's
# approach at one end, midfield in the middle, and the two lines that define
# "backed up" at the other. Football changes what it is doing at each of
# them, which is the only reason to let the curve bend there.
_FIELD_KNOTS: tuple[tuple[str, float], ...] = (
    ("yards_past_opponent_10", 0.10),
    ("yards_past_opponent_20", 0.20),
    ("yards_past_midfield", 0.50),
    ("yards_past_own_20", 0.80),
    ("yards_past_own_10", 0.90),
)

EP_FEATURE_NAMES: tuple[str, ...] = (
    "down_2",
    "down_3",
    "down_4",
    "log_distance",
    "goal_to_go",
    "yards_to_goal",
    *(name for name, _ in _FIELD_KNOTS),
    "fraction_half_remaining",
    "half_remaining_x_yards_to_goal",
)


def to_ep_features(state: GameState) -> np.ndarray:
    """
    One state as a row of `EP_FEATURE_NAMES`, float64 -- what
    `lucky_ones.points` fits expected points on.

    Everything here is the *offense's*, and that is the whole difference from
    `to_features` above. Expected points is a property of a situation, not of
    a team: first and ten on your own 25 is worth what it is worth to whoever
    is standing there, so an EPA computed from it is comparable between the
    two teams in one game and between every game in a season.

    Two things are deliberately not in the list.

    **The score isn't.** Not the margin, not the clock left in the game. A
    team down 28 in the fourth throws on every down and a team up 28 kneels,
    so the score genuinely predicts the next points -- which is exactly why
    it stays out. Putting it in would fold "this game was over" into the
    expected value and leave EPA quietly measuring how far ahead you already
    were. Blowouts are handled where the distortion actually is, in the
    weighting `lucky_ones.epa` puts on each play, and that is a knob a caller
    can turn off and look at. A coefficient isn't.

    **Which team is home isn't.** Home field is worth real points over a
    season, and including it would make the same snap worth more to one side
    than the other -- so a home offense and an away offense could post the
    same EPA per play off different football. The convention every published
    expected points model follows, and the one that keeps the two sides of a
    game on one scale.

    What is in it:

    `down_2`/`down_3`/`down_4` are indicators against first down, not a
    number. Down isn't a quantity -- the drop from third to fourth is nothing
    like the drop from first to second -- and three flags let the fit say so
    in three numbers instead of forcing a line through four points.

    `log_distance` is logged for the reason it is in the win probability
    features: 1 versus 5 yards to go is most of the difference, 20 versus 24
    is none of it.

    `goal_to_go` because the last few yards are their own situation. Second
    and 8 from the 8 is a goal line snap; second and 8 from midfield is a
    normal one, and `distance` alone can't tell them apart.

    `yards_to_goal` and the five `yards_past_*` terms are the field, as a
    linear spline: a slope, plus a change of slope at each of `_FIELD_KNOTS`.
    The field is where the curve bends and a polynomial in it was the first
    thing tried and the wrong one -- a quadratic reads first and ten inside
    your own 10 as *half a point better* than it is (+0.37 against a measured
    -0.16), because it has one bend to spend and the middle of the field
    outvotes both ends for it. The spline gets that snap to within a tenth,
    and a play from there is one of the biggest EPAs in football, so the end
    of the curve is not the part to approximate.

    `fraction_half_remaining` is the clock that matters here -- the half's,
    not the game's (see `GameState.half_seconds_remaining`). Time is what
    stands between a drive and points, and the half is where that runs out.

    `half_remaining_x_yards_to_goal` is the one interaction, and it earns its
    place because the clock running out is not worth the same everywhere on
    the field. With a minute left in a half, a team on its own 10 has nothing
    -- the possession is over and the drive was never going anywhere -- while
    a team on the opponent's 15 still has a field goal. A model with a level
    for the clock and no interaction has to split the difference, and it
    misses both ends the same way: a couple of tenths too generous on the
    first and too stingy on the second. The interaction roughly halves both.
    """
    yards_to_goal = (100.0 - state.yardline) / 100.0
    fraction_half_remaining = state.half_seconds_remaining / HALF_SECONDS
    return np.array(
        [
            1.0 if state.down == 2 else 0.0,
            1.0 if state.down == 3 else 0.0,
            1.0 if state.down >= 4 else 0.0,
            np.log1p(max(0, state.distance)),
            # A distance that reaches the end zone, however it was spelled --
            # ESPN has no goal-to-go column, and the yardline says it anyway.
            1.0 if state.distance >= 100 - state.yardline else 0.0,
            yards_to_goal,
            *(max(0.0, yards_to_goal - knot) for _, knot in _FIELD_KNOTS),
            fraction_half_remaining,
            fraction_half_remaining * yards_to_goal,
        ]
    )


def ep_feature_matrix(states: Iterable[GameState]) -> np.ndarray:
    """
    A `(len(states), len(EP_FEATURE_NAMES))` matrix.

    Right width when empty, for the same reason `feature_matrix` is: a caller
    concatenating weeks shouldn't special-case the empty one.
    """
    rows = [to_ep_features(state) for state in states]
    if not rows:
        return np.empty((0, len(EP_FEATURE_NAMES)))
    return np.vstack(rows)
