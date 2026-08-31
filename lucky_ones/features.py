"""
`GameState` to numbers.

One module, on purpose: the feature list is the modelling decision in this
package, and a fitted model is only meaningful next to the exact
transformation that produced its inputs. `FEATURE_NAMES` is the contract --
`LogisticWinProbability` saves it alongside its coefficients and refuses to
load a fit whose features don't match the code doing the loading.
"""

from typing import Iterable, Sequence

import numpy as np

from .state import REGULATION_SECONDS, GameState

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
