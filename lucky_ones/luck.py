"""
Game control with the coin flips taken out.

`lucky_ones.curve.game_control` measures what happened. That is the honest
answer to "who controlled this game", and it is also the answer that hands a
team full credit for a fumble that bounced their way -- an outcome nobody on
the field decided. This module answers the neighbouring question: *how much
of the game would this team have controlled if the fifty-fifty balls had gone
fifty-fifty?*

The shape of it is three steps:

1. **Find the coin flips.** `find_lucky_plays` reads `Play.text` for the
   handful of plays whose result was decided by a bounce -- a fumble, and an
   interception off a tipped ball. See `LuckKind` for the whole list and why
   it is short.
2. **Ask the model what the other branch was worth.** Every play here has
   exactly two outcomes that matter: the ball changed hands or it didn't. The
   one that didn't happen is a `GameState` we can build (`_counterfactual`)
   and the model can price, so the counterfactual is the model's own number
   rather than a constant someone picked.
3. **Replace the swing with the average of the two.** The play's win
   probability move becomes `retained * what happened + (1 - retained) * the
   other branch`, and every later point on the curve carries the difference
   forward.

`retained` is a probability: how often the outcome that happened is the one
that happens. A fumble is recovered by either side about half the time, so
0.5 -- half the credit for the recovery, and half of the branch where it went
the other way. A tipped pass is intercepted a good deal less often than it
falls incomplete, so a tipped interception keeps 0.25 of its swing.

Three things this deliberately is not:

- **A prediction.** The adjusted curve does not end at 0 or 1 the way the
  real one does. The game had a winner; this is a counterfactual, and its
  last point is where the game *would* have stood.
- **Symmetric in what it can see.** A fumble the offense recovered is in the
  play text, so the team that got away with one is charged for it. A pass
  that was tipped and *not* intercepted mostly isn't in the text at all, so
  it can't be. That is a real gap, and the reason it isn't fatal is that a
  near-miss's realized swing is already near zero -- what's missing is the
  discount on the *other* side of the same coin, which is small.
- **An adjustment to the model.** The fit is untouched. This reweights the
  curve that fit produces; retrain nothing.
"""

from enum import StrEnum
from typing import Iterable, Mapping, NamedTuple, Sequence

import numpy as np

from .curve import CurvePoint, GameControl, curve_from_states, game_control
from .game import GamePlays
from .model import WinProbabilityModel
from .plays import Play
from .state import GameState, iter_states


class LuckKind(StrEnum):
    """
    The kinds of play this discounts.

    A short list on purpose. Every entry has to clear two bars: the play text
    says it happened, and the outcome that didn't happen is one this module
    can build a `GameState` for. Both branches of a fumble and of a tipped
    pass are "one team has the ball here or the other does", which
    `_counterfactual` builds exactly.

    What that rules out, and why each is left in the realized number rather
    than guessed at:

    - **Muffed punts and kickoff fumbles.** `iter_states` keeps scrimmage
      snaps only, so a kickoff isn't on the curve to adjust, and a punt's
      counterfactual isn't "the punting team keeps it on the next down".
    - **Blocked kicks.** The block is a play someone made; only the bounce
      after it is luck, and the two aren't separable from the text.
    - **A field goal off the upright.** No branch to build: the counterfactual
      to a 51-yarder that went in is a 51-yarder that didn't, which is the
      same state with three fewer points and a different spot.
    - **Untipped interceptions.** A throw into coverage is a decision, not a
      bounce.
    """

    FUMBLE_LOST = "fumble_lost"
    """A fumble the other team came up with."""

    FUMBLE_KEPT = "fumble_kept"
    """A fumble the offense recovered -- the one that got away with it."""

    TIPPED_INTERCEPTION = "tipped_interception"
    """An interception off a tipped, batted or deflected ball."""


DEFAULT_RETAINED: Mapping[LuckKind, float] = {
    # Loose-ball recovery is close enough to a coin flip that the NFL's
    # aggregate rate is the coin: neither side recovers its own fumbles at a
    # rate that holds up year over year. Same number both ways, because it is
    # the same coin seen from the two sides.
    LuckKind.FUMBLE_LOST: 0.5,
    LuckKind.FUMBLE_KEPT: 0.5,
    # A ball off a hand or a helmet falls incomplete far more often than a
    # defender comes down with it, so an interception is the unlikely branch
    # and keeps the smaller share of its swing.
    LuckKind.TIPPED_INTERCEPTION: 0.25,
}
"""
How much of each kind's swing is real, as `P(the outcome that happened)`.

Estimates, not fits -- there is no labelled set of counterfactual fumbles to
fit them on. They are a keyword argument everywhere they are used so that a
caller who disagrees can say so at the call site rather than by editing this
file, and so that `retained=1.0` for every kind reproduces the unadjusted
curve exactly, which is what `luck_test` checks.
"""


class LuckyPlay(NamedTuple):
    """One play whose result was decided by a bounce."""

    play_id: str
    play_number: int
    kind: LuckKind

    retained: float
    """`P(this outcome)` -- the weight its realized swing keeps."""

    changed_possession: bool
    """
    Whether the ball actually changed hands on it.

    What picks the branch to price: the counterfactual to a fumble that was
    lost is the offense keeping it, and the counterfactual to one that was
    recovered is the other team taking over. Read from `Play.is_turnover`
    rather than from the kind, because the schema's own column is a better
    witness than a keyword in a sentence.
    """


class AdjustedCurve(NamedTuple):
    """
    A game's curve both ways, and what moved it.

    Both curves rather than just the adjusted one because every use of this
    wants the pair: a chart draws them together, and the number worth
    reporting is the gap between their two `game_control`s. Computing the
    realized curve is most of the work here, so handing it back costs nothing
    and saves the caller a second `predict` over the whole game.
    """

    points: list[CurvePoint]
    """The adjusted curve. Same points, same labels, reweighted probability."""

    realized: list[CurvePoint]
    """What actually happened -- identical to `win_probability_curve`."""

    lucky_plays: list[LuckyPlay]
    """
    The plays that were actually discounted, in game order.

    A subset of what `find_lucky_plays` found: a lucky play only counts if it
    is a snap on the curve *and* something comes after it, since the last
    snap of a game has no later probability to move.
    """


def luck_adjusted_curve(
    model: WinProbabilityModel,
    game: GamePlays,
    *,
    retained: Mapping[LuckKind, float] = DEFAULT_RETAINED,
) -> AdjustedCurve:
    """The game's curve, with each coin flip replaced by its average."""
    return adjusted_curve_from_states(
        model, list(iter_states(game)), find_lucky_plays(game.plays, retained=retained)
    )


def luck_adjusted_game_control(
    model: WinProbabilityModel,
    game: GamePlays,
    *,
    retained: Mapping[LuckKind, float] = DEFAULT_RETAINED,
) -> GameControl | None:
    """
    `game_control` over the adjusted curve: the share of the game each team
    would have controlled with the bounces split evenly.

    None on the same games `game_control` is None on -- no elapsed regulation
    clock to weight by. A game with no lucky plays in it returns exactly the
    unadjusted number, not something near it.
    """
    return game_control(luck_adjusted_curve(model, game, retained=retained).points)


def adjusted_curve_from_states(
    model: WinProbabilityModel,
    states: Sequence[GameState],
    lucky_plays: Iterable[LuckyPlay],
) -> AdjustedCurve:
    """
    The adjusted curve for states and lucky plays a caller already has.

    The one a backend calls, for the same reason `curve_from_states` exists:
    it walked the game once already. Two `predict` calls for the whole game --
    one for the curve, one for every counterfactual at once -- rather than one
    per lucky play.

    Adjusting is a walk over the curve accumulating deltas:

        adjusted[0]   = realized[0]
        adjusted[i+1] = adjusted[i] + (logit(target) - logit(realized[i]))

    where `target` is `realized[i + 1]` for an ordinary snap, and the blend of
    the two branches for a lucky one. Two scales, each for a reason. The
    *blend* is in probability space, because it is an expectation over
    outcomes and that is where an expectation lives. The *accumulation* is in
    log-odds, because deltas there are the model's own additive scale -- the
    coefficients are log-odds -- and because a shifted curve stays inside
    (0, 1) on its own, with no clamp to bend the endgame back into range.

    The shift persists to the final whistle, which is the point: a fumble
    returned for a touchdown in the first quarter is still on the scoreboard
    in the fourth, so a curve that discounts the recovery has to keep
    discounting it.
    """
    realized = curve_from_states(model, states)
    if len(realized) < 2:
        # One snap has no delta to adjust, and no elapsed time either -- the
        # curve is the curve.
        return AdjustedCurve(points=realized, realized=realized, lucky_plays=[])

    by_play_id = {lucky.play_id: lucky for lucky in lucky_plays}
    # `[:-1]` because a lucky play on the last snap moves nothing after it.
    applied = [
        (index, by_play_id[point.play_id])
        for index, point in enumerate(realized[:-1])
        if point.play_id in by_play_id
    ]
    if not applied:
        return AdjustedCurve(points=realized, realized=realized, lucky_plays=[])

    branches = model.predict(
        [
            _counterfactual(states[index], lucky.changed_possession)
            for index, lucky in applied
        ]
    )
    counterfactual = {
        index: float(probability) for (index, _), probability in zip(applied, branches)
    }
    lucky_at = dict(applied)

    logit = _logit(realized[0].home_win_probability)
    points = [realized[0]]
    for index, point in enumerate(realized[:-1]):
        after = realized[index + 1].home_win_probability
        lucky = lucky_at.get(index)
        if lucky is not None:
            after = (
                lucky.retained * after + (1.0 - lucky.retained) * counterfactual[index]
            )
        logit += _logit(after) - _logit(point.home_win_probability)
        points.append(
            realized[index + 1]._replace(home_win_probability=_sigmoid(logit))
        )
    return AdjustedCurve(
        points=points,
        realized=realized,
        lucky_plays=[lucky for _, lucky in applied],
    )


# Words that say the ball was in the air off something other than the
# receiver's hands. ESPN writes these into the play text in the clause about
# the interception ("intercepted ... (tipped by ...)"), which is why the check
# below is "an interception *and* one of these" rather than either alone.
_TIP_MARKERS: tuple[str, ...] = ("tipped", "tip by", "deflect", "batted")


def find_lucky_plays(
    plays: Iterable[Play],
    *,
    retained: Mapping[LuckKind, float] = DEFAULT_RETAINED,
) -> list[LuckyPlay]:
    """
    Every play in `plays` whose result was decided by a bounce, in order.

    Classification is substring matching over `Play.text`, which is what
    there is: ESPN ships a sentence, and the manner of a play is in the
    sentence or nowhere. That makes this a filter with a false-negative rate
    rather than a parser -- a fumble phrased in a way these words miss is
    simply not adjusted, which leaves the realized number, which is the safe
    direction to be wrong in.

    A play with no text is skipped, and so is one whose text says "no play":
    a fumble wiped out by a penalty never happened, and its win probability
    swing is a penalty's, not a bounce's.

    Whether the ball changed hands comes from `is_turnover` rather than the
    text. A fumble whose `is_turnover` is null is dropped instead of guessed
    at -- getting that backwards prices the wrong branch, which is worse than
    not adjusting the play at all.
    """
    lucky = []
    for play in plays:
        classified = _classify(play)
        if classified is None:
            continue
        kind, changed_possession = classified
        lucky.append(
            LuckyPlay(
                play_id=play.play_id,
                play_number=play.play_number,
                kind=kind,
                retained=retained.get(kind, DEFAULT_RETAINED[kind]),
                changed_possession=changed_possession,
            )
        )
    return lucky


def _classify(play: Play) -> tuple[LuckKind, bool] | None:
    """The play's `LuckKind` and whether it turned the ball over, or None."""
    text = (play.text or "").lower()
    if not text or "no play" in text:
        return None
    if "intercept" in text:
        if any(marker in text for marker in _TIP_MARKERS):
            return LuckKind.TIPPED_INTERCEPTION, True
        # An untipped interception is a decision, not a bounce. Returning
        # here rather than falling through matters: an interception returned
        # and fumbled has both words in it, and the interception is the play.
        return None
    if "fumble" in text:
        if play.is_turnover is None:
            return None
        return (
            LuckKind.FUMBLE_LOST if play.is_turnover else LuckKind.FUMBLE_KEPT
        ), bool(play.is_turnover)
    return None


def _counterfactual(state: GameState, changed_possession: bool) -> GameState:
    """
    The state the branch that didn't happen would have led to.

    Built from the snap itself rather than from the next one, which is what
    keeps it a counterfactual: the realized play's yards and any points it put
    on the board belong to the branch that happened, and would be wrong on the
    other one -- a fumble returned for a touchdown didn't score in the branch
    where it was recovered by the offense.

    So the ball is at the spot of the snap, and the only thing that moves is
    who has it:

    - **It changed hands, so the other branch is keeping it.** Same team, same
      spot, next down. On fourth down that's a first: a fourth-down snap the
      offense held onto is one they converted.
    - **It didn't, so the other branch is losing it.** The other team, first
      and ten, at the mirror of the same yardline -- `yardline` is measured
      from the offense's own goal line, so possession flipping flips it.

    An approximation, and the shape of the error is worth knowing: it prices
    a turnover at the line of scrimmage, ignoring the yards the play gained
    before the ball came loose and the yards a return added after. Against the
    thing being measured -- who has the ball -- that is small. A 25-yard
    change in field position is worth a few points of win probability; the
    possession itself is worth several times that.
    """
    if changed_possession:
        down = state.down + 1
        if down > 4:
            return state._replace(down=1, distance=10)
        return state._replace(down=down)
    return state._replace(
        offense_is_home=not state.offense_is_home,
        yardline=max(1, min(99, 100 - state.yardline)),
        down=1,
        distance=10,
    )


# Far enough from 0 and 1 that a log is finite, close enough that clipping
# never bites a probability a fit would actually produce. The endgame is where
# it would matter -- `predict` returns genuine 0.9999s in the last minute of a
# blowout -- and 1e-9 leaves those alone.
_EPSILON = 1e-9


def _logit(probability: float) -> float:
    clipped = min(max(probability, _EPSILON), 1.0 - _EPSILON)
    return float(np.log(clipped / (1.0 - clipped)))


def _sigmoid(logit: float) -> float:
    """
    The inverse, written the overflow-free way for the same reason
    `model._sigmoid` is: a three-score lead in the last minute is a genuinely
    enormous log-odds, and `1 / (1 + exp(-x))` returns garbage for it.
    """
    exponentiated = float(np.exp(-abs(logit)))
    if logit >= 0:
        return 1.0 / (1.0 + exponentiated)
    return exponentiated / (1.0 + exponentiated)
