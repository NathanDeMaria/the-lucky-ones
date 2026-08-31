"""
A game's win probability over time, and what it adds up to.

Two things live here, and they're the two things a consumer actually wants
out of this package:

- `win_probability_curve`, the series a chart draws -- one point per snap,
  carrying enough alongside the probability (period, clock, score) that the
  caller doesn't need the plays again to label an axis or fill a tooltip.
- `game_control`, one number per team for how much of the game they spent
  ahead of it: the average win probability, weighted by how long each one
  was on the board.

The weighting is what makes the second worth having. An unweighted mean over
snaps counts a two-minute drill's fifteen plays the same as a quarter of
grinding, so a team that trailed all game and lost on a frantic last drive
comes out looking like it was in it. Weighting by elapsed clock says what
"most of the game" means.
"""

from typing import NamedTuple, Sequence

from .game import GamePlays
from .model import WinProbabilityModel
from .state import GameState, iter_states


class CurvePoint(NamedTuple):
    """
    One snap of the curve.

    Carries the state's own labels rather than a reference back to it,
    because the shape a chart wants and the shape a model wants aren't the
    same: down and distance don't belong on an axis, and `home_score` does.
    """

    play_id: str
    play_number: int
    period: int
    clock_seconds: int
    """Left in `period`."""

    seconds_remaining: int
    """Left in regulation. 0 in overtime -- see `state._seconds_remaining`."""

    home_score: int
    away_score: int
    home_win_probability: float


class GameControl(NamedTuple):
    """
    Average win probability per team, weighted by time.

    `home` and `away` sum to 1, so one number is really enough -- both are
    here because a caller displaying "who controlled this game" wants to name
    the team, and deriving the other side at each call site is how a `1 -` in
    the wrong place gets shipped.

    `seconds` is how much game time the average covers, which is the honest
    part of the number: it's regulation only (see `game_control`), so a game
    that went to overtime reports less than a full 3600.
    """

    home: float
    away: float
    seconds: int


def win_probability_curve(
    model: WinProbabilityModel, game: GamePlays
) -> list[CurvePoint]:
    """
    The home team's win probability at every snap of `game`.

    Empty for a game with no scrimmage plays, which is what a game ESPN has
    no play-by-play for looks like once `iter_states` has dropped the
    kickoffs and the clock stoppages.
    """
    return curve_from_states(model, list(iter_states(game)))


def curve_from_states(
    model: WinProbabilityModel, states: Sequence[GameState]
) -> list[CurvePoint]:
    """
    The same curve, for states a caller already has.

    This is the one a backend serving a game calls: it has walked the game
    once to build states, and walking it again to draw the curve would double
    the work for nothing.

    One `model.predict` for the whole game rather than one per play -- the
    baseline's predict is a matrix multiply, and a per-play loop over a season
    is the difference between seconds and minutes.
    """
    if not states:
        return []
    probabilities = model.predict(states)
    return [
        CurvePoint(
            play_id=state.play_id,
            play_number=state.play_number,
            period=state.period,
            clock_seconds=state.clock_seconds,
            seconds_remaining=state.seconds_remaining,
            home_score=state.home_score,
            away_score=state.away_score,
            home_win_probability=float(probability),
        )
        for state, probability in zip(states, probabilities)
    ]


def game_control(points: Sequence[CurvePoint]) -> GameControl | None:
    """
    Time-weighted average win probability: the fraction of the game each team
    spent winning it.

    Each point is weighted by how long its situation stood -- the clock from
    that snap to the next one, and for the last snap, the clock from it to
    the end of regulation. A 50/50 game comes out at 0.5; a wire-to-wire
    blowout approaches 1.

    Read it as "share of the game controlled", not as a win probability. 0.68
    doesn't mean the home team was ever 68% to win; it means that averaged
    over the sixty minutes, that's where the model had them.

    Regulation only. `seconds_remaining` is pinned to zero in overtime,
    because college overtime has no clock at all to weight by -- so overtime
    snaps have no duration and drop out. `GameControl.seconds` reports what
    the number actually covers.

    None when there's no elapsed time to weight by: an empty curve, or a game
    whose only snaps were in overtime. Not an error -- a caller scoring a
    week has to expect a game the play-by-play barely covered -- but not a
    0.5 either, which would read as a genuinely even game.
    """
    weights = _durations(points)
    total = sum(weights)
    if total <= 0:
        return None
    home = (
        sum(
            weight * point.home_win_probability
            for weight, point in zip(weights, points)
        )
        / total
    )
    return GameControl(home=home, away=1.0 - home, seconds=total)


def _durations(points: Sequence[CurvePoint]) -> list[int]:
    """
    How long each snap's situation stood, in seconds of regulation clock.

    Clamped at zero. Two snaps can share a clock reading -- a penalty and the
    replay of the down, a spike, anything the clock didn't run on -- and
    ESPN's clock occasionally goes backwards between plays. Both are zero
    seconds of game time, not negative ones, and a negative weight would pull
    the average the wrong way rather than just contributing nothing.
    """
    durations = []
    for index, point in enumerate(points):
        next_remaining = (
            points[index + 1].seconds_remaining if index + 1 < len(points) else 0
        )
        durations.append(max(0, point.seconds_remaining - next_remaining))
    return durations
