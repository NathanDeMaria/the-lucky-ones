"""
EPA per play: how much a team moved the ball's value, a snap at a time.

`lucky_ones.points` prices a situation. The difference between the price
before a snap and the price after it is that snap's expected points added,
and it is the one number in football that is about a play rather than about a
game:

    EPA = (what the situation is worth now) - (what it was worth at the snap)

signed for the team that had the ball. A 3-yard gain on third and 8 is
negative; the punt that follows is only slightly negative; an interception is
worth about as much as a touchdown, in the other direction. Averaged over a
game it is the closest thing play-by-play offers to "how well did this team
play", and unlike `game_control` it doesn't care who won.

Averaging is where the care goes, and there are two problems with a plain
mean that this module fixes rather than leaves to the caller:

- **A handful of plays own the distribution.** EPA has fat tails -- most snaps
  are worth a fraction of a point, a pick-six is worth eight, and one thrown
  in the red zone eleven -- so a game's mean is largely a report of whether
  one enormous play happened. `clip` bounds each play's contribution.
- **Blowouts aren't football.** A team up 28 in the fourth runs into the line
  and kneels; a team down 28 throws deep on every down against a defense that
  has stopped covering. Both post EPA per play that measures the score rather
  than the team. Each play is weighted by how much its game was still in
  doubt, which is exactly `4p(1-p)` on the win probability at the snap.

Both are knobs, both are keyword arguments at every call site, and both are
off at their identity values -- `clip=inf, weight_power=0.0` is the plain
unweighted mean of raw EPA, which is what `epa_test` checks. That is the same
arrangement `lucky_ones.luck` gives `retained`, and for the same reason: the
adjustment should be arguable at the point of use, not by editing this file.

What this is not is a rating. It is a game's worth of snaps, and a game's
worth of snaps is a noisy estimate of anything -- see `EpaPerPlay` for how to
add games up, which is not by averaging the numbers it returns.
"""

from typing import Mapping, NamedTuple, Sequence, overload

import numpy as np

from .game import GamePlays
from .model import WinProbabilityModel
from .points import ExpectedPointsModel, scoring_plays
from .state import GameState, iter_states

DEFAULT_CLIP = 5.0
"""
The most expected points one play is allowed to contribute, either way.

Measured rather than picked. The 99th percentile of |EPA| is 5.03 to 5.42
across all twenty-four league-seasons of NFL and NCAAFB 2014-2025, so 5.0
bounds about one snap in a hundred -- 1.02% to 1.41%, season by season -- and
leaves the other ninety-nine exactly as they are. `train.py bounds` is the
measurement.

The reason to believe the number isn't the percentile, though -- it is that
the football agrees with it. Ask a shipped fit what the biggest ordinary
snaps are worth and it says +5.07 for a 60-yard touchdown from your own 40
and -5.12 for an interception at your own 25 returned to the spot. Those are
the ceiling of what a team does on purpose, they happen a few times a game,
and a metric that flinches at them is measuring something else. The clip sits
on them to within a tenth of a point, so what it truncates is the compound
play instead: the same interception returned for a touchdown, at -8.11, which
is two events the box score charges to one snap.

Which is also why 4 was the wrong first guess. It is the 97.5th percentile
and bites about one snap in forty, so it truncates the ordinary deep
interception and the ordinary long touchdown too.

Why bound at all, given the tail is real: a game is only ~130 snaps. At its
full -11 a red-zone pick-six moves a team's average by 0.08, which is most of
the gap between a good offense and a bad one, off one play. Bounded it moves
it by 0.04. The play still dominates the game; it just doesn't get to be three
plays.
"""

DEFAULT_WEIGHT_POWER = 1.0
"""
The exponent on `competitiveness`. 1.0 weights by it, 0.0 turns it off.

Between them, a power below 1 keeps more of the blowout and a power above 1
throws away more of it. There is no measurement that settles this the way
`DEFAULT_CLIP` is settled -- it is a statement about what you want the number
to mean -- so 1.0 is the unexaggerated choice: the weight *is* the variance
of the game's outcome at that snap, with nothing done to it.
"""


@overload
def competitiveness(win_probability: float, power: float = 1.0) -> float: ...


@overload
def competitiveness(win_probability: np.ndarray, power: float = 1.0) -> np.ndarray: ...


def competitiveness(win_probability, power=1.0):
    """
    How much the game was still in doubt at a snap, on a 0-to-1 scale.

    Takes a float or an array of them and gives back the same, because it is
    arithmetic and nothing else -- a caller weighting a season's worth of
    snaps at once shouldn't have to reimplement the formula to do it, and
    `validate.py` is that caller.

    `4p(1-p)`, which is the variance of the game's outcome at that moment,
    scaled to peak at 1. A coin-flip game weighs 1.00, a three-score game
    (p = 0.9) weighs 0.36, a decided one (p = 0.99) weighs 0.04.

    Chosen over the usual garbage-time *filter* -- drop everything outside
    0.05 to 0.95 -- because a filter puts a cliff in the middle of the fourth
    quarter, and two teams whose games hovered on either side of it get
    incomparable numbers off comparable football. This fades instead, and it
    fades on a quantity that already means something rather than on a curve
    someone shaped.

    It also disposes of the kneel-downs and the clock-killing runs without
    ever naming them: they happen at p above 0.97, where they weigh about
    nothing. There is no list of play types here and there doesn't need to
    be.
    """
    return (4.0 * win_probability * (1.0 - win_probability)) ** power


class PlayEPA(NamedTuple):
    """One snap's expected points added, and what it counted for."""

    play_id: str
    play_number: int

    offense_is_home: bool
    """Which team the number belongs to. EPA is always the offense's."""

    expected_points: float
    """What the situation was worth to the offense at the snap."""

    epa: float
    """
    The raw difference, unbounded and unweighted.

    Kept as well as `bounded` so a caller can see which plays the bound
    actually bit on, and so that `train.py bounds` can measure the
    distribution through the same code path that averages it.
    """

    bounded: float
    """`epa` clipped to +/- the `clip` in force. What the average is over."""

    win_probability: float
    """The home team's, at the snap -- what `weight` is computed from."""

    weight: float
    """`competitiveness(win_probability, weight_power)`."""

    @property
    def clipped(self) -> bool:
        """Whether the bound moved this play."""
        return self.bounded != self.epa


class EpaPerPlay(NamedTuple):
    """
    A game's EPA per play, one number per offense.

    `home` is what the home team's offense averaged, `away` what the away
    team's did -- so `home` is also the away defense's number with the sign
    kept, and reporting a defense means reading the other side of the same
    pair. `net` is the home team's margin, which is the single number a table
    sorts on.

    Unlike `GameControl` these do not sum to anything: they are two separate
    averages over two disjoint sets of snaps, in points, and both can be
    positive in a game where everybody moved the ball.

    None means there was nothing to average -- no snaps for that offense, or
    (only reachable with the weighting on) a team whose every snap came with
    the game already decided. Not 0.0, which would read as a team that played
    exactly averagely.

    **Adding games up is a weighted sum, not a mean of means.** A game
    contributes `home * home_weight` to the numerator and `home_weight` to the
    denominator; averaging the per-game numbers instead lets a 12-snap
    weather-shortened game count as much as a full one, and lets a blowout --
    whose weight is almost all gone -- count as much as a game that was live
    throughout. That is what the weights are on the tuple for.
    """

    home: float | None
    away: float | None

    home_plays: int
    """Snaps the home offense ran, before weighting."""

    away_plays: int

    home_weight: float
    """
    The effective sample behind `home`: the sum of the play weights.

    The honest part of the number, in the way `GameControl.seconds` is. 60 of
    70 snaps means a game that was live throughout; 12 of 70 means an average
    over the first quarter and a half of a rout, and should be read as one.
    """

    away_weight: float

    plays: list[PlayEPA]
    """Every snap in the two averages, in game order."""

    @property
    def net(self) -> float | None:
        """
        `home - away`: how much better the home team was per play.

        None if either side has nothing to average, because a margin against
        a missing number isn't a margin.
        """
        if self.home is None or self.away is None:
            return None
        return self.home - self.away


def epa_per_play(
    points: ExpectedPointsModel,
    win_probability: WinProbabilityModel,
    game: GamePlays,
    *,
    clip: float = DEFAULT_CLIP,
    weight_power: float = DEFAULT_WEIGHT_POWER,
) -> EpaPerPlay:
    """
    Both offenses' EPA per play over `game`.

    Two models, because the metric is two questions: `points` says what each
    snap was worth, and `win_probability` says how much the game it happened
    in was still worth playing. Neither is retrained by any of this -- both
    are read, exactly as `lucky_ones.luck` reads the win probability fit.

    Regulation only, matching `game_control` and for the same reason:
    expected points has no meaning in an untimed college overtime possession,
    and `lucky_ones.training.build_expected_points_set` doesn't fit one there.
    """
    return epa_per_play_from_states(
        points,
        win_probability,
        list(iter_states(game)),
        scoring_plays(list(game.plays)),
        clip=clip,
        weight_power=weight_power,
    )


def epa_per_play_from_states(
    points: ExpectedPointsModel,
    win_probability: WinProbabilityModel,
    states: Sequence[GameState],
    scored: Mapping[str, float],
    *,
    clip: float = DEFAULT_CLIP,
    weight_power: float = DEFAULT_WEIGHT_POWER,
) -> EpaPerPlay:
    """
    The same numbers for states a caller already walked the game to build,
    and the scoring plays it already read off the score columns
    (`lucky_ones.points.scoring_plays`).

    The backend path, for the same reason `curve_from_states` and
    `adjusted_curve_from_states` exist: two `predict` calls for the whole
    game rather than one per snap.
    """
    plays = play_epa(
        points,
        win_probability,
        states,
        scored,
        clip=clip,
        weight_power=weight_power,
    )
    home = [play for play in plays if play.offense_is_home]
    away = [play for play in plays if not play.offense_is_home]
    return EpaPerPlay(
        home=_weighted_mean(home),
        away=_weighted_mean(away),
        home_plays=len(home),
        away_plays=len(away),
        home_weight=float(sum(play.weight for play in home)),
        away_weight=float(sum(play.weight for play in away)),
        plays=plays,
    )


def play_epa(
    points: ExpectedPointsModel,
    win_probability: WinProbabilityModel,
    states: Sequence[GameState],
    scored: Mapping[str, float],
    *,
    clip: float = DEFAULT_CLIP,
    weight_power: float = DEFAULT_WEIGHT_POWER,
) -> list[PlayEPA]:
    """
    Every regulation snap's EPA, before anything is averaged.

    Where a play *ended* is one of three things, in this order:

    1. **It scored.** The points are the answer, and no model is consulted:
       `scored` says what went on the board and the state says who had the
       ball, so a pick-six is minus seven to the offense that threw it. This
       comes first because a scoring play has no next snap to be priced from
       -- the drive is over, and the ensuing kickoff is not this play's doing.
    2. **A snap followed it, in the same half.** That snap's expected points,
       negated if the ball changed hands: what is worth +2.1 to the team that
       just took it over is worth -2.1 to the team that just gave it up. This
       is the ordinary case and it is where the sign convention lives.
    3. **Neither.** The half ran out with nobody scoring, which is worth
       exactly zero -- the same `NO_SCORE` the fit was trained against.

    Two `predict` calls for the whole game, one per model. Both are matrix
    multiplies, and a per-play loop over a season is the difference between
    seconds and minutes.
    """
    regulation = [state for state in states if not state.is_overtime]
    if not regulation:
        return []
    expected = points.predict(regulation)
    probabilities = win_probability.predict(regulation)

    plays: list[PlayEPA] = []
    for index, state in enumerate(regulation):
        after = regulation[index + 1] if index + 1 < len(regulation) else None
        home_points = scored.get(state.play_id)
        if home_points is not None:
            ended = home_points if state.offense_is_home else -home_points
        elif after is not None and after.half == state.half:
            ended = float(expected[index + 1])
            if after.offense_is_home != state.offense_is_home:
                ended = -ended
        else:
            ended = 0.0
        raw = ended - float(expected[index])
        probability = float(probabilities[index])
        plays.append(
            PlayEPA(
                play_id=state.play_id,
                play_number=state.play_number,
                offense_is_home=state.offense_is_home,
                expected_points=float(expected[index]),
                epa=raw,
                bounded=max(-clip, min(clip, raw)),
                win_probability=probability,
                weight=competitiveness(probability, weight_power),
            )
        )
    return plays


def _weighted_mean(plays: Sequence[PlayEPA]) -> float | None:
    """
    The average this module reports, or None when there's nothing behind it.

    None rather than 0.0 for a team with no weight left, because 0.0 is a
    real EPA per play -- a team that played exactly to expectation -- and a
    metric that says that about a game it couldn't measure is worse than one
    that says nothing.
    """
    total = sum(play.weight for play in plays)
    if total <= 0.0:
        return None
    return float(sum(play.weight * play.bounded for play in plays) / total)
