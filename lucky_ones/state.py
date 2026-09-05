"""
The situation the models are a function of, and the results they're
predicting.

A `GameState` is everything a model conditions on at one snap, on one scale,
with no Nones left in it -- both models, which read different parts of it:
win probability takes the score and the clock left in the *game*, expected
points takes the down, the field and the clock left in the *half* (see
`half_seconds_remaining`). Getting from a `Play` to one is mostly
bookkeeping, except for a detail that decides whether either works at all:

    endgame's `home_score` and `away_score` are cumulative *after* the play.

So a touchdown play carries the touchdown in its own score columns. Training
on that teaches the model that a team about to score is already ahead, and
the fitted probabilities look superb right up until they're used on a game
still being played -- where the score of a play that hasn't happened isn't
available. `iter_states` therefore takes the score from the *previous* play
(0-0 before the first), and the situation columns from the play itself, which
are already pre-snap.
"""

from logging import getLogger
from typing import Iterator, NamedTuple

from .game import GamePlays
from .plays import Play

logger = getLogger(__name__)

# Football's regulation clock: four periods of fifteen minutes. Both leagues
# this reads play it; a period beyond the fourth is overtime, whose length is
# a league rule (ten minutes in the NFL, untimed possessions in college), so
# `seconds_remaining` stops at zero there rather than pretending to know.
PERIOD_SECONDS = 15 * 60
REGULATION_PERIODS = 4
REGULATION_SECONDS = PERIOD_SECONDS * REGULATION_PERIODS

# Halves, which matter to expected points rather than to win probability: the
# clock running out on a half is an absorbing "nobody scored" that the score
# itself doesn't record, so `lucky_ones.points` needs to know where the
# boundary is and how far away it is.
PERIODS_PER_HALF = 2
HALF_SECONDS = PERIOD_SECONDS * PERIODS_PER_HALF


def half_of(period: int) -> int:
    """
    Which half `period` belongs to: 1 for Q1-Q2, 2 for Q3-Q4.

    Every overtime period is its own half, numbered on from there, because
    that is what the boundary is *for* -- the next scoring drive doesn't
    carry across a kickoff that resets the situation, and college overtime
    resets it every possession pair. Nothing here plays a period long enough
    for the numbering past 2 to mean anything else.
    """
    if period > REGULATION_PERIODS:
        return period - REGULATION_PERIODS + PERIODS_PER_HALF
    return 1 if period <= PERIODS_PER_HALF else 2


class GameState(NamedTuple):
    """
    One snap, as the model sees it.

    Everything is oriented to the *home* team -- `score_margin` is home minus
    away, and the model's output is the home team's win probability -- except
    `yardline` and `down`/`distance`, which belong to whoever has the ball.
    `offense_is_home` is what ties the two orientations together, and is why
    the field-position features can't be read without it.
    """

    game_id: str
    play_id: str
    play_number: int

    period: int

    clock_seconds: int
    """Left in `period`. What a chart labels a point with ("Q3 7:22")."""

    seconds_remaining: int
    """Left in regulation, across all four periods. 0 in overtime."""

    is_overtime: bool

    home_score: int
    """*Before* this play -- see the module docstring."""

    away_score: int

    offense_is_home: bool
    down: int
    distance: int
    yardline: int
    """1 is the offense's own 1-yard line, 99 is the defense's."""

    @property
    def half(self) -> int:
        """
        1 in the first half, 2 in the second, and one per overtime period
        after that. See `half_of`.
        """
        return half_of(self.period)

    @property
    def half_seconds_remaining(self) -> int:
        """
        Seconds left in *this half*, which is the clock expected points runs
        on.

        Win probability asks how much game is left to change the result, so it
        reads `seconds_remaining`. Expected points asks how much time this
        drive has to turn field position into points, and that clock stops at
        the half: a first down at midfield with eight seconds left in the
        second quarter is worth almost nothing, and `seconds_remaining` would
        call it a normal midfield first down with a quarter and a half to go.

        Zero in overtime, for the same reason `seconds_remaining` is -- there
        is no league-agnostic length to count down from.
        """
        if self.is_overtime:
            return 0
        # Odd periods have the rest of their half after them; even ones end it.
        return self.clock_seconds + (
            PERIOD_SECONDS if self.period % PERIODS_PER_HALF == 1 else 0
        )

    @property
    def score_margin(self) -> int:
        """
        Home minus away, before this play. The model's strongest feature, and
        derived rather than stored so it can't disagree with the two scores a
        chart puts in a tooltip.
        """
        return self.home_score - self.away_score


class GameOutcome(NamedTuple):
    """What actually happened, for training and evaluation."""

    game_id: str
    home_score: int
    away_score: int

    @property
    def home_won(self) -> bool:
        return self.home_score > self.away_score

    @property
    def tied(self) -> bool:
        return self.home_score == self.away_score


NOT_A_SNAP: tuple[str, ...] = (
    "timeout",
    "two-minute warning",
    "kickoff",
    "coin toss",
)
"""
Play types that aren't a scrimmage play, matched as substrings of a lowered
`play_type`.

Every one of these arrives in the source data looking like a snap -- a down,
a distance, a yardline, a clock and a possession team -- so the missing-column
test below can't see them, and for a long time this module let them through.
What they actually are:

- **Timeouts**, including the TV timeout ESPN writes as `Official Timeout`.
  They carry the situation of the snap around them, or a placeholder: a
  first and ten at the 35 is the commonest single row in the NFL's 2023 feed.
- **The two-minute warning**, which is the same thing under its own name.
- **Kickoffs**, which mostly have no down and are dropped by that test -- but
  in NCAAFB usually do have one, so `Kickoff` and `Kickoff Return (Offense)`
  together were 5.5% of that league's states.
- **The coin toss**, and (through the `end ` prefix below) `End Period`,
  `End of Game`, `End of Regulation`.

Together they were 7.1% of NFL states and 9.5% of NCAAFB ones, and they are
not harmless padding. They break both models in different ways. Expected
points is the worse of the two: a timeout on the kickoff after a score is
priced as first and ten at your own 35 with the possession stale, so it is
labelled with a next score that belongs to the other team. Measured, that put
the fit 1.6 points wrong on the whole 33-to-37 slice of the field. For EPA it
is simpler and just as wrong -- a timeout is not a play, and it has no
business in a per-play denominator.

A null `play_type` is *kept*. Absence of evidence isn't evidence of a
timeout, and a real snap whose type didn't parse should be modelled rather
than dropped.
"""


def is_scrimmage_play(play_type: str | None) -> bool:
    """
    Whether `play_type` names something run from scrimmage.

    True for a null type, for the reason `NOT_A_SNAP` gives. Note what is
    deliberately True here: a penalty is a snap (the flag came out during a
    play, and the standard treatment counts it), and so are punts and field
    goal attempts, which are downs and are priced as downs.
    """
    if play_type is None:
        return True
    lowered = play_type.lower()
    if lowered.startswith("end "):
        return False
    return not any(marker in lowered for marker in NOT_A_SNAP)


def iter_states(game: GamePlays) -> Iterator[GameState]:
    """
    One `GameState` per scrimmage play, in order.

    Plays that aren't a snap are skipped rather than filled in, and it takes
    two tests to find them all:

    - **The columns.** Extra points and most kickoffs have no down, END
      QUARTER and END GAME have no possession team, and a play missing its
      clock has no place on the time axis.
    - **The type.** `NOT_A_SNAP` is the rest, and it is the half that isn't
      obvious: a timeout arrives with every column a snap has, so nothing but
      `play_type` says it wasn't one.

    All of them are ordinary rows in the source data, so this is a filter, not
    an error path -- roughly one play in four of a real game.

    Skipping them costs nothing either model wants. A kickoff's win
    probability is the state before it and after the score it follows, both of
    which are kept; a timeout's is the snap on either side of it, twice.
    """
    home_score, away_score = 0, 0
    for play in game.plays:
        state = _to_state(game, play, home_score=home_score, away_score=away_score)
        if state is not None:
            yield state
        # After yielding, so the state above is the one before this play.
        if play.home_score is not None:
            home_score = play.home_score
        if play.away_score is not None:
            away_score = play.away_score


def final_outcome(game: GamePlays) -> GameOutcome | None:
    """
    The final score, read off the last play that carries one.

    The last play of a game is usually END GAME, which has scores but no
    possession -- hence "the last play that carries one" rather than the last
    play. None when no play in the game has a score at all, which is what a
    game whose play-by-play never loaded looks like.
    """
    for play in reversed(game.plays):
        if play.home_score is not None and play.away_score is not None:
            return GameOutcome(
                game_id=game.game_id,
                home_score=play.home_score,
                away_score=play.away_score,
            )
    logger.warning("No play in %s carries a score", game.game_id)
    return None


def _to_state(
    game: GamePlays, play: Play, home_score: int, away_score: int
) -> GameState | None:
    if not is_scrimmage_play(play.play_type):
        return None
    if (
        play.period is None
        or play.clock_seconds is None
        or play.down is None
        or play.distance is None
        or play.yardline is None
        or play.offense_team_id is None
    ):
        return None
    return GameState(
        game_id=game.game_id,
        play_id=play.play_id,
        play_number=play.play_number,
        period=play.period,
        clock_seconds=play.clock_seconds,
        seconds_remaining=_seconds_remaining(play.period, play.clock_seconds),
        is_overtime=play.period > REGULATION_PERIODS,
        home_score=home_score,
        away_score=away_score,
        offense_is_home=play.offense_team_id == game.home_team_id,
        down=play.down,
        distance=play.distance,
        yardline=play.yardline,
    )


def _seconds_remaining(period: int, clock_seconds: int) -> int:
    """
    Seconds left in regulation: this period's clock plus the periods after
    it.

    Clamped at both ends. Zero in overtime, because there is no "remaining"
    to speak of once regulation is over and the model's time features would
    read an eleventh-period clock as a very long game. Zero also for a period
    beyond the fourth arriving with a full clock, which is the same thing.
    """
    if period > REGULATION_PERIODS:
        return 0
    remaining = (REGULATION_PERIODS - period) * PERIOD_SECONDS + clock_seconds
    return max(0, min(REGULATION_SECONDS, remaining))
