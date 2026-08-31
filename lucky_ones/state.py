"""
The situation a win probability is a function of, and the result it's
predicting.

A `GameState` is everything the model conditions on at one snap, on one
scale, with no Nones left in it. Getting from a `Play` to one is mostly
bookkeeping, except for a detail that decides whether the model works at all:

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


def iter_states(game: GamePlays) -> Iterator[GameState]:
    """
    One `GameState` per scrimmage play, in order.

    Plays that aren't a snap are skipped rather than filled in: kickoffs and
    extra points have no down, END QUARTER and END GAME have no possession
    team, and a play missing its clock has no place on the time axis. All of
    them are ordinary rows in the source data, so this is a filter, not an
    error path -- roughly one play in six of a real game.

    Skipping them costs nothing the model wants. A kickoff's win probability
    is the state before it and after the score it follows, both of which are
    kept.
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
