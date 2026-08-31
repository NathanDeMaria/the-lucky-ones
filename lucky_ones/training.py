"""
Turning games into rows a model can be fit on.

Small, but it's where two decisions live that would otherwise get made
differently in every script: what to do with a game that has no result, and
what to do with a tie.
"""

import random
from logging import getLogger
from typing import Iterable, NamedTuple

from .game import GamePlays
from .state import GameState, final_outcome, iter_states

logger = getLogger(__name__)


class TrainingSet(NamedTuple):
    """
    States and their labels, one label per state.

    Flat rather than nested by game: every state of a game shares that game's
    label, and fanning it out here means the fitting code never has to know
    that rows are correlated within a game. Note that they *are* -- 150 snaps
    of one blowout are not 150 independent observations -- so a standard
    error off this matrix is optimistic, and a train/test split has to be by
    game rather than by row. `split_games` is what to use for the latter.
    """

    states: list[GameState]
    home_won: list[bool]

    @property
    def rows(self) -> int:
        """
        How many snaps are in here.

        Named rather than `__len__`, because this is a NamedTuple and
        `len(training)` is already 2 -- the number of fields. Shadowing that
        would be a lie to anything that unpacks it.
        """
        return len(self.states)


def build_training_set(games: Iterable[GamePlays]) -> TrainingSet:
    """
    Every scrimmage snap of every finished game, labelled with whether the
    home team went on to win.

    Two kinds of game are dropped:

    - one with no score on any play, which is a game whose play-by-play never
      loaded rather than a 0-0 result;
    - a tie, which has no binary label. Real but rare (a couple of NFL games
      a season, none in college), and inventing a 0.5 for them would mean
      writing a loss function around a handful of rows.

    A game still in progress is *not* filtered out here, because the plays
    don't say: the last play of a stored week is the last one ESPN had, not
    necessarily the last one played. Feed this finished games -- endgame's
    `Game.status` is what knows.
    """
    states: list[GameState] = []
    home_won: list[bool] = []
    for game in games:
        outcome = final_outcome(game)
        if outcome is None:
            continue
        if outcome.tied:
            logger.info("Skipping %s: it ended tied", game.game_id)
            continue
        game_states = list(iter_states(game))
        states.extend(game_states)
        home_won.extend([outcome.home_won] * len(game_states))
    return TrainingSet(states=states, home_won=home_won)


def split_games(
    games: Iterable[GamePlays], holdout_fraction: float = 0.2, seed: int = 0
) -> tuple[list[GamePlays], list[GamePlays]]:
    """
    Split by game, never by play.

    Splitting rows would put snaps from the same game on both sides, and
    every one of them carries the same label -- so the holdout would be
    scoring the model on games it had already been told the answer to, and
    every metric off it would be too good.

    Deterministic given `seed`: a holdout that moves between runs makes two
    fits incomparable.
    """
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction has to be in [0, 1): {holdout_fraction}")
    # Sorted first so the shuffle is a function of the seed and the set of
    # games, not of whatever order they arrived in.
    ordered = sorted(games, key=lambda game: game.game_id)
    random.Random(seed).shuffle(ordered)
    cut = int(len(ordered) * (1.0 - holdout_fraction))
    return ordered[:cut], ordered[cut:]
