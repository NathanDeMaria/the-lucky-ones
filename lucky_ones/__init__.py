"""
In-game win probability for football, from play-by-play.

The whole top-level API is `MODELS` and what it takes:

    from lucky_ones import MODELS, group_by_game

    (game,) = group_by_game(plays)
    points = MODELS.NCAAFB.curve(game)          # the graph
    control = MODELS.NCAAFB.game_control(game)  # the number under it

`MODELS.NFL` and `MODELS.NCAAFB` are fitted releases that ship inside the
wheel, so scoring a game needs no bucket, no credentials and no fitting
stack -- pin this package by rev and you have pinned the model with it. Each
carries how it scored and what it was fit on (`.metrics`, `.trained_on`), and
the fit itself is on `.model`. See `lucky_ones.bundled`.

Five names, because a consumer wanting a win probability curve needs exactly
five: the models, the one function that turns plays into games, and the three
types on the boundary. Everything else -- the protocols, the free functions
the methods above wrap, the feature list, the fitting stack -- is a module
import away, and the table below says which module.

The pipeline, in the order the modules run:

    plays     what a play is, and where plays come from -- protocols only
    arrow     the adapter from endgame's stored parquet to those protocols
    game      plays grouped into games, each with its home side worked out
    state     one snap as the model sees it, pre-snap score and all
    features  a state as a row of numbers
    training  games as a labelled matrix, split by game rather than by row
    model     `WinProbabilityModel`, and a logistic baseline
    metrics   scoring the result
    curve     a game's win probability over time, and game control
    release   the artifact a fit is stored as
    bundled   the fits that ship with the package, and `MODELS`

Installing it comes in two sizes, because those modules do two jobs. A bare
`pip install lucky-ones` is the inference half -- numpy and pydantic -- which
is everything the five names above reach. Producing a fit, or reading
endgame's stored parquet, needs `lucky-ones[train]`: pyarrow, the solver, and
the store client, about 250MB that a consumer drawing a graph has no use for.
That half is not exported here, and importing it is what asks for the extra:

    from endgame_aws.pbp_parquet import get_processed_plays_store

    from lucky_ones import group_by_game
    from lucky_ones.arrow import StorePlaySource
    from lucky_ones.model import LogisticWinProbability
    from lucky_ones.training import build_training_set

    source = StorePlaySource(get_processed_plays_store())
    plays = await source.load_weeks("nfl", 2025, range(1, 19))
    training = build_training_set(group_by_game(plays))
    model = LogisticWinProbability.fit(training.states, training.home_won)

which is `train.py`'s job, not a consumer's. The split is enforced by the
import graph: nothing reachable from this module imports pyarrow or
scikit-learn, and `init_test` re-imports the package with pyarrow blocked to
keep that true.
"""

from .bundled import MODELS
from .curve import CurvePoint, GameControl
from .game import GamePlays, group_by_game

__all__ = [
    "MODELS",
    "CurvePoint",
    "GameControl",
    "GamePlays",
    "group_by_game",
]
