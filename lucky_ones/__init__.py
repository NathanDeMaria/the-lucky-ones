"""
In-game win probability for football, from play-by-play.

The whole top-level API is `MODELS` and what it takes:

    from lucky_ones import MODELS, group_by_game

    (game,) = group_by_game(plays)
    points = MODELS.NCAAFB.curve(game)          # the graph
    control = MODELS.NCAAFB.game_control(game)  # the number under it

`game_control` says who was ahead of the game. Two neighbours say what the
bounces had to do with it, and -- given the name of this package -- are
arguably the point of it:

    earned = MODELS.NCAAFB.luck_adjusted_game_control(game)
    breaks = MODELS.NCAAFB.lucky_wp(game)

The first is the same share of the game with the fumbles and the tipped balls
split evenly rather than credited to whoever they fell to. The second is the
win probability those bounces handed each team, totalled -- the size of the
breaks rather than the game without them. See `lucky_ones.luck`.

A fourth number answers a different question again -- not who was ahead, and
not what the bounces were worth, but who actually played better:

    epa = MODELS.NCAAFB.epa_per_play(game)

Expected points added per snap, bounded so one enormous play can't be the
whole number and weighted by how much the game was still in doubt when it
happened. It reads on the scoreboard's scale rather than on a share of
anything, and it is the one here that doesn't care who won. See
`lucky_ones.epa`.

`MODELS.NFL` and `MODELS.NCAAFB` are fitted releases that ship inside the
wheel, so scoring a game needs no bucket, no credentials and no fitting
stack -- pin this package by rev and you have pinned the model with it. Each
carries how it scored and what it was fit on (`.metrics`, `.trained_on`), and
the fit itself is on `.model`. See `lucky_ones.bundled`.

Six names, because that is what scoring a game needs: the models, the one
function that turns plays into games, and the four types on the boundary --
the ones a consumer has to be able to write down the type of. Everything else
-- the protocols, the free functions the methods above wrap, the feature
lists, the fitting stack -- is a module import away, and the table below says
which module.

The pipeline, in the order the modules run:

    plays     what a play is, and where plays come from -- protocols only
    arrow     the adapter from endgame's stored parquet to those protocols
    game      plays grouped into games, each with its home side worked out
    state     one snap as the model sees it, pre-snap score and all
    features  a state as a row of numbers, for either model
    training  games as labelled matrices, split by game rather than by row
    model     `WinProbabilityModel`, and a logistic baseline
    points    `ExpectedPointsModel`: what a snap is worth on the scoreboard
    metrics   scoring the result
    curve     a game's win probability over time, and game control
    luck      the coin flips: the curve without them, and what they were worth
    epa       expected points added per play, bounded and weighted
    release   the artifacts a fit is stored as
    bundled   the fits that ship with the package, and `MODELS`

Installing it comes in three sizes, because those modules do three jobs. A
bare `pip install lucky-ones` is the inference half -- numpy and pydantic --
which is everything the five names above reach. Beyond it are two extras, and
they are separate because the jobs are: `lucky-ones[data]` adds pyarrow, for
reading endgame's stored plays, and `lucky-ones[fit]` adds the solver, for
producing a fit. A service that scores games it didn't fit takes the first and
not the second; this repo takes both. Neither half is exported here, and
importing it is what asks for its extra:

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
from .epa import EpaPerPlay
from .game import GamePlays, group_by_game

__all__ = [
    "MODELS",
    "CurvePoint",
    "EpaPerPlay",
    "GameControl",
    "GamePlays",
    "group_by_game",
]
