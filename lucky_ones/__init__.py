"""
In-game win probability for football, from play-by-play.

The pipeline, in the order the modules run:

    plays     what a play is, and where plays come from -- protocols only
    arrow     the adapter from endgame's stored parquet to those protocols
    game      plays grouped into games, each with its home side worked out
    state     one snap as the model sees it, pre-snap score and all
    features  a state as a row of numbers
    training  games as a labelled matrix, split by game rather than by row
    model     `WinProbabilityModel`, and a logistic baseline
    metrics   scoring the result

Fitting, end to end:

    from endgame_aws.pbp_parquet import get_processed_plays_store
    from lucky_ones import (
        LogisticWinProbability, StorePlaySource, build_training_set,
        group_by_game,
    )

    async with get_processed_plays_store() as store:
        plays = await StorePlaySource(store).load_weeks("nfl", 2025, range(1, 19))
    training = build_training_set(group_by_game(plays))
    model = LogisticWinProbability.fit(training.states, training.home_won)
    model.save("nfl.json")

and scoring a game needs neither the store nor scikit-learn -- a saved fit,
some plays, and `model.predict(list(iter_states(game)))`.
"""

from .arrow import (
    ArrowPlay,
    ProcessedPlaysStoreLike,
    StorePlaySource,
    TablePlaySource,
    table_to_plays,
)
from .features import FEATURE_NAMES, feature_matrix, to_features
from .game import GamePlays, group_by_game, infer_home_team_id
from .metrics import brier_score, log_loss
from .model import LogisticWinProbability, WinProbabilityModel, predict_one
from .plays import Play, PlaySource
from .state import GameOutcome, GameState, final_outcome, iter_states
from .training import TrainingSet, build_training_set, split_games

__all__ = [
    "FEATURE_NAMES",
    "ArrowPlay",
    "GameOutcome",
    "GamePlays",
    "GameState",
    "LogisticWinProbability",
    "Play",
    "PlaySource",
    "ProcessedPlaysStoreLike",
    "StorePlaySource",
    "TablePlaySource",
    "TrainingSet",
    "WinProbabilityModel",
    "brier_score",
    "build_training_set",
    "feature_matrix",
    "final_outcome",
    "group_by_game",
    "infer_home_team_id",
    "iter_states",
    "log_loss",
    "predict_one",
    "split_games",
    "table_to_plays",
    "to_features",
]
