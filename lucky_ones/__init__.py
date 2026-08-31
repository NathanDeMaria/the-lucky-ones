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
    curve     a game's win probability over time, and game control
    release   the artifact a consumer reads

Fitting, end to end:

    from endgame_aws.pbp_parquet import get_processed_plays_store
    from lucky_ones import (
        LogisticWinProbability, StorePlaySource, build_training_set,
        group_by_game,
    )

    source = StorePlaySource(get_processed_plays_store())
    plays = await source.load_weeks("nfl", 2025, range(1, 19))
    training = build_training_set(group_by_game(plays))
    model = LogisticWinProbability.fit(training.states, training.home_won)
    model.save("nfl.json")

and scoring a game needs neither the store nor scikit-learn -- a saved
release, some plays, and

    release = WinProbabilityRelease.model_validate_json(Path("nfl.json").read_text())
    points = win_probability_curve(release.to_model(), game)
    control = game_control(points)

which is the pair a chart draws and the number underneath it.
"""

from .arrow import (
    ArrowPlay,
    DatasetPlaySource,
    ProcessedPlaysStoreLike,
    StorePlaySource,
    TablePlaySource,
    table_to_plays,
)
from .curve import (
    CurvePoint,
    GameControl,
    curve_from_states,
    game_control,
    win_probability_curve,
)
from .features import FEATURE_NAMES, feature_matrix, to_features
from .game import GamePlays, group_by_game, infer_home_team_id
from .metrics import brier_score, log_loss
from .model import LogisticWinProbability, WinProbabilityModel, predict_one
from .plays import Play, PlaySource
from .release import WinProbabilityRelease, release_json_schema
from .state import GameOutcome, GameState, final_outcome, iter_states
from .training import TrainingSet, build_training_set, split_games

__all__ = [
    "ArrowPlay",
    "CurvePoint",
    "DatasetPlaySource",
    "FEATURE_NAMES",
    "GameControl",
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
    "WinProbabilityRelease",
    "brier_score",
    "build_training_set",
    "curve_from_states",
    "feature_matrix",
    "final_outcome",
    "game_control",
    "group_by_game",
    "infer_home_team_id",
    "iter_states",
    "log_loss",
    "predict_one",
    "release_json_schema",
    "split_games",
    "table_to_plays",
    "to_features",
    "win_probability_curve",
]
