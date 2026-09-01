"""
The model itself: what one is, and the logistic baseline.

`WinProbabilityModel` is a Protocol for the same reason `PlaySource` is --
everything that consumes a win probability (a report, a backtest, a live
score) should be written against the shape, so that replacing the baseline
below with something better is a new class and no other edits.

`LogisticWinProbability` is that baseline. Logistic regression on the eight
features in `lucky_ones.features`, which is not a state of the art win
probability model and isn't trying to be: it's the model whose coefficients
you can read, whose fit takes a second, and which anything more elaborate has
to beat before it's worth the complexity.
"""

import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .features import FEATURE_NAMES, feature_matrix
from .state import GameState


class WinProbabilityModel(Protocol):
    """
    Anything that can put a number on the home team's chances.

    Vector in, vector out. A per-state signature would be friendlier to read
    and much slower to use -- scoring a season is millions of snaps, and the
    baseline's `predict` is one matrix multiply for all of them.
    """

    def predict(self, states: Sequence[GameState]) -> np.ndarray:
        """
        P(home team wins), one per state, each in [0, 1].

        Same length and order as `states`, so a caller can zip the two back
        together.
        """


def predict_one(model: WinProbabilityModel, state: GameState) -> float:
    """`model.predict` for a single state, for the call sites where the
    vectorized signature is just noise."""
    return float(model.predict([state])[0])


class LogisticWinProbability:
    """
    Logistic regression over `lucky_ones.features.FEATURE_NAMES`.

    Holds nothing but coefficients, an intercept, and the feature names they
    were fit against -- so an instance is a few hundred bytes, `save` writes
    a readable JSON file, and scoring needs numpy and nothing else. That
    split is deliberate: `fit` imports scikit-learn inside the method, so a
    process that only ever loads a saved fit never pays for the import and
    doesn't need the package installed at all (see the `fit` dependency group
    in pyproject.toml).
    """

    def __init__(
        self,
        coefficients: np.ndarray,
        intercept: float,
        feature_names: Sequence[str] = FEATURE_NAMES,
    ) -> None:
        coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
        if len(coefficients) != len(feature_names):
            raise ValueError(
                f"Got {len(coefficients)} coefficients for "
                f"{len(feature_names)} features"
            )
        self.coefficients = coefficients
        self.intercept = float(intercept)
        self.feature_names = tuple(feature_names)

    @classmethod
    def fit(
        cls,
        states: Sequence[GameState],
        home_won: Sequence[bool],
        *,
        regularization: float = 1.0,
    ) -> "LogisticWinProbability":
        """
        Fit on states already labelled with their game's result -- what
        `lucky_ones.training.build_training_set` produces.

        `regularization` is sklearn's `C`, so larger means *less*
        regularization. The default is sklearn's own; with eight features and
        a season of snaps it barely bites, and it's here so a fit on one
        week's games doesn't have to be.

        Raises `ImportError` with something useful to read if scikit-learn
        isn't installed, which is the expected state of a serving
        environment.
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as error:  # pragma: no cover - depends on the install
            raise ImportError(
                "Fitting needs scikit-learn, which isn't an install dependency "
                "of lucky-ones -- scoring a game never calls a solver. Install "
                "the fit extra: `pip install 'lucky-ones[fit]'`, or `uv sync` "
                "in this repo. That is the one that adds the solver and "
                "nothing else -- fitting doesn't need Arrow."
            ) from error

        if len(states) != len(home_won):
            raise ValueError(
                f"Got {len(states)} states and {len(home_won)} labels; they have "
                "to line up row for row"
            )
        labels = np.asarray(home_won, dtype=int)
        if len(np.unique(labels)) < 2:
            raise ValueError(
                "Every game in the training set ended the same way; there's "
                "nothing to fit"
            )
        model = LogisticRegression(C=regularization, max_iter=1000)
        model.fit(feature_matrix(states), labels)
        return cls(
            coefficients=model.coef_[0],
            intercept=float(model.intercept_[0]),
        )

    def predict(self, states: Sequence[GameState]) -> np.ndarray:
        features = feature_matrix(states)
        if features.shape[1] != len(self.coefficients):
            raise ValueError(
                f"This fit has {len(self.coefficients)} coefficients but the "
                f"current features produce {features.shape[1]} columns"
            )
        return _sigmoid(features @ self.coefficients + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, saved: Mapping[str, Any]) -> "LogisticWinProbability":
        """
        Rehydrate a saved fit, refusing one whose features aren't the ones
        this code builds.

        The check is the point of storing the names at all. Coefficients are
        positional, so adding a feature to `FEATURE_NAMES` silently
        reinterprets every number in an older file -- the fit would still
        load, still predict, and be wrong in a way nothing surfaces.
        """
        names = tuple(saved["feature_names"])
        if names != FEATURE_NAMES:
            raise ValueError(
                "This fit was made against different features than the current "
                f"code builds.\n  saved:   {names}\n  current: {FEATURE_NAMES}"
            )
        return cls(
            coefficients=np.asarray(saved["coefficients"], dtype=float),
            intercept=float(saved["intercept"]),
            feature_names=names,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticWinProbability":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """
    Overflow-free logistic function.

    `1 / (1 + exp(-x))` warns and returns garbage for large negative x, which
    the margin-over-root-time feature produces routinely: a three-score lead
    with ten seconds left is a genuinely enormous logit, not a bug.
    """
    positive = logits >= 0
    exponentiated = np.exp(-np.abs(logits))
    return np.where(
        positive, 1.0 / (1.0 + exponentiated), exponentiated / (1.0 + exponentiated)
    )
