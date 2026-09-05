"""
The release artifacts: the contract between this package and its consumers.

A release is a self-contained snapshot of one fitted model -- its
coefficients, what it was trained on, and how it scored on a holdout. There
are two of them, `WinProbabilityRelease` and `ExpectedPointsRelease`, because
there are two fits and they answer different questions; they share `TrainedOn`
and differ in their metrics, because a probability and a number of points
aren't scored the same way.
A consumer reads the JSON and never fits anything, which is what lets a web
service draw a win probability curve without scikit-learn, without a bucket,
and without this package's training stack.

Same arrangement as cassandra's `ModelRelease`, deliberately: invisible-string
already reads one of those, and a second artifact that works a different way
is a second thing to learn. This module depends on nothing heavier than
pydantic and `lucky_ones.model` -- neither of which imports scikit-learn at
module scope -- which is the property that lets a consumer install lucky-ones
without the `fit` group and still read a release.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .features import EP_FEATURE_NAMES, FEATURE_NAMES
from .model import LogisticWinProbability
from .points import MultinomialExpectedPoints, ScoreKind


class TrainedOn(BaseModel):
    """
    What the fit saw.

    Seasons and weeks rather than a game count alone, because the first
    question about a win probability curve that looks wrong is always "what
    was this trained on" -- and the second is whether the game being drawn
    was in it.
    """

    league: str
    seasons: list[int] = Field(default_factory=list)
    weeks: list[int] = Field(default_factory=list)
    n_games: int
    n_snaps: int


class Metrics(BaseModel):
    """
    How the fit scored on games it wasn't fit on.

    Both are error measures, so lower is better -- a consumer picking between
    two releases wants a min, not a max. `n_games` and `n_snaps` are here
    because a Brier score off three games is not a Brier score.

    Held-out by *game*, never by snap: every snap of a game carries that
    game's result, so a row-wise split would score the model on games it had
    already been told the answer to. See `lucky_ones.training.split_games`.
    """

    brier_score: float
    log_loss: float
    n_games: int
    n_snaps: int


class WinProbabilityRelease(BaseModel):
    """One fitted model at a point in time."""

    # `model_...` is pydantic's protected namespace, and `model` is the
    # natural name for the field holding the fit.
    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal[1] = 1
    run_id: str
    league: str
    model_kind: Literal["logistic"] = "logistic"
    feature_names: list[str] = Field(default_factory=lambda: list(FEATURE_NAMES))
    coefficients: list[float]
    intercept: float
    trained_on: TrainedOn
    metrics: Metrics
    created_at: datetime
    created_by: str

    def to_model(self) -> LogisticWinProbability:
        """
        The fit, as something you can call `predict` on.

        Raises `ValueError` if the release's features aren't the ones this
        build produces -- the expected outcome for a release written by a
        newer lucky-ones, and much better than silently reinterpreting
        positional coefficients against a different feature list.
        """
        return LogisticWinProbability.from_dict(
            {
                "feature_names": self.feature_names,
                "coefficients": self.coefficients,
                "intercept": self.intercept,
            }
        )

    @classmethod
    def from_model(
        cls,
        model: LogisticWinProbability,
        *,
        run_id: str,
        league: str,
        trained_on: TrainedOn,
        metrics: Metrics,
        created_at: datetime,
        created_by: str,
    ) -> "WinProbabilityRelease":
        return cls(
            run_id=run_id,
            league=league,
            feature_names=list(model.feature_names),
            coefficients=model.coefficients.tolist(),
            intercept=model.intercept,
            trained_on=trained_on,
            metrics=metrics,
            created_at=created_at,
            created_by=created_by,
        )


class ExpectedPointsMetrics(BaseModel):
    """
    How an expected points fit scored on snaps it wasn't fit on.

    Two numbers, and they say different things. `log_loss` is over the seven
    classes and is the one to compare two fits on -- `log(7)` = 1.95 is what
    guessing the base rates gets, so a fit near it has learned nothing.
    `mean_absolute_error` is over the expected value against the points that
    actually came next, in points, and is the one to quote when somebody asks
    what an expected points number is worth: it is large by construction --
    the next score is 7 or 0 or -3 and the fit says 2.1 -- and its size is the
    reason EPA is a per-play average rather than a per-play claim.

    Held out by *game*, like the win probability metrics and for a sharper
    reason: every snap of a drive usually shares one label, so a row-wise
    split scores the fit on drives it was told the answer to.
    """

    log_loss: float
    mean_absolute_error: float
    n_games: int
    n_snaps: int


class ExpectedPointsRelease(BaseModel):
    """
    One fitted expected points model at a point in time.

    Shipped alongside the win probability release for the same league, as a
    separate file: they are fit separately, they move separately, and a
    retrain of one shouldn't rewrite the other's diff. `lucky_ones.bundled`
    is what puts the pair back together, and `lucky_ones.epa` is the one
    caller that needs both.
    """

    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal[1] = 1
    run_id: str
    league: str
    model_kind: Literal["multinomial"] = "multinomial"
    feature_names: list[str] = Field(default_factory=lambda: list(EP_FEATURE_NAMES))
    kinds: list[ScoreKind]
    """
    The classes the coefficients are rows of, in order.

    Stored rather than assumed to be all seven: a fit off a short season may
    never have seen a defensive safety, and which rows are missing is not
    something to infer from a row count.
    """

    coefficients: list[list[float]]
    """One row per kind, one column per feature."""

    intercepts: list[float]
    trained_on: TrainedOn
    metrics: ExpectedPointsMetrics
    created_at: datetime
    created_by: str

    def to_model(self) -> MultinomialExpectedPoints:
        """
        The fit, as something you can call `predict` on.

        Raises `ValueError` if the release's features aren't the ones this
        build produces -- see `WinProbabilityRelease.to_model`.
        """
        return MultinomialExpectedPoints.from_dict(
            {
                "feature_names": self.feature_names,
                "kinds": [kind.value for kind in self.kinds],
                "coefficients": self.coefficients,
                "intercepts": self.intercepts,
            }
        )

    @classmethod
    def from_model(
        cls,
        model: MultinomialExpectedPoints,
        *,
        run_id: str,
        league: str,
        trained_on: TrainedOn,
        metrics: ExpectedPointsMetrics,
        created_at: datetime,
        created_by: str,
    ) -> "ExpectedPointsRelease":
        return cls(
            run_id=run_id,
            league=league,
            feature_names=list(model.feature_names),
            kinds=list(model.kinds),
            coefficients=model.coefficients.tolist(),
            intercepts=model.intercepts.tolist(),
            trained_on=trained_on,
            metrics=metrics,
            created_at=created_at,
            created_by=created_by,
        )


def release_json_schema() -> dict[str, Any]:
    """The JSON Schema for a release, for consumers checking their fixtures."""
    return WinProbabilityRelease.model_json_schema()


def expected_points_release_json_schema() -> dict[str, Any]:
    """The same, for the expected points artifact."""
    return ExpectedPointsRelease.model_json_schema()
