"""
Scoring a set of win probabilities against what happened.

The first three are proper scoring rules, which is the only kind worth using
on a probability: they're minimized by saying what you actually believe, so a
model can't improve its score by shading its predictions towards the middle.
`brier_score` and `log_loss` score a win probability against a game that was
won or lost; `multiclass_log_loss` scores an expected points fit against
which of the seven things scored next.

`mean_absolute_error` is the odd one out and is here for expected points
alone: it isn't a scoring rule at all, it's the size of the miss in the units
the number is quoted in. A log loss says a fit is better than another one; a
mean absolute error of 3.6 says what "expected points" is worth as an
estimate of the next score, which is the thing a reader of an EPA wants to
know.
"""

import numpy as np


def brier_score(predicted: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Mean squared error against a 0/1 outcome. Lower is better; 0.25 is what
    predicting 0.5 for everything gets you.

    The headline number for a win probability model, because it's on the
    scale of the thing being predicted -- unlike log loss, which is only
    comparable to another log loss.
    """
    predicted, outcomes = _aligned(predicted, outcomes)
    return float(np.mean((predicted - outcomes) ** 2))


def log_loss(
    predicted: np.ndarray, outcomes: np.ndarray, epsilon: float = 1e-12
) -> float:
    """
    Mean negative log likelihood. Lower is better.

    Worth reporting next to the Brier score because it punishes confident
    mistakes far harder, and a win probability model's failures are exactly
    that: 0.99 on a team that lost. `epsilon` clips away the infinity a
    prediction of exactly 0 or 1 would otherwise produce.
    """
    predicted, outcomes = _aligned(predicted, outcomes)
    clipped = np.clip(predicted, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped))
    )


def _aligned(
    predicted: np.ndarray, outcomes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if predicted.shape != outcomes.shape:
        raise ValueError(
            f"Predictions and outcomes don't line up: {predicted.shape} vs "
            f"{outcomes.shape}"
        )
    if predicted.size == 0:
        raise ValueError("Nothing to score")
    return predicted, outcomes


def multiclass_log_loss(
    probabilities: np.ndarray, actual: np.ndarray, epsilon: float = 1e-12
) -> float:
    """
    Mean negative log likelihood over `(n, k)` class probabilities and the
    `n` column indices of what actually happened. Lower is better.

    The expected points counterpart of `log_loss` above. `log(k)` is what
    predicting the class frequencies gets you, so an eight-way fit that
    scores worse than ~1.95 has learned nothing about the situation.

    Rows are not renormalized. A fit whose rows don't sum to 1 is broken in a
    way this should report rather than repair.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    actual = np.asarray(actual, dtype=int).reshape(-1)
    if probabilities.ndim != 2 or len(probabilities) != len(actual):
        raise ValueError(
            f"Predictions and outcomes don't line up: {probabilities.shape} vs "
            f"{actual.shape}"
        )
    if probabilities.size == 0:
        raise ValueError("Nothing to score")
    if actual.size and (actual.min() < 0 or actual.max() >= probabilities.shape[1]):
        raise ValueError(
            f"Class indices have to be within the {probabilities.shape[1]} "
            "columns predicted"
        )
    chosen = probabilities[np.arange(len(actual)), actual]
    return float(-np.mean(np.log(np.clip(chosen, epsilon, 1.0))))


def mean_absolute_error(predicted: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Mean `|predicted - actual|`, in whatever units they came in.

    Absolute rather than squared on purpose. A snap's next score is one of
    seven discrete values and the fit's job is to average them, so most of
    the squared error is the irreducible spread of the thing being averaged
    -- squaring it just reports the tail again. The absolute version is
    readable as "how far off is this, typically".
    """
    predicted, outcomes = _aligned(predicted, outcomes)
    return float(np.mean(np.abs(predicted - outcomes)))
