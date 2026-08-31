"""
Scoring a set of win probabilities against what happened.

Both metrics here are proper scoring rules, which is the only kind worth
using on a probability: they're minimized by saying what you actually
believe, so a model can't improve its score by shading its predictions
towards the middle.
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
