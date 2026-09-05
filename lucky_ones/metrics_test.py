import numpy as np
import pytest

from .metrics import (
    brier_score,
    log_loss,
    mean_absolute_error,
    multiclass_log_loss,
)


def test_brier_of_a_perfect_forecast():
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_brier_of_a_coin_flip():
    assert brier_score(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == pytest.approx(
        0.25
    )


def test_brier_rewards_being_right():
    confident = brier_score(np.array([0.9]), np.array([1.0]))
    hedged = brier_score(np.array([0.6]), np.array([1.0]))

    assert confident < hedged


def test_log_loss_punishes_confident_mistakes_harder_than_brier():
    """
    Why both are reported: a 0.99 on a team that lost costs about the same as
    four coin flips under Brier, and vastly more under log loss.
    """
    wrong = np.array([0.99])
    lost = np.array([0.0])

    assert brier_score(wrong, lost) < 1.0
    assert log_loss(wrong, lost) > 4.0


def test_log_loss_of_certainty_is_finite():
    assert np.isfinite(log_loss(np.array([1.0]), np.array([0.0])))


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="line up"):
        brier_score(np.array([0.5]), np.array([1.0, 0.0]))


def test_scoring_nothing_is_refused():
    with pytest.raises(ValueError, match="Nothing to score"):
        brier_score(np.array([]), np.array([]))


def test_multiclass_log_loss_of_a_perfect_forecast():
    certain = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    assert multiclass_log_loss(certain, np.array([0, 2])) == pytest.approx(0.0)


def test_multiclass_log_loss_of_a_uniform_guess():
    """
    `log(k)` is what predicting the class frequencies gets you, so a fit
    scoring worse than this has learned nothing about the situation.
    """
    uniform = np.full((4, 7), 1 / 7)

    assert multiclass_log_loss(uniform, np.zeros(4, dtype=int)) == pytest.approx(
        np.log(7)
    )


def test_multiclass_log_loss_of_a_class_the_fit_never_saw_is_finite():
    """
    A holdout class with a column of zeros is a genuine miss, and one that
    should be charged for rather than crash the scoring.
    """
    value = multiclass_log_loss(np.array([[1.0, 0.0]]), np.array([1]))

    assert np.isfinite(value) and value > 10.0


def test_multiclass_log_loss_refuses_an_index_off_the_end():
    with pytest.raises(ValueError, match="within"):
        multiclass_log_loss(np.array([[0.5, 0.5]]), np.array([2]))


def test_mean_absolute_error_is_in_the_units_it_was_given():
    predicted = np.array([2.0, 0.0, -3.0])
    actual = np.array([7.0, 0.0, -3.0])

    assert mean_absolute_error(predicted, actual) == pytest.approx(5 / 3)
