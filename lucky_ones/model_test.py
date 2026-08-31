import json

import numpy as np
import pytest

from .conftest import make_state as _state
from .features import FEATURE_NAMES
from .model import LogisticWinProbability, WinProbabilityModel, predict_one


def _model(**overrides) -> LogisticWinProbability:
    """
    A hand-made fit: everything zero except a positive weight on the score
    margin, so its predictions are readable without fitting anything.
    """
    coefficients = np.zeros(len(FEATURE_NAMES))
    coefficients[FEATURE_NAMES.index("score_margin")] = 0.1
    return LogisticWinProbability(coefficients=coefficients, intercept=0.0, **overrides)


def _accepts_model(model: WinProbabilityModel) -> WinProbabilityModel:
    """Static conformance check -- see the same helper in arrow_test."""
    return model


def test_the_baseline_is_a_win_probability_model():
    _accepts_model(_model())


def test_predictions_are_probabilities_that_follow_the_lead():
    model = _model()

    behind, level, ahead = model.predict(
        [_state(score_margin=-14), _state(), _state(score_margin=14)]
    )

    assert 0.0 < behind < level < ahead < 1.0
    assert level == pytest.approx(0.5)


def test_predict_one():
    assert predict_one(_model(), _state()) == pytest.approx(0.5)


def test_an_enormous_lead_does_not_overflow():
    """
    A three-score lead in the last seconds is a genuinely huge logit, not a
    bug -- `1 / (1 + exp(-x))` would warn and hand back garbage.
    """
    model = _model()

    with np.errstate(over="raise"):
        probability = predict_one(model, _state(score_margin=1000))

    assert probability == pytest.approx(1.0)


def test_the_wrong_number_of_coefficients_is_refused():
    with pytest.raises(ValueError, match="coefficients"):
        LogisticWinProbability(coefficients=np.zeros(2), intercept=0.0)


def test_a_fit_round_trips_through_a_file(tmp_path):
    model = _model()
    path = tmp_path / "fit.json"

    model.save(path)
    loaded = LogisticWinProbability.load(path)

    assert loaded.intercept == model.intercept
    assert np.allclose(loaded.coefficients, model.coefficients)
    assert loaded.feature_names == FEATURE_NAMES


def test_a_fit_from_different_features_is_refused():
    """
    Coefficients are positional, so a fit made before a feature was added
    would still load and still predict -- wrongly, and silently.
    """
    saved = _model().to_dict()
    saved["feature_names"] = ["score_margin"]

    with pytest.raises(ValueError, match="different features"):
        LogisticWinProbability.from_dict(saved)


def test_the_saved_file_is_readable_json(tmp_path):
    path = tmp_path / "fit.json"

    _model().save(path)

    saved = json.loads(path.read_text())
    assert set(saved) == {"feature_names", "coefficients", "intercept"}


def test_fitting_recovers_the_sign_of_the_lead():
    """
    An end-to-end fit on synthetic games where the leader always wins. Not a
    quality bar -- just that the plumbing between features, labels and
    sklearn is the right way round.
    """
    states, labels = [], []
    for margin in range(-21, 22, 3):
        for seconds in (3000, 1500, 60):
            states.append(_state(score_margin=margin, seconds_remaining=seconds))
            labels.append(margin > 0)

    model = LogisticWinProbability.fit(states, labels)

    assert predict_one(model, _state(score_margin=14, seconds_remaining=60)) > 0.5
    assert predict_one(model, _state(score_margin=-14, seconds_remaining=60)) < 0.5


def test_fitting_a_set_with_one_outcome_is_refused():
    states = [_state(score_margin=margin) for margin in (-7, 0, 7)]

    with pytest.raises(ValueError, match="nothing to fit"):
        LogisticWinProbability.fit(states, [True, True, True])


def test_fitting_mismatched_lengths_is_refused():
    with pytest.raises(ValueError, match="line up"):
        LogisticWinProbability.fit([_state()], [True, False])
