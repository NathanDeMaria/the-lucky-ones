from datetime import datetime, timezone

import numpy as np
import pytest

from .conftest import make_state
from .features import FEATURE_NAMES
from .model import LogisticWinProbability
from .release import (
    Metrics,
    TrainedOn,
    WinProbabilityRelease,
    release_json_schema,
)


def _release(**overrides) -> WinProbabilityRelease:
    model = LogisticWinProbability(
        coefficients=np.arange(len(FEATURE_NAMES), dtype=float) / 10,
        intercept=0.25,
    )
    return WinProbabilityRelease.from_model(
        model,
        run_id="20260831-000000",
        league="nfl",
        trained_on=TrainedOn(
            league="nfl", seasons=[2024, 2025], weeks=[1, 2], n_games=400, n_snaps=60000
        ),
        metrics=Metrics(brier_score=0.12, log_loss=0.38, n_games=100, n_snaps=15000),
        created_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        created_by="tests",
        **overrides,
    )


def test_a_release_round_trips_through_json():
    original = _release()

    restored = WinProbabilityRelease.model_validate_json(original.model_dump_json())

    assert restored == original


def test_the_release_rehydrates_a_model_that_predicts():
    release = _release()

    model = release.to_model()

    assert np.allclose(model.coefficients, release.coefficients)
    assert model.intercept == release.intercept
    probability = model.predict([make_state(score_margin=10)])[0]
    assert 0.0 < probability < 1.0


def test_a_release_from_different_features_is_refused():
    """
    Coefficients are positional. A release written before a feature was added
    would otherwise rehydrate into a model that predicts confidently and
    wrongly.
    """
    release = _release()
    stale = release.model_copy(update={"feature_names": ["score_margin"]})

    with pytest.raises(ValueError, match="different features"):
        stale.to_model()


def test_the_schema_is_versioned():
    """
    A consumer pins this package by rev and reads the JSON. `schema_version`
    is how it tells a format change from a retrain.
    """
    assert _release().schema_version == 1
    assert "schema_version" in release_json_schema()["properties"]


def test_the_json_has_no_surprises_for_a_consumer():
    payload = _release().model_dump(mode="json")

    assert set(payload) == {
        "schema_version",
        "run_id",
        "league",
        "model_kind",
        "feature_names",
        "coefficients",
        "intercept",
        "trained_on",
        "metrics",
        "created_at",
        "created_by",
    }
    assert payload["metrics"]["brier_score"] == 0.12
