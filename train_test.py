"""
The training script, end to end.

Runs the real `train` and `curve` entry points against a synthetic tree, so
the path a person takes -- `make train`, then `make curve` -- is exercised by
CI rather than first discovered on a laptop with credentials.
"""

import json

import pytest

from lucky_ones import WinProbabilityRelease
from synthetic import write_tree
from train import _parse_seasons, _parse_weeks, curve, train


@pytest.fixture(name="tree")
def tree_fixture(tmp_path):
    return write_tree(tmp_path / "plays", weeks=4, games_per_week=8)


def test_seasons_parse_the_way_the_flag_reads():
    assert _parse_seasons("2025") == [2025]
    # Inclusive at both ends: "2022-2025" is four seasons
    assert _parse_seasons("2022-2025") == [2022, 2023, 2024, 2025]
    assert _parse_seasons("2022,2024") == [2022, 2024]
    assert _parse_seasons(2025) == [2025]
    assert _parse_weeks(None)[:3] == [0, 1, 2]


def test_train_writes_a_release_that_loads(tmp_path, tree):
    out = tmp_path / "nfl.json"

    train(league="nfl", seasons="2025", root=str(tree), out=str(out))

    release = WinProbabilityRelease.model_validate_json(out.read_text())
    assert release.league == "nfl"
    assert release.trained_on.n_games > 0
    assert release.trained_on.n_snaps > release.trained_on.n_games
    assert release.metrics.n_games > 0
    # A model that has learned anything at all beats predicting 0.5 for
    # everything, which scores 0.25.
    assert 0.0 < release.metrics.brier_score < 0.25
    assert release.to_model().predict([]).shape == (0,)


def test_train_on_a_season_nobody_has_synced_says_so(tmp_path, tree):
    with pytest.raises(ValueError, match="No nfl games"):
        train(
            league="nfl",
            seasons="1999",
            root=str(tree),
            out=str(tmp_path / "nope.json"),
        )


def test_curve_prints_a_series_and_its_control(tmp_path, tree, capsys):
    out = tmp_path / "nfl.json"
    train(league="nfl", seasons="2025", root=str(tree), out=str(out))

    curve("g100", league="nfl", season=2025, week=1, model=str(out), root=str(tree))

    payload = json.loads(capsys.readouterr().out)
    assert payload["game_id"] == "g100"
    assert payload["home_team_id"] == "home100"
    assert len(payload["points"]) > 100
    first = payload["points"][0]
    assert set(first) == {
        "play_id",
        "play_number",
        "period",
        "clock_seconds",
        "seconds_remaining",
        "home_score",
        "away_score",
        "home_win_probability",
    }
    assert 0.0 < first["home_win_probability"] < 1.0
    control = payload["game_control"]
    assert control["home"] + control["away"] == pytest.approx(1.0)
    assert control["seconds"] == 3600
