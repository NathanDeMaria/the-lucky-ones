"""
The training script, end to end.

Runs the real entry points against a synthetic tree, so the path a person
takes -- `make train` and `make train-ep`, then `make curve` and `make epa` --
is exercised by CI rather than first discovered on a laptop with credentials.
"""

import json

import numpy as np
import pytest

from lucky_ones.epa import DEFAULT_CLIP
from lucky_ones.features import EP_FEATURE_NAMES
from lucky_ones.luck import DEFENDED_MARKERS
from lucky_ones.points import ScoreKind
from lucky_ones.release import ExpectedPointsRelease, WinProbabilityRelease
from synthetic import write_tree
from train import (
    DEFENDED_FAMILIES,
    _parse_seasons,
    _parse_weeks,
    bounds,
    curve,
    epa,
    rates,
    train,
    train_ep,
)

# The first game `write_tree` writes for 2025, and the team at home in it.
# Both are deterministic -- that is what a fixture is for -- and both carry
# the season, since a tree with two of them in it must not collide.
FIXTURE_GAME = "g2025100"
FIXTURE_HOME = "t1"


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

    curve(
        FIXTURE_GAME, league="nfl", season=2025, week=1, model=str(out), root=str(tree)
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["game_id"] == FIXTURE_GAME
    assert payload["home_team_id"] == FIXTURE_HOME
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
        "luck_adjusted_win_probability",
    }
    assert 0.0 < first["home_win_probability"] < 1.0
    control = payload["game_control"]
    assert control["home"] + control["away"] == pytest.approx(1.0)
    assert control["seconds"] == 3600


def test_curve_prints_the_luck_adjusted_control_too(tmp_path, tree, capsys):
    """
    The pair a reader wants: what happened, and what happened on purpose. The
    synthetic tree carries fumble text (see `synthetic.FUMBLE_CHANCE`) so this
    exercises the adjustment rather than the no-luck shortcut through it.
    """
    out = tmp_path / "nfl.json"
    train(league="nfl", seasons="2025", root=str(tree), out=str(out))

    curve(
        FIXTURE_GAME, league="nfl", season=2025, week=1, model=str(out), root=str(tree)
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["lucky_plays"], "the fixture should contain some fumbles"
    assert {lucky["kind"] for lucky in payload["lucky_plays"]} <= {
        "fumble_lost",
        "fumble_kept",
        "pass_defended_interception",
        "pass_defended_incomplete",
    }
    earned = payload["luck_adjusted_game_control"]
    assert earned["home"] + earned["away"] == pytest.approx(1.0)
    assert earned["seconds"] == payload["game_control"]["seconds"]
    # Bounces went someone's way, so the two numbers are not the same one
    assert earned["home"] != pytest.approx(payload["game_control"]["home"])


def test_curve_prints_what_the_bounces_were_worth(tmp_path, tree, capsys):
    """
    The other luck number, and the per-play arithmetic under it: each lucky
    play carries both branches and the share of the gap the bounce decided.
    """
    out = tmp_path / "nfl.json"
    train(league="nfl", seasons="2025", root=str(tree), out=str(out))

    curve(
        FIXTURE_GAME, league="nfl", season=2025, week=1, model=str(out), root=str(tree)
    )

    payload = json.loads(capsys.readouterr().out)
    breaks = payload["lucky_wp"]
    assert breaks["home"] >= 0.0 and breaks["away"] >= 0.0
    assert breaks["net"] == pytest.approx(breaks["home"] - breaks["away"])
    # Not a share of the game -- a total of win probability. See `lucky_ones.luck`.
    assert breaks["home"] + breaks["away"] != pytest.approx(1.0)

    (lucky, *_) = payload["lucky_plays"]
    assert lucky["expected"] == pytest.approx(
        lucky["retained"] * lucky["realized"]
        + (1 - lucky["retained"]) * lucky["counterfactual"]
    )
    assert lucky["home_delta"] == pytest.approx(lucky["realized"] - lucky["expected"])
    assert sum(
        abs(play["home_delta"]) for play in payload["lucky_plays"]
    ) == pytest.approx(breaks["home"] + breaks["away"])


def test_rates_measures_the_fumble_coin(tree, capsys):
    """
    The measurement behind `DEFAULT_RETAINED[FUMBLE_LOST]`, run end to end.

    It counts through `find_lucky_plays`, so what comes out is the rate over
    the population the model actually adjusts rather than a second opinion
    about which plays those are.
    """
    rates(league="nfl", seasons="2025", root=str(tree))

    payload = json.loads(capsys.readouterr().out)
    assert payload["retained_in_use"]["fumble_lost"] == 0.5
    fumbles = payload["fumbles"]
    assert fumbles["classified"] > 0, "the fixture should contain some fumbles"
    # The pair, which is the constraint on any value these could be given.
    assert fumbles["lost"] + fumbles["kept"] == pytest.approx(1.0, abs=1e-3)
    assert "2025" in fumbles["by_season"]


def test_rates_reports_what_the_feed_wrote_down_alongside_what_it_means(tree, capsys):
    """
    `coverage` describes the feed -- how much of the denominator a season's
    text records at all -- and runs over every game, including the ones the
    share excludes. It is what says whether a season is legible; on real
    NCAAFB it moves by a factor of four while the football underneath doesn't.
    """
    rates(league="nfl", seasons="2025", root=str(tree))

    defended = json.loads(capsys.readouterr().out)["defended_passes"]
    assert defended["attempts"] > 0
    assert defended["defended"] > 0, "the fixture should defend some passes"
    assert defended["coverage"] == pytest.approx(
        defended["defended"] / defended["attempts"], abs=1e-4
    )
    assert defended["gated_coverage"] == pytest.approx(
        defended["gated_defended"] / defended["gated_attempts"], abs=1e-4
    )


def test_rates_splits_the_denominator_by_the_phrase_that_caught_it(tree, capsys):
    """
    `by_family` answers "did the annotation stop, or is it worded differently
    now?" -- the question the share's instability always raises. It omits the
    families that caught nothing, which on real data is nearly all of them.
    """
    rates(league="nfl", seasons="2025", root=str(tree))

    defended = json.loads(capsys.readouterr().out)["defended_passes"]
    assert defended["by_family"] == {"broken_up": defended["defended"]}
    assert set(defended["by_family"]) <= {*DEFENDED_FAMILIES, "defensed_syntax"}


def test_the_feed_profiler_covers_what_the_classifier_reads():
    """
    `rates` splits the denominator by family for diagnostics while the total
    comes from `is_defended_pass`. If a marker were added to the classifier
    and not to the families, the split would stop summing to the total and
    quietly under-report whichever family it belonged to.
    """
    covered = {marker for markers in DEFENDED_FAMILIES.values() for marker in markers}

    assert set(DEFENDED_MARKERS) <= covered


def test_rates_measures_the_share_only_where_both_sides_are_recorded(tree, capsys):
    """
    The correctness of the share, and the thing that is easy to get wrong: a
    game whose feed records none of its breakups still records all of its
    interceptions, so counting it puts a numerator into the ratio with nothing
    underneath it. On real NCAAFB that is four games in five in 2023, and it
    is the difference between reading 0.53 and reading 0.19.
    """
    rates(league="nfl", seasons="2025", root=str(tree))

    defended = json.loads(capsys.readouterr().out)["defended_passes"]
    assert 0 < defended["games_recording"] <= defended["games"]
    # The share is the gated counts and nothing else.
    assert defended["interception_share"] == pytest.approx(
        defended["gated_interceptions"]
        / (defended["gated_interceptions"] + defended["gated_defended"]),
        abs=1e-4,
    )
    # The feed-wide counts are still reported, and are the larger ones.
    assert defended["interceptions"] >= defended["gated_interceptions"]
    assert defended["defended"] >= defended["gated_defended"]


def test_the_denominator_is_every_contested_pass(tree, capsys):
    """
    The pass coin is the fumble coin in a different hat: the denominator is
    every pass a defender got to, and which side came up is whether it was
    caught. So the share and `DEFAULT_RETAINED` have to mean the same thing --
    `interception_share` is what `PASS_DEFENDED_INTERCEPTION` is set from, and
    its complement is the other kind.
    """
    rates(league="nfl", seasons="2025", root=str(tree))

    payload = json.loads(capsys.readouterr().out)
    defended = payload["defended_passes"]
    contested = defended["gated_interceptions"] + defended["gated_defended"]

    assert contested == defended["gated_interceptions"] + defended["gated_defended"]
    assert defended["interception_share"] == pytest.approx(
        defended["gated_interceptions"] / contested, abs=1e-4
    )
    retained = payload["retained_in_use"]
    assert retained["pass_defended_interception"] + retained[
        "pass_defended_incomplete"
    ] == pytest.approx(1.0)
    assert retained["fumble_lost"] + retained["fumble_kept"] == pytest.approx(1.0)


def test_train_ep_writes_a_release_that_loads(tmp_path, tree):
    out = tmp_path / "nfl-ep.json"

    train_ep(league="nfl", seasons="2025", root=str(tree), out=str(out))

    release = ExpectedPointsRelease.model_validate_json(out.read_text())
    assert release.league == "nfl"
    assert release.trained_on.n_snaps > release.trained_on.n_games
    # `log(k)` is what predicting the base rates gets you, so a fit above it
    # has learned nothing about the situation.
    assert 0.0 < release.metrics.log_loss < np.log(len(release.kinds))
    assert release.metrics.mean_absolute_error > 0.0

    model = release.to_model()
    assert model.coefficients.shape == (len(release.kinds), len(EP_FEATURE_NAMES))
    assert model.predict([]).shape == (0,)


def test_train_ep_ships_the_classes_it_actually_saw(tmp_path, tree):
    """
    The fixture never scores a safety, so its fit has five classes and not
    seven -- which is exactly why the release stores them rather than
    assuming the full list.
    """
    out = tmp_path / "nfl-ep.json"

    train_ep(league="nfl", seasons="2025", root=str(tree), out=str(out))

    release = ExpectedPointsRelease.model_validate_json(out.read_text())
    assert set(release.kinds) < set(ScoreKind)
    assert len(release.coefficients) == len(release.kinds)


def test_epa_prints_a_number_per_offense_and_the_plays_behind_it(
    tmp_path, tree, capsys
):
    win = tmp_path / "nfl.json"
    points = tmp_path / "nfl-ep.json"
    train(league="nfl", seasons="2025", root=str(tree), out=str(win))
    train_ep(league="nfl", seasons="2025", root=str(tree), out=str(points))

    epa(
        FIXTURE_GAME,
        league="nfl",
        season=2025,
        week=1,
        model=str(win),
        ep_model=str(points),
        root=str(tree),
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["game_id"] == FIXTURE_GAME
    result = payload["epa_per_play"]
    assert result["home_plays"] + result["away_plays"] == len(payload["plays"])
    assert result["net"] == pytest.approx(result["home"] - result["away"])
    # The effective sample is what the average actually covers, and it can
    # only be smaller than the play count.
    assert 0 < result["home_weight"] <= result["home_plays"]

    first = payload["plays"][0]
    assert set(first) == {
        "play_id",
        "play_number",
        "offense_is_home",
        "expected_points",
        "epa",
        "bounded",
        "win_probability",
        "weight",
    }
    assert abs(first["bounded"]) <= payload["clip"]


def test_epa_turns_both_knobs_off_at_the_call_site(tmp_path, tree, capsys):
    """
    `--clip inf --weight_power 0` is the plain unweighted mean of raw EPA,
    which is the number to compare against when you want to see what the two
    adjustments are doing.
    """
    win = tmp_path / "nfl.json"
    points = tmp_path / "nfl-ep.json"
    train(league="nfl", seasons="2025", root=str(tree), out=str(win))
    train_ep(league="nfl", seasons="2025", root=str(tree), out=str(points))

    epa(
        FIXTURE_GAME,
        league="nfl",
        season=2025,
        week=1,
        model=str(win),
        ep_model=str(points),
        root=str(tree),
        clip=float("inf"),
        weight_power=0.0,
    )

    payload = json.loads(capsys.readouterr().out)
    home = [play for play in payload["plays"] if play["offense_is_home"]]
    assert all(play["weight"] == 1.0 for play in payload["plays"])
    assert payload["epa_per_play"]["home"] == pytest.approx(
        sum(play["epa"] for play in home) / len(home)
    )


def test_bounds_measures_the_tail_behind_the_clip(tmp_path, tree, capsys):
    """
    What `rates` is to `DEFAULT_RETAINED`, this is to `DEFAULT_CLIP` -- and it
    runs through `play_epa`, so the distribution it reports is the one the
    metric averages rather than a second opinion about it.
    """
    win = tmp_path / "nfl.json"
    points = tmp_path / "nfl-ep.json"
    train(league="nfl", seasons="2025", root=str(tree), out=str(win))
    train_ep(league="nfl", seasons="2025", root=str(tree), out=str(points))

    bounds(
        league="nfl",
        seasons="2025",
        root=str(tree),
        model=str(win),
        ep_model=str(points),
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["clip_in_use"] == DEFAULT_CLIP
    season = payload["by_season"]["2025"]
    assert season["plays"] > 0
    quantiles = season["quantiles"]
    assert quantiles["p1"] < quantiles["p50"] < quantiles["p99"]
    # |EPA| is what a symmetric bound is a quantile of, and it only rises.
    absolute = season["abs_quantiles"]
    assert absolute["p90"] <= absolute["p99"] <= absolute["p99.9"]
    assert 0.0 <= season["beyond_clip"] <= 1.0
    assert 0.0 < season["effective_play_share"] <= 1.0
    assert season["game_shift"]["team_games"] > 0
    assert season["game_shift"]["median_abs"] >= 0.0
