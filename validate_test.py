"""
The validation script, end to end.

Runs the real `validate` entry point against a synthetic tree, so `make
validate` is exercised by CI rather than first discovered the next time a
release changes. It cannot check that the *numbers* are right -- the fixture
isn't football -- so it checks the things that would silently produce a
plausible-looking wrong report: that the fit and test seasons really are
disjoint, that every section is present and internally consistent, and that
the charts are well-formed SVG whose marks land inside the canvas.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import charts
from synthetic import write_tree
from validate import CHARTS, redraw, validate, weighting


@pytest.fixture(name="tree")
def tree_fixture(tmp_path):
    # Two seasons, because the whole design is fitting on one and measuring on
    # the other. Twenty-four teams and six weeks, because the split-half
    # section wants twenty team-seasons before it will report anything and a
    # team wants several games before it can be dealt into two sets.
    root = tmp_path / "plays"
    for season in (2024, 2025):
        write_tree(root, season=season, weeks=6, games_per_week=12)
    return root


@pytest.fixture(name="report")
def report_fixture(tmp_path, tree):
    out = tmp_path / "docs"
    validate(
        league="nfl",
        root=str(tree),
        out=str(out),
        fit_seasons="2024",
        test_seasons="2025",
        min_games=4,
        draws=60,
    )
    return out, json.loads((out / "validation-nfl.json").read_text())


def test_the_fit_and_test_seasons_have_to_be_disjoint(tmp_path, tree):
    """
    The one mistake that would invalidate every number here without changing
    how any of them look.
    """
    with pytest.raises(ValueError, match="overlap"):
        validate(
            league="nfl",
            root=str(tree),
            out=str(tmp_path / "docs"),
            fit_seasons="2024-2025",
            test_seasons="2025",
        )


def test_the_report_says_what_it_measured_and_on_what(report):
    _, payload = report

    assert payload["fit_seasons"] == [2024]
    assert payload["test_seasons"] == [2025]
    assert payload["n_fit_games"] > 0 and payload["n_test_games"] > 0
    assert payload["skill"]["n_snaps"] > payload["n_test_games"]


def test_skill_is_reported_against_both_baselines(report):
    """
    A log loss on its own says nothing, so both baselines are reported and
    the skill is stated relative to the one that matters -- how often each
    outcome happens, ignoring the situation.

    What this checks is the arithmetic, not that the fit wins. It can't check
    that: the fixture's downs and yardlines are drawn at random and have
    nothing to do with its scoring, so there is no situation to learn and a
    fit that beat the base rates here would be reading noise. Whether the
    real fits win is `docs/epa-validation.md`, on real football.
    """
    _, payload = report
    skill = payload["skill"]

    assert 0.0 < skill["log_loss"] < skill["log_loss_uniform"]
    assert skill["log_loss_skill"] == pytest.approx(
        1 - skill["log_loss"] / skill["log_loss_base_rates"], abs=1e-3
    )
    assert skill["mean_absolute_error_skill"] == pytest.approx(
        1 - skill["mean_absolute_error"] / skill["mean_absolute_error_base_rates"],
        abs=1e-3,
    )
    assert skill["bias"] == pytest.approx(
        skill["mean_predicted"] - skill["mean_actual"], abs=1e-3
    )


def test_calibration_bins_are_equal_count_and_ordered(report):
    _, payload = report
    bins = payload["calibration"]["bins"]

    assert len(bins) == 20
    assert max(row["n"] for row in bins) - min(row["n"] for row in bins) <= 1
    # Sorted by prediction, because that's what binning by prediction means.
    assert [row["predicted"] for row in bins] == sorted(
        row["predicted"] for row in bins
    )
    assert payload["calibration"]["worst_gap"] == pytest.approx(
        max(abs(row["predicted"] - row["actual"]) for row in bins), abs=1e-3
    )


def test_the_surface_carries_the_fit_and_the_observed_side(report):
    """
    The observed side is the point of that chart: a curve nobody can check
    against the data is a curve.
    """
    _, payload = report
    surface = payload["surface"]

    for down in ("1", "2", "3", "4"):
        assert len(surface["fitted"][down]) == len(surface["yardlines"])
        assert len(surface["observed"][down]) == len(surface["yardlines"])
    assert any(
        row["actual"] is not None
        for down in surface["observed"].values()
        for row in down
    )


def test_expected_points_stays_inside_what_can_be_scored(report):
    """
    A classifier over what scores next can't leave the range its classes
    cover, however strange the situation -- which is most of why it is a
    classifier and not a regression on points.
    """
    _, payload = report

    assert all(-7.0 < row["expected_points"] < 7.0 for row in payload["landmarks"])
    for down in payload["surface"]["fitted"].values():
        assert all(-7.0 < value < 7.0 for value in down)


def test_the_split_half_deal_reports_what_it_covered(report):
    _, payload = report
    split = payload["split_half"]

    assert split["team_seasons"] >= 20, "the fixture should have enough teams"
    assert split["metrics"], split.get("note")
    for row in split["metrics"].values():
        for target in ("reliability", "all_scoring", "live_scoring"):
            assert row[target]["low"] <= row[target]["r"] <= row[target]["high"]
    assert 0.0 <= split["epa_vs_points"]["reliability"]["p_better"] <= 1.0


def test_every_chart_is_well_formed_svg(report):
    out, payload = report

    for name in CHARTS:
        path = out / f"{name}-nfl.svg"
        assert path.is_file(), f"{name} was not drawn"
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")
        markup = path.read_text()
        # A NaN reaches the file as the literal string and renders as nothing,
        # which is the failure mode a chart hides best.
        assert not re.search(r"\bnan\b|\binf\b|None", markup)


def _number(element: ET.Element, name: str) -> float:
    """One required numeric attribute. `.get` is optional and these aren't."""
    value = element.get(name)
    assert value is not None, f"<{element.tag}> has no {name}"
    return float(value)


def test_chart_marks_land_on_the_canvas(report):
    """
    The bug this catches for real: transposing the x and y limits, which
    produces a valid SVG whose every mark is off the top of the picture.
    """
    out, _ = report

    for name in CHARTS:
        path = out / f"{name}-nfl.svg"
        root = ET.parse(path).getroot()
        width, height = _number(root, "width"), _number(root, "height")
        for element in root.iter():
            tag = element.tag.split("}")[-1]
            if tag == "circle":
                x, y = _number(element, "cx"), _number(element, "cy")
            elif tag == "line":
                x, y = _number(element, "x1"), _number(element, "y1")
            else:
                continue
            assert -2 <= x <= width + 2, f"{path.name}: {tag} at x={x}"
            assert -2 <= y <= height + 2, f"{path.name}: {tag} at y={y}"


def test_redraw_rebuilds_the_charts_without_refitting(tmp_path, report):
    """
    The measurement is minutes and the charts are milliseconds, so adjusting
    a label shouldn't mean refitting two models.
    """
    out, _ = report
    path = out / "calibration-nfl.svg"
    before = path.read_text()
    path.write_text("<svg/>")

    redraw(league="nfl", out=str(out))

    assert path.read_text() == before


def test_redraw_says_what_to_run_when_there_is_nothing_to_draw(tmp_path):
    with pytest.raises(FileNotFoundError, match="make validate"):
        redraw(league="xfl", out=str(tmp_path))


def test_nice_ticks_are_numbers_a_person_would_have_picked():
    assert charts.nice_ticks(0.0, 1.0, 6) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert charts.nice_ticks(-3.0, 7.0, 6) == [-2.0, 0.0, 2.0, 4.0, 6.0]
    # Degenerate ranges are a real case -- a season where nothing varied.
    assert charts.nice_ticks(1.0, 1.0) == [1.0]


def test_the_axes_transform_puts_the_limits_on_the_plot_edges():
    axes = charts.Axes(200, 100, (0.0, 10.0), (-1.0, 1.0))

    assert axes.x(0.0) == pytest.approx(axes.left)
    assert axes.x(10.0) == pytest.approx(axes.right)
    # SVG's y grows downward, so the axis is flipped exactly once.
    assert axes.y(-1.0) == pytest.approx(axes.bottom)
    assert axes.y(1.0) == pytest.approx(axes.top)


def test_a_chart_paints_its_own_background():
    """
    GitHub renders these against whatever theme the reader chose, so a
    transparent ground is dark text on dark for half the audience.
    """
    axes = charts.Axes(200, 100, (0.0, 1.0), (0.0, 1.0))

    markup = axes.render("title")

    assert charts.GROUND in markup
    assert 'width="200"' in markup


def test_chart_text_is_escaped():
    axes = charts.Axes(200, 100, (0.0, 1.0), (0.0, 1.0))

    axes.text(0.5, 0.5, "punts & <sacks>")

    assert "punts &amp; &lt;sacks&gt;" in axes.render()


def test_docs_reference_charts_that_exist():
    """
    A broken image in the report reads as a broken model. Every figure the
    page links has to be a file `make validate` writes.
    """
    docs = Path("docs")
    page = (docs / "epa-validation.md").read_text()

    for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", page):
        assert (docs / target).is_file(), f"{target} is linked but missing"


# --- the weighting trial ------------------------------------------------


@pytest.fixture(name="trial")
def trial_fixture(tmp_path, tree):
    out = tmp_path / "docs"
    weighting(
        league="nfl",
        root=str(tree),
        out=str(out),
        fit_seasons="2024",
        test_seasons="2025",
        min_games=4,
        draws=40,
    )
    return json.loads((out / "weighting-nfl.json").read_text())


def test_the_weighting_trial_reports_every_control(trial):
    """
    Three numbers on the same footing plus the two same-games ones. The
    controls are the whole point: `weighted` against `unadjusted` is the
    comparison that can't be read, and `weighted` against `thinned` is the one
    that can.
    """
    assert set(trial["metrics"]) == {
        "unadjusted",
        "weighted",
        "thinned",
        "live_unadjusted",
        "live_weighted",
        "live_thinned",
    }
    for key in (
        "weighted_vs_thinned",
        "thinned_vs_unadjusted",
        "weighted_vs_unadjusted",
        "live_weighted_vs_live_unadjusted",
        "live_weighted_vs_live_thinned",
    ):
        for target in ("reliability", "all_scoring", "live_scoring"):
            assert 0.0 <= trial[key][target]["p_better"] <= 1.0


def test_the_thinning_matches_the_weighting_effective_sample(trial):
    """
    Kish's `n_eff` is what makes the control a control. It has to be a real
    fraction of the sample -- below 1 because weighting always costs
    precision, and above 0 because a weight of `4p(1-p)` is only zero on a
    game that was never in doubt for a single snap.
    """
    assert 0.0 < trial["effective_sample_share"] < 1.0


def test_the_trial_decomposes_the_raw_penalty(trial):
    """
    The arithmetic the report turns on: whatever the weighting appears to cost
    against the plain number is the sample it spends plus what it buys back by
    choosing snaps. Those two have to add up, or the decomposition is a story
    rather than a measurement.
    """
    for target in ("reliability", "all_scoring", "live_scoring"):
        whole = trial["weighted_vs_unadjusted"][target]["difference"]
        cost = trial["thinned_vs_unadjusted"][target]["difference"]
        bought = trial["weighted_vs_thinned"][target]["difference"]
        assert whole == pytest.approx(cost + bought, abs=1e-3)


def test_the_trial_refuses_overlapping_seasons(tmp_path, tree):
    with pytest.raises(ValueError, match="overlap"):
        weighting(
            league="nfl",
            root=str(tree),
            out=str(tmp_path / "docs"),
            fit_seasons="2024-2025",
            test_seasons="2025",
        )
