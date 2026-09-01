"""
The fits that ship in the wheel, and the `MODELS` that reaches them.

Two jobs. The first is that `MODELS.NCAAFB.curve(game)` works, which is the
whole point of bundling. The second is the one that catches a real break: a
shipped fit is coefficients against a positional feature list, so editing
`FEATURE_NAMES` without retraining silently reinterprets every number in
these files. `LogisticWinProbability.from_dict` refuses that, and the tests
below make it refuse it here rather than in a consumer.
"""

import pytest

from .arrow import table_to_plays
from .bundled import MODELS, RELEASE_DIR, BundledModel
from .conftest import make_play, make_state, make_table
from .curve import win_probability_curve
from .features import FEATURE_NAMES
from .game import GamePlays
from .state import iter_states


def _game(rows) -> GamePlays:
    return GamePlays(
        game_id="g1",
        league="nfl",
        season=2025,
        week=1,
        home_team_id="home",
        away_team_id="away",
        plays=table_to_plays(make_table(rows)),
    )


def _a_close_game() -> GamePlays:
    """Four snaps of a one-score game, enough to have a curve and a control."""
    return _game(
        [
            make_play(
                play_id=f"p{number}",
                play_number=number,
                period=period,
                clock_seconds=clock,
                home_score=home,
                away_score=away,
            )
            for number, (period, clock, home, away) in enumerate(
                [(1, 900, 0, 0), (2, 400, 7, 3), (3, 300, 7, 10), (4, 120, 14, 10)],
                start=1,
            )
        ]
    )


@pytest.mark.parametrize("bundled", list(MODELS), ids=lambda model: model.league)
def test_every_shipped_fit_matches_the_current_features(bundled: BundledModel) -> None:
    """
    The regression guard.

    A fit whose `feature_names` aren't `FEATURE_NAMES` can't be scored -- the
    coefficients mean something else -- and `to_model` raises rather than
    quietly mismatching them. Failing here means the features moved and the
    releases need `make train` re-run before they ship.
    """
    assert tuple(bundled.release.feature_names) == FEATURE_NAMES
    assert len(bundled.model.coefficients) == len(FEATURE_NAMES)


@pytest.mark.parametrize("bundled", list(MODELS), ids=lambda model: model.league)
def test_every_shipped_fit_predicts_probabilities(bundled: BundledModel) -> None:
    states = [
        make_state(score_margin=0, seconds_remaining=3600),
        make_state(score_margin=21, seconds_remaining=60),
        make_state(score_margin=-21, seconds_remaining=60),
    ]

    probabilities = bundled.predict(states)

    assert len(probabilities) == len(states)
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
    # A three-score lead with a minute left beats a three-score deficit. Not a
    # test of the fit's quality -- a test that the coefficients are the right
    # way round, which is what a corrupted or stale file gets wrong.
    assert probabilities[1] > probabilities[0] > probabilities[2]


@pytest.mark.parametrize("bundled", list(MODELS), ids=lambda model: model.league)
def test_every_shipped_fit_reports_a_holdout_score(bundled: BundledModel) -> None:
    """
    A release with no metrics is one written by hand or by an older schema.

    Brier is bounded above by 0.25 for a model that always says 0.5, so a
    shipped fit above that is worse than a coin flip and shouldn't ship.
    """
    assert 0.0 < bundled.metrics.brier_score < 0.25
    assert bundled.metrics.n_games > 0
    assert bundled.trained_on.seasons
    assert bundled.run_id


def test_the_declared_models_are_the_files_that_ship() -> None:
    """
    A fit in the directory that no attribute reaches would be dead weight in
    the wheel; an attribute with no file is an AttributeError at first use.
    """
    on_disk = {path.stem for path in RELEASE_DIR.glob("*.json")}

    assert {model.league for model in MODELS} == on_disk


def test_the_curve_comes_out_of_a_bundled_model() -> None:
    game = _a_close_game()

    points = MODELS.NFL.curve(game)
    control = MODELS.NFL.game_control(game)

    assert len(points) == len(game.plays)
    assert control is not None
    assert control.home + control.away == pytest.approx(1.0)


def test_a_bundled_model_is_a_win_probability_model() -> None:
    """
    It satisfies the protocol, so it goes anywhere a fit goes -- which is what
    lets a caller keep using the free functions.
    """
    game = _a_close_game()

    assert win_probability_curve(MODELS.NFL, game) == MODELS.NFL.curve(game)


def test_curve_from_states_skips_walking_the_game_twice() -> None:
    game = _a_close_game()
    points = MODELS.NFL.curve(game)

    assert MODELS.NFL.curve_from_states(list(iter_states(game))) == points


def test_lookup_by_league_name() -> None:
    assert MODELS["nfl"] is MODELS.NFL
    # Case-insensitive, because the league in a request path or a filename is
    # as likely to arrive shouting as not.
    assert MODELS["NCAAFB"] is MODELS.NCAAFB


def test_an_unknown_league_says_what_there_is() -> None:
    with pytest.raises(KeyError, match="ncaafb"):
        MODELS["cfl"]


def test_a_release_is_read_once() -> None:
    """Cached, so a service scoring a week doesn't re-parse the JSON per game."""
    assert MODELS.NFL.release is MODELS.NFL.release
    assert MODELS.NFL.model is MODELS.NFL.model


def test_a_missing_fit_says_how_to_make_one() -> None:
    with pytest.raises(FileNotFoundError, match="make train"):
        BundledModel("xfl").release


def test_repr_does_not_read_the_release() -> None:
    """
    A repr that touches the disk makes a debugger crawl and can raise while
    formatting a traceback. Checked on a league with no file, which would
    raise if the repr loaded anything.
    """
    assert repr(BundledModel("xfl")) == "BundledModel('xfl')"
