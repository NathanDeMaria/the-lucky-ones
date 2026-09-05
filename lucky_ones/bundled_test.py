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
from .bundled import EP_SUFFIX, MODELS, RELEASE_DIR, BundledModel
from .conftest import make_play, make_state, make_table
from .curve import win_probability_curve
from .epa import DEFAULT_CLIP
from .features import EP_FEATURE_NAMES, FEATURE_NAMES
from .game import GamePlays
from .luck import LuckKind
from .points import ScoreKind
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

    Two files per league, and the check is that the directory holds exactly
    those: a win probability fit under the league's name and an expected
    points fit under the same name plus `EP_SUFFIX`.
    """
    on_disk = {path.stem for path in RELEASE_DIR.glob("*.json")}
    leagues = {model.league for model in MODELS}

    assert on_disk == leagues | {league + EP_SUFFIX for league in leagues}


def test_the_curve_comes_out_of_a_bundled_model() -> None:
    game = _a_close_game()

    points = MODELS.NFL.curve(game)
    control = MODELS.NFL.game_control(game)

    assert len(points) == len(game.plays)
    assert control is not None
    assert control.home + control.away == pytest.approx(1.0)


def _a_game_with_a_bounce() -> GamePlays:
    """
    Four consecutive snaps of a scoreless first quarter, the home team putting
    the ball on the ground on the second one.

    Consecutive and scoreless on purpose: the realized branch is read off the
    next snap and the counterfactual off this one, so a fixture whose score
    moves between the two would be measuring the points rather than the
    bounce. Here the only thing that differs between the branches is who has
    the ball, which is the thing being priced.
    """
    return _game(
        [
            make_play(
                play_id=f"p{number}",
                play_number=number,
                clock_seconds=clock,
                yardline=yardline,
                offense_team_id=offense,
                defense_team_id="away" if offense == "home" else "home",
                text=text,
                is_turnover=text is not None,
            )
            for number, (clock, offense, yardline, text) in enumerate(
                [
                    (900, "home", 25, None),
                    (870, "home", 30, "Barkley rush. FUMBLE, recovered by DAL."),
                    (840, "away", 70, None),
                    (810, "away", 72, None),
                ],
                start=1,
            )
        ]
    )


def test_the_luck_numbers_come_out_of_a_bundled_model() -> None:
    """
    The three calls a consumer makes for the pair of luck numbers, against a
    real shipped fit rather than a stand-in -- `luck_test` covers the
    arithmetic, this covers the wiring reaching actual coefficients.
    """
    game = _a_game_with_a_bounce()

    adjusted = MODELS.NFL.luck_adjusted_curve(game)
    earned = MODELS.NFL.luck_adjusted_game_control(game)
    breaks = MODELS.NFL.lucky_wp(game)

    assert [lucky.kind for lucky in adjusted.lucky_plays] == [LuckKind.FUMBLE_LOST]
    assert adjusted.realized == MODELS.NFL.curve(game)
    assert adjusted.points != adjusted.realized
    assert earned is not None
    assert earned.home + earned.away == pytest.approx(1.0)
    # Dallas came up with it, so the break is on their side of the ledger --
    # and it is a total of win probability, not a share of the game.
    assert breaks.home == 0.0
    assert 0.0 < breaks.away < 1.0
    assert breaks.net == pytest.approx(-breaks.away)
    assert [swing.play_id for swing in breaks.swings] == [
        lucky.play_id for lucky in adjusted.lucky_plays
    ]


def test_the_luck_numbers_take_their_weights_at_the_call_site() -> None:
    """`retained=1.0` is the identity, through the bundled wrapper too."""
    game = _a_game_with_a_bounce()
    everything = {kind: 1.0 for kind in LuckKind}

    # `approx` rather than `==`: the identity is exact in the arithmetic, and
    # the adjusted curve still makes a round trip through log-odds to get
    # there, which costs a couple of bits on the way back.
    assert [
        point.home_win_probability
        for point in MODELS.NFL.luck_adjusted_curve(game, retained=everything).points
    ] == pytest.approx([point.home_win_probability for point in MODELS.NFL.curve(game)])
    assert MODELS.NFL.lucky_wp(game, retained=everything).away == pytest.approx(0.0)


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
    with pytest.raises(FileNotFoundError, match="make train "):
        BundledModel("xfl").release


def test_a_missing_expected_points_fit_names_its_own_target() -> None:
    """
    The two fits are made by two commands, so the message has to name the
    right one -- a league can perfectly well have a curve and no EPA.
    """
    with pytest.raises(FileNotFoundError, match="make train-ep"):
        BundledModel("xfl").expected_points_release


def test_repr_does_not_read_the_release() -> None:
    """
    A repr that touches the disk makes a debugger crawl and can raise while
    formatting a traceback. Checked on a league with no file, which would
    raise if the repr loaded anything.
    """
    assert repr(BundledModel("xfl")) == "BundledModel('xfl')"


@pytest.mark.parametrize("bundled", list(MODELS), ids=lambda model: model.league)
def test_every_shipped_ep_fit_matches_the_current_features(
    bundled: BundledModel,
) -> None:
    """
    The same regression guard as the win probability one, and it bites the
    same way: coefficients are positional, so a feature added to
    `EP_FEATURE_NAMES` without a retrain reinterprets every number in the
    file.
    """
    release = bundled.expected_points_release
    assert tuple(release.feature_names) == EP_FEATURE_NAMES
    assert bundled.expected_points.coefficients.shape == (
        len(release.kinds),
        len(EP_FEATURE_NAMES),
    )


@pytest.mark.parametrize("bundled", list(MODELS), ids=lambda model: model.league)
def test_every_shipped_ep_fit_prices_the_field(bundled: BundledModel) -> None:
    """
    Not a test of the fit's quality -- a test that the coefficients are the
    right way round, which is what a corrupted or stale file gets wrong. The
    numbers themselves are football everybody already knows: first and ten on
    your own 1 is worth less than nothing, because the next points in the half
    are more often the other team's, and first and goal is worth nearly a
    touchdown.
    """
    own_one, midfield, goal_line = bundled.expected_points.predict(
        [
            make_state(down=1, distance=10, yardline=1),
            make_state(down=1, distance=10, yardline=50),
            make_state(down=1, distance=2, yardline=98),
        ]
    )

    assert own_one < 0.0 < midfield < goal_line
    assert 1.5 < midfield < 3.5
    assert 4.5 < goal_line < 7.0


@pytest.mark.parametrize("bundled", list(MODELS), ids=lambda model: model.league)
def test_every_shipped_ep_fit_saw_every_outcome(bundled: BundledModel) -> None:
    """
    A shipped fit is off whole seasons, so all seven happened -- a file
    missing one is a fit off far less data than it claims.
    """
    assert set(bundled.expected_points.kinds) == set(ScoreKind)


@pytest.mark.parametrize("bundled", list(MODELS), ids=lambda model: model.league)
def test_every_shipped_ep_fit_reports_a_holdout_score(bundled: BundledModel) -> None:
    """
    `log(7)` = 1.95 is what predicting the base rates gets you, so a shipped
    fit above it has learned nothing about the situation.
    """
    metrics = bundled.expected_points_metrics
    assert 0.0 < metrics.log_loss < 1.95
    assert metrics.mean_absolute_error > 0.0
    assert metrics.n_games > 0


def _a_game_with_a_drive() -> GamePlays:
    """
    Three snaps of the home team moving the ball and scoring, then two of the
    away team going nowhere -- enough for both offenses to have a number.
    """
    return _game(
        [
            make_play(
                play_id=f"p{number}",
                play_number=number,
                clock_seconds=clock,
                yardline=yardline,
                offense_team_id=offense,
                defense_team_id="away" if offense == "home" else "home",
                home_score=home,
                scoring_play=home > 0 and number == 3,
            )
            for number, (clock, offense, yardline, home) in enumerate(
                [
                    (900, "home", 25, 0),
                    (870, "home", 55, 0),
                    (840, "home", 95, 7),
                    (700, "away", 25, 7),
                    (670, "away", 20, 7),
                ],
                start=1,
            )
        ]
    )


def test_epa_per_play_comes_out_of_a_bundled_model() -> None:
    """
    The call that reads both of a league's fits, against real shipped
    coefficients rather than stand-ins -- `epa_test` covers the arithmetic,
    this covers the wiring reaching two files.
    """
    game = _a_game_with_a_drive()

    result = MODELS.NFL.epa_per_play(game)

    assert (result.home_plays, result.away_plays) == (3, 2)
    assert result.home is not None and result.away is not None
    # The home team drove eighty yards and scored; the away team lost five.
    assert result.home > 0.0 > result.away
    assert result.net == pytest.approx(result.home - result.away)
    assert [play.play_id for play in result.plays] == [
        f"p{number}" for number in range(1, 6)
    ]


def test_epa_per_play_takes_its_knobs_at_the_call_site() -> None:
    """
    `clip=inf, weight_power=0` is the plain unweighted mean of raw EPA --
    the same escape hatch `retained=1.0` is next door, and the same reason
    for it.
    """
    game = _a_game_with_a_drive()

    bounded = MODELS.NFL.epa_per_play(game, clip=0.5)
    plain = MODELS.NFL.epa_per_play(game, clip=float("inf"), weight_power=0.0)

    assert bounded.home is not None and plain.home is not None
    assert bounded.home < plain.home
    assert plain.home == pytest.approx(
        sum(play.epa for play in plain.plays if play.offense_is_home) / 3
    )
    assert all(abs(play.bounded) <= DEFAULT_CLIP for play in plain.plays)


def test_the_two_releases_are_read_once_each() -> None:
    assert MODELS.NFL.expected_points_release is MODELS.NFL.expected_points_release
    assert MODELS.NFL.expected_points is MODELS.NFL.expected_points
