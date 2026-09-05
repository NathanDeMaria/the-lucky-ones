import json

import numpy as np
import pytest

from .arrow import table_to_plays
from .conftest import make_play, make_state, make_table
from .features import EP_FEATURE_NAMES
from .game import GamePlays
from .points import (
    SCORE_VALUES,
    ExpectedPointsModel,
    MultinomialExpectedPoints,
    ScoreKind,
    next_scores,
    score_events,
    scoring_plays,
)


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


def _drive(*, offense="home", home=0, away=0, start=1, period=1, clock=900, **shared):
    """
    Snaps in order, each carrying the score *after* it. `shared` goes on every
    row, which is how a case says "and this one scored" once.
    """
    return make_play(
        play_id=f"p{start}",
        play_number=start,
        period=period,
        clock_seconds=clock,
        home_score=home,
        away_score=away,
        offense_team_id=offense,
        defense_team_id="away" if offense == "home" else "home",
        **shared,
    )


def _accepts_model(model: ExpectedPointsModel) -> ExpectedPointsModel:
    """Static conformance check -- see the same helper in model_test."""
    return model


def _model(**overrides) -> MultinomialExpectedPoints:
    """
    A hand-made fit over three kinds, with every coefficient zero except a
    weight on `yards_to_goal`, and intercepts that put the crossover at
    midfield.

    So its predictions are readable without fitting anything: near the
    offense's own goal line it says the other team scores next, at midfield
    it says the two are even, and near the opponent's it says the offense
    scores.
    """
    kinds = (
        ScoreKind.OFFENSE_TOUCHDOWN,
        ScoreKind.NO_SCORE,
        ScoreKind.DEFENSE_TOUCHDOWN,
    )
    coefficients = np.zeros((len(kinds), len(EP_FEATURE_NAMES)))
    field = EP_FEATURE_NAMES.index("yards_to_goal")
    coefficients[0][field] = -4.0
    coefficients[2][field] = +4.0
    return MultinomialExpectedPoints(
        coefficients=coefficients,
        intercepts=np.array([2.0, 0.0, -2.0]),
        kinds=kinds,
        **overrides,
    )


# --- Reading the scores off the columns --------------------------------


def test_a_touchdown_and_its_try_are_one_event():
    """
    The extra point is its own row and its own single point. Counting it
    would put a scoring event in the middle of a kickoff, and price the snap
    before the touchdown at 1.
    """
    game = _game(
        [
            _drive(start=1),
            _drive(start=2, home=6, scoring_play=True),
            _drive(start=3, home=7, down=None, scoring_play=True),
        ]
    )

    (event,) = score_events(list(game.plays))

    assert (event.play_id, event.points, event.home_scored) == ("p2", 7.0, True)


@pytest.mark.parametrize(
    "change,points",
    [(6, 7.0), (7, 7.0), (8, 7.0), (3, 3.0), (2, 2.0)],
    ids=["touchdown", "touchdown-with-the-try", "two-point", "field-goal", "safety"],
)
def test_the_size_of_the_jump_says_what_scored(change, points):
    game = _game([_drive(start=1), _drive(start=2, home=change, scoring_play=True)])

    (event,) = score_events(list(game.plays))

    assert event.points == points


def test_a_score_the_feed_does_not_flag_is_not_a_score():
    """
    The second witness, and the reason for it: before 2014 both leagues'
    score columns jitter, putting points on a play and taking them off a snap
    later. Read alone they invent a touchdown and then an untouchdown.
    """
    game = _game(
        [
            _drive(start=1),
            _drive(start=2, home=7),
            _drive(start=3, home=0),
            _drive(start=4, home=7, scoring_play=True),
        ]
    )

    (event,) = score_events(list(game.plays))

    assert event.play_id == "p4"


def test_a_correction_resets_what_the_next_jump_is_measured_from():
    """
    A score that goes backwards is the feed telling us the total, so the next
    real score has to be a jump from the corrected one -- not from the number
    that was withdrawn.
    """
    game = _game(
        [
            _drive(start=1, home=7),
            _drive(start=2, home=0),
            _drive(start=3, home=3, scoring_play=True),
        ]
    )

    (event,) = score_events(list(game.plays))

    assert (event.play_id, event.points) == ("p3", 3.0)


def test_scoring_plays_are_signed_towards_the_home_team():
    game = _game(
        [
            _drive(start=1),
            _drive(start=2, home=7, scoring_play=True),
            _drive(start=3, home=7, away=3, scoring_play=True),
        ]
    )

    assert scoring_plays(list(game.plays)) == {"p2": 7.0, "p3": -3.0}


# --- The label ---------------------------------------------------------


def test_a_snap_is_labelled_with_the_score_its_own_play_produced():
    """
    First and goal is worth what it is worth because the touchdown that
    follows is *this* snap's next score, not the one after it.
    """
    game = _game(
        [_drive(start=1, yardline=98), _drive(start=2, home=6, scoring_play=True)]
    )

    assert next_scores(game) == [
        ScoreKind.OFFENSE_TOUCHDOWN,
        ScoreKind.OFFENSE_TOUCHDOWN,
    ]


def test_a_score_by_the_other_team_is_the_defense_s():
    game = _game(
        [
            _drive(start=1),
            _drive(start=2, away=7, scoring_play=True),
        ]
    )

    assert next_scores(game)[0] == ScoreKind.DEFENSE_TOUCHDOWN


def test_the_sides_are_the_ones_at_the_snap_not_at_the_score():
    """
    `OFFENSE_*` is whoever has the ball on the snap being labelled. The away
    team's field goal is the home offense's `DEFENSE_FIELD_GOAL` and the away
    offense's `OFFENSE_FIELD_GOAL`, off the same three points.
    """
    game = _game(
        [
            _drive(start=1, offense="home"),
            _drive(start=2, offense="away"),
            _drive(start=3, offense="away", away=3, scoring_play=True),
        ]
    )

    assert next_scores(game) == [
        ScoreKind.DEFENSE_FIELD_GOAL,
        ScoreKind.OFFENSE_FIELD_GOAL,
        ScoreKind.OFFENSE_FIELD_GOAL,
    ]


def test_a_score_in_the_next_half_does_not_count_for_this_one():
    """
    The clock running out is absorbing. A drive that stalled at the two
    minute warning was worth nothing, whatever the third quarter went on to
    produce -- and a model told otherwise prices the end of a half like the
    middle of one.
    """
    game = _game(
        [
            _drive(start=1, period=2, clock=30),
            _drive(start=2, period=3, clock=800),
            _drive(start=3, period=3, clock=700, home=7, scoring_play=True),
        ]
    )

    assert next_scores(game) == [
        ScoreKind.NO_SCORE,
        ScoreKind.OFFENSE_TOUCHDOWN,
        ScoreKind.OFFENSE_TOUCHDOWN,
    ]


def test_a_scoreless_game_is_all_no_score():
    game = _game([_drive(start=1), _drive(start=2)])

    assert next_scores(game) == [ScoreKind.NO_SCORE, ScoreKind.NO_SCORE]


def test_there_is_one_label_per_snap_on_the_curve():
    """
    Aligned with `iter_states`, so a caller can zip them -- which means the
    plays it drops (no down, no clock) drop out of the labels too.
    """
    game = _game([_drive(start=1), _drive(start=2, down=None), _drive(start=3)])

    assert len(next_scores(game)) == 2


# --- The fit -----------------------------------------------------------


def test_the_baseline_is_an_expected_points_model():
    _accepts_model(_model())


def test_expected_points_follow_the_field():
    model = _model()

    backed_up, midfield, goal_line = model.predict(
        [
            make_state(yardline=5),
            make_state(yardline=50),
            make_state(yardline=95),
        ]
    )

    assert backed_up < midfield < goal_line
    # Negative deep in your own end is the part worth stating: the next score
    # there really is more often the other team's.
    assert backed_up < 0.0 < goal_line
    assert midfield == pytest.approx(0.0)
    # A classifier over what scores next can't leave the range its classes
    # cover, however strange the situation.
    assert -7.0 < backed_up and goal_line < 7.0


def test_the_probabilities_are_a_distribution():
    proba = _model().predict_proba([make_state(), make_state(yardline=90)])

    assert proba.shape == (2, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_the_expectation_is_the_probabilities_against_their_values():
    model = _model()
    states = [make_state(yardline=30)]

    expected = model.predict_proba(states) @ np.array(
        [SCORE_VALUES[kind] for kind in model.kinds]
    )

    assert model.predict(states) == pytest.approx(expected)


def test_a_lopsided_situation_does_not_overflow():
    """
    First and goal at the 1 is a genuinely enormous multinomial logit, which
    is why the softmax subtracts the row max before it exponentiates.
    """
    coefficients = np.zeros((2, len(EP_FEATURE_NAMES)))
    coefficients[0][EP_FEATURE_NAMES.index("yards_to_goal")] = -900.0
    model = MultinomialExpectedPoints(
        coefficients=coefficients,
        intercepts=np.zeros(2),
        kinds=(ScoreKind.OFFENSE_TOUCHDOWN, ScoreKind.NO_SCORE),
    )

    with np.errstate(over="raise", invalid="raise"):
        points = model.predict([make_state(yardline=1)])

    assert np.isfinite(points).all()


def test_the_wrong_shape_of_coefficients_is_refused():
    with pytest.raises(ValueError, match="classes"):
        MultinomialExpectedPoints(
            coefficients=np.zeros((2, len(EP_FEATURE_NAMES))),
            intercepts=np.zeros(3),
            kinds=(ScoreKind.NO_SCORE,) * 3,
        )


def test_a_fit_round_trips_through_json():
    model = _model()

    restored = MultinomialExpectedPoints.from_dict(
        json.loads(json.dumps(model.to_dict()))
    )

    assert restored.kinds == model.kinds
    assert np.allclose(restored.coefficients, model.coefficients)


def test_a_fit_against_other_features_is_refused():
    """
    Coefficients are positional, so a feature added to `EP_FEATURE_NAMES`
    silently reinterprets every number in an older file. It would still load
    and still predict, which is the whole problem.
    """
    saved = _model().to_dict()
    saved["feature_names"] = ["down_2", "something_else"]

    with pytest.raises(ValueError, match="different features"):
        MultinomialExpectedPoints.from_dict(saved)


def test_the_classes_are_stored_rather_than_assumed():
    """
    A fit off a short season may never have seen a defensive safety, and
    which rows are missing isn't something to infer from a row count.
    """
    saved = _model().to_dict()

    assert saved["kinds"] == [
        "offense_touchdown",
        "no_score",
        "defense_touchdown",
    ]


def test_fitting_needs_more_than_one_outcome():
    states = [make_state(yardline=y) for y in (10, 50, 90)]

    with pytest.raises(ValueError, match="nothing to fit"):
        MultinomialExpectedPoints.fit(states, [ScoreKind.NO_SCORE] * 3)


def test_a_two_class_fit_is_still_a_multinomial():
    """
    sklearn drops to binary logistic for two classes and returns one row of
    coefficients. A softmax over [0, z] is that same sigmoid, so the fit is
    rewritten with a zero row in front rather than special-cased in `predict`.
    """
    states = [make_state(yardline=y) for y in range(5, 100, 5)]
    labels = [
        ScoreKind.OFFENSE_TOUCHDOWN if state.yardline > 50 else ScoreKind.NO_SCORE
        for state in states
    ]

    model = MultinomialExpectedPoints.fit(states, labels)

    assert model.coefficients.shape == (2, len(EP_FEATURE_NAMES))
    assert len(model.intercepts) == 2
    near, far = model.predict([make_state(yardline=10), make_state(yardline=90)])
    assert near < far


def test_a_fit_recovers_the_shape_it_was_given():
    """
    Not a test of accuracy -- a test that the pieces are wired the right way
    round, on data where the answer is obvious.
    """
    states = [make_state(yardline=y) for y in range(5, 100, 5)] * 20
    labels = [
        (
            ScoreKind.OFFENSE_TOUCHDOWN
            if state.yardline > 60
            else ScoreKind.DEFENSE_TOUCHDOWN
            if state.yardline < 40
            else ScoreKind.NO_SCORE
        )
        for state in states
    ]

    model = MultinomialExpectedPoints.fit(states, labels)

    backed_up, midfield, goal_line = model.predict(
        [make_state(yardline=10), make_state(yardline=50), make_state(yardline=90)]
    )
    assert backed_up < midfield < goal_line
    assert backed_up < 0.0 < goal_line


def test_fitting_refuses_labels_that_do_not_line_up():
    with pytest.raises(ValueError, match="line up"):
        MultinomialExpectedPoints.fit([make_state()], [])
