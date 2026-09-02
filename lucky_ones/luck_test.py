import numpy as np
import pytest

from .arrow import table_to_plays
from .conftest import make_play, make_state, make_table
from .curve import game_control, win_probability_curve
from .game import GamePlays
from .luck import (
    DEFAULT_RETAINED,
    LuckKind,
    _counterfactual,
    adjusted_curve_from_states,
    find_lucky_plays,
    luck_adjusted_curve,
    luck_adjusted_game_control,
)
from .state import iter_states


class _FixedModel:
    """
    A model that reads its answer off the state rather than fitting one, so a
    test can say what the curve and the counterfactual are both worth.

    `probabilities` is consumed in order for the real snaps; anything the
    adjustment asks about afterwards -- the counterfactual states, which are
    always predicted in one batch of their own -- gets `counterfactual`.
    """

    def __init__(self, probabilities, counterfactual: float = 0.5) -> None:
        self._probabilities = list(probabilities)
        self._counterfactual = counterfactual
        self.spent = False

    def predict(self, states) -> np.ndarray:
        if self.spent:
            return np.full(len(states), self._counterfactual, dtype=float)
        self.spent = True
        return np.array(self._probabilities[: len(states)], dtype=float)


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


def _snaps(*texts_and_turnovers) -> GamePlays:
    """
    A game of ordinary snaps, one per argument, each 30 seconds apart.

    Each argument is `(text, is_turnover)`; `(None, False)` is a play with
    nothing interesting in it.
    """
    return _game(
        [
            make_play(
                play_id=f"p{number}",
                play_number=number,
                clock_seconds=900 - 30 * number,
                text=text,
                is_turnover=is_turnover,
            )
            for number, (text, is_turnover) in enumerate(texts_and_turnovers, start=1)
        ]
    )


# --- Finding them ------------------------------------------------------


def test_a_lost_fumble_is_the_other_team_getting_lucky():
    game = _snaps(("Barkley rush for 3 yards. FUMBLE, recovered by DAL.", True))

    (lucky,) = find_lucky_plays(game.plays)

    assert lucky.kind is LuckKind.FUMBLE_LOST
    assert lucky.changed_possession is True
    assert lucky.retained == 0.5


def test_a_recovered_fumble_is_the_offense_getting_lucky():
    """
    The near-miss the play text does record, and the reason the metric can
    charge a team for the ones it got away with.
    """
    game = _snaps(("Hurts rush for 2 yards. FUMBLE, recovered by PHI.", False))

    (lucky,) = find_lucky_plays(game.plays)

    assert lucky.kind is LuckKind.FUMBLE_KEPT
    assert lucky.changed_possession is False


@pytest.mark.parametrize(
    "text",
    [
        "Pass intercepted by Slay at PHI 40 (tipped by Sweat)",
        "Pass INTERCEPTED, ball deflected at the line",
        "Pass batted by Carter, intercepted by Blankenship",
    ],
)
def test_a_tipped_interception_keeps_a_quarter_of_its_swing(text):
    (lucky,) = find_lucky_plays(_snaps((text, True)).plays)

    assert lucky.kind is LuckKind.TIPPED_INTERCEPTION
    assert lucky.retained == 0.25


def test_an_untipped_interception_is_a_decision_not_a_bounce():
    game = _snaps(("Pass intercepted by Slay at PHI 40", True))

    assert find_lucky_plays(game.plays) == []


def test_an_interception_that_was_later_fumbled_is_still_an_interception():
    """
    Both words are in the sentence. The play is the interception -- reading it
    as a fumble would price the wrong branch, and get the direction of the
    possession flip backwards.
    """
    game = _snaps(
        ("Pass intercepted by Slay, FUMBLE on the return, recovered by PHI", True)
    )

    assert find_lucky_plays(game.plays) == []


def test_a_fumble_wiped_out_by_a_penalty_never_happened():
    game = _snaps(("PENALTY holding. FUMBLE, recovered by DAL. No Play.", True))

    assert find_lucky_plays(game.plays) == []


def test_a_fumble_with_no_turnover_flag_is_left_alone():
    """
    Which branch happened is the whole input to the adjustment. Guessing it
    is worse than not adjusting the play.
    """
    game = _snaps(("Barkley rush for 3 yards. FUMBLE.", None))

    assert find_lucky_plays(game.plays) == []


def test_a_play_with_no_text_says_nothing():
    assert find_lucky_plays(_snaps((None, True)).plays) == []


def test_the_weights_are_a_keyword_not_a_constant():
    game = _snaps(("FUMBLE, recovered by DAL.", True))

    (lucky,) = find_lucky_plays(game.plays, retained={LuckKind.FUMBLE_LOST: 0.9})

    assert lucky.retained == 0.9


# --- The counterfactual ------------------------------------------------


def test_the_branch_after_a_turnover_is_keeping_it_on_the_next_down():
    state = make_state(offense_is_home=True, down=2, distance=7, yardline=40)

    branch = _counterfactual(state, changed_possession=True)

    assert (branch.down, branch.distance, branch.yardline) == (3, 7, 40)
    assert branch.offense_is_home is True


def test_a_fourth_down_the_offense_held_onto_is_a_first_down():
    branch = _counterfactual(make_state(down=4, distance=2), changed_possession=True)

    assert (branch.down, branch.distance) == (1, 10)


def test_the_branch_after_keeping_it_is_the_other_team_first_and_ten():
    """
    `yardline` is measured from the offense's own goal line, so the ball
    sitting still while possession flips means the number mirrors.
    """
    state = make_state(offense_is_home=True, down=2, distance=7, yardline=40)

    branch = _counterfactual(state, changed_possession=False)

    assert branch.offense_is_home is False
    assert (branch.down, branch.distance, branch.yardline) == (1, 10, 60)


def test_the_counterfactual_keeps_the_score_off_the_realized_play():
    """
    A scoop and score didn't score in the branch where the offense recovered,
    so the branch is built from the snap and carries the snap's score.
    """
    state = make_state(home_score=7, away_score=3)

    branch = _counterfactual(state, changed_possession=True)

    assert (branch.home_score, branch.away_score) == (7, 3)


# --- Adjusting the curve -----------------------------------------------


def test_a_lucky_recovery_is_worth_half_of_what_it_swung():
    """
    The home team fumbles and gets it back, and the model likes them a lot
    more for it than it would have if Dallas had come up with the ball. Half
    the difference between those two is the part they earned.
    """
    game = _snaps(
        (None, False),
        ("Hurts rush for 2 yards. FUMBLE, recovered by PHI.", False),
        (None, False),
    )
    model = _FixedModel([0.5, 0.5, 0.9], counterfactual=0.1)

    adjusted = luck_adjusted_curve(model, game)

    assert [lucky.kind for lucky in adjusted.lucky_plays] == [LuckKind.FUMBLE_KEPT]
    assert [point.home_win_probability for point in adjusted.realized] == [
        0.5,
        0.5,
        0.9,
    ]
    # The blend is 0.5 * 0.9 + 0.5 * 0.1, and the two points before it are
    # untouched because nothing before the fumble was luck.
    assert [point.home_win_probability for point in adjusted.points] == pytest.approx(
        [0.5, 0.5, 0.5]
    )


def test_the_discount_stays_on_the_board_for_the_rest_of_the_game():
    """
    A fumble returned for a touchdown in the first quarter is still on the
    scoreboard in the fourth. The adjusted curve carries the difference
    forward rather than snapping back to what happened.
    """
    game = _snaps(
        ("Barkley rush for 3 yards. FUMBLE, recovered by DAL.", True),
        (None, False),
        (None, False),
    )
    model = _FixedModel([0.5, 0.2, 0.2], counterfactual=0.6)

    adjusted = luck_adjusted_curve(model, game)

    # 0.5 * 0.2 + 0.5 * 0.6 = 0.4 at the snap after the fumble, and the last
    # point moved by exactly as much as the one before it, in log-odds.
    assert adjusted.points[1].home_win_probability == pytest.approx(0.4)
    assert adjusted.points[2].home_win_probability == pytest.approx(0.4)
    assert adjusted.realized[2].home_win_probability == pytest.approx(0.2)


def test_the_adjusted_curve_is_not_a_prediction():
    """
    The real curve ends where the game ended. The counterfactual one doesn't,
    and shouldn't -- that is the number being reported.
    """
    game = _snaps(
        ("Barkley rush. FUMBLE, recovered by DAL.", True),
        (None, False),
    )
    model = _FixedModel([0.5, 0.99], counterfactual=0.0)

    adjusted = luck_adjusted_curve(model, game)

    assert adjusted.realized[-1].home_win_probability == pytest.approx(0.99)
    assert adjusted.points[-1].home_win_probability < 0.99


def test_the_adjusted_probability_stays_a_probability():
    """
    The accumulation is in log-odds precisely so this holds without a clamp:
    a run of discounts against a curve already at 0.999 can't push a point
    outside (0, 1).
    """
    game = _snaps(
        *[("FUMBLE, recovered by DAL.", True)] * 6,
        (None, False),
    )
    model = _FixedModel([0.999] * 7, counterfactual=1e-9)

    for point in luck_adjusted_curve(model, game).points:
        assert 0.0 < point.home_win_probability < 1.0


def test_a_game_with_no_luck_in_it_is_the_curve_unchanged():
    game = _snaps((None, False), ("Barkley rush for 3 yards", False), (None, False))
    model = _FixedModel([0.4, 0.6, 0.7])

    adjusted = luck_adjusted_curve(model, game)

    assert adjusted.lucky_plays == []
    assert adjusted.points == adjusted.realized


def test_retaining_everything_reproduces_the_curve():
    """
    The dial's identity setting. If this drifts, the difference between the
    two numbers isn't the luck -- it's the arithmetic.
    """
    game = _snaps(
        ("FUMBLE, recovered by DAL.", True),
        ("Pass intercepted by Slay (tipped by Sweat)", True),
        (None, False),
    )
    model = _FixedModel([0.4, 0.55, 0.7], counterfactual=0.01)
    everything = {kind: 1.0 for kind in LuckKind}

    adjusted = luck_adjusted_curve(model, game, retained=everything)

    assert [point.home_win_probability for point in adjusted.points] == pytest.approx(
        [point.home_win_probability for point in adjusted.realized]
    )


def test_luck_on_the_last_snap_has_nothing_left_to_move():
    game = _snaps((None, False), ("FUMBLE, recovered by DAL.", True))
    model = _FixedModel([0.4, 0.6], counterfactual=0.9)

    adjusted = luck_adjusted_curve(model, game)

    assert adjusted.lucky_plays == []
    assert adjusted.points == adjusted.realized


def test_a_game_with_no_scrimmage_plays_has_no_adjusted_curve():
    game = _game([make_play(down=None, offense_team_id=None, yardline=None)])

    adjusted = luck_adjusted_curve(_FixedModel([0.5]), game)

    assert adjusted.points == [] and adjusted.realized == []


def test_a_kickoff_fumble_is_not_on_the_curve_to_adjust():
    """
    `iter_states` keeps scrimmage snaps only, so a kickoff has no state to
    build a counterfactual from. `find_lucky_plays` still reports it -- it is
    a bounce -- and the adjustment leaves it alone.
    """
    game = _game(
        [
            make_play(play_id="k", play_number=1, down=None, yardline=None),
            make_play(
                play_id="ko",
                play_number=2,
                down=None,
                yardline=None,
                play_type="Kickoff Return",
                text="MUFFED catch. FUMBLE, recovered by DAL.",
                is_turnover=True,
            ),
            make_play(play_id="s", play_number=3, clock_seconds=800),
        ]
    )
    model = _FixedModel([0.5])

    assert [lucky.play_id for lucky in find_lucky_plays(game.plays)] == ["ko"]
    assert luck_adjusted_curve(model, game).lucky_plays == []


# --- Game control ------------------------------------------------------


def test_control_and_luck_adjusted_control_answer_different_questions():
    """
    The case the whole module is for: the home team is behind all game, gets a
    fumble to fall their way, and finishes ahead. Control says they were in
    it; luck-adjusted control says less of that was theirs.
    """
    game = _snaps(
        (None, False),
        ("Prescott rush for 3 yards. FUMBLE, recovered by PHI.", True),
        (None, False),
    )
    model = _FixedModel([0.3, 0.3, 0.9], counterfactual=0.2)

    control = game_control(win_probability_curve(model, game))
    adjusted = luck_adjusted_game_control(_FixedModel([0.3, 0.3, 0.9], 0.2), game)

    assert control is not None and adjusted is not None
    assert adjusted.home < control.home
    assert adjusted.home + adjusted.away == pytest.approx(1.0)
    assert adjusted.seconds == control.seconds


def test_adjusted_control_is_none_where_control_is():
    """Same games, same answer -- a caller reporting the pair can't get one
    number and a None."""
    game = _game([make_play(period=5, clock_seconds=0)])

    assert luck_adjusted_game_control(_FixedModel([0.5]), game) is None


def test_states_a_caller_already_has_give_the_same_answer():
    """
    The backend path. `adjusted_curve_from_states` exists so a service that
    walked the game once doesn't walk it again; if it disagreed with
    `luck_adjusted_curve`, the served curve and the checked one would differ.
    """
    game = _snaps(
        (None, False),
        ("FUMBLE, recovered by DAL.", True),
        (None, False),
    )
    probabilities = [0.5, 0.5, 0.8]

    walked = luck_adjusted_curve(_FixedModel(probabilities, 0.2), game)
    from_states = adjusted_curve_from_states(
        _FixedModel(probabilities, 0.2),
        list(iter_states(game)),
        find_lucky_plays(game.plays),
    )

    assert walked.points == from_states.points


def test_the_shipped_weights_cover_every_kind():
    """
    `find_lucky_plays` falls back to `DEFAULT_RETAINED` for a kind a caller's
    mapping doesn't mention, so a kind missing from it is a KeyError at
    classification time rather than here.
    """
    assert set(DEFAULT_RETAINED) == set(LuckKind)
    assert all(0.0 <= weight <= 1.0 for weight in DEFAULT_RETAINED.values())
