from collections import Counter

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
    lucky_wp,
    lucky_wp_from_states,
    records_defended_passes,
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
        given = self._probabilities[: len(states)]
        # A gate-passing game needs a dozen incompletions in it, and a test
        # that cared about each one's probability would be a test about
        # padding. The tail holds the last value it was given.
        tail = [given[-1]] * (len(states) - len(given))
        return np.array(given + tail, dtype=float)


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


def _pass_game(*specs, incompletions: int = 12) -> GamePlays:
    """
    A game with enough incompletions for `records_defended_passes` to have an
    opinion about its feed, padded with passes nobody got near.

    The padding is deliberately undefended, so whether the gate opens is
    decided by the plays the test actually wrote.
    """
    rows = list(specs)
    while sum(1 for text, _ in rows if text and "incomplete" in text.lower()) < (
        incompletions
    ):
        rows.append(("Pass incomplete short right to Smith.", False))
    return _snaps(*rows)


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


def test_an_interception_is_a_ball_a_defender_came_down_with():
    """
    Every interception, not only the deflected ones -- the manner isn't in
    the text and the population that is measurable is "a defender reached
    it". Its 0.20 is the measured share of those that end up caught.
    """
    game = _pass_game(
        ("Pass intercepted by Slay at PHI 40", True),
        ("Pass incomplete deep left to Brown, broken up by Slay", False),
        ("Pass incomplete short right to Smith, broken up by Bates", False),
    )

    kinds = {lucky.kind for lucky in find_lucky_plays(game.plays)}

    assert kinds == {
        LuckKind.PASS_DEFENDED_INTERCEPTION,
        LuckKind.PASS_DEFENDED_INCOMPLETE,
    }
    (pick,) = [
        lucky
        for lucky in find_lucky_plays(game.plays)
        if lucky.kind is LuckKind.PASS_DEFENDED_INTERCEPTION
    ]
    assert pick.retained == 0.20
    assert pick.changed_possession is True


def test_a_broken_up_pass_is_the_interception_that_wasnt():
    """The other side of the same coin, and the reason the first can be used."""
    game = _pass_game(
        ("Pass incomplete deep left to Brown, broken up by Slay", False),
        ("Pass incomplete short right to Smith, broken up by Bates", False),
    )

    lucky = find_lucky_plays(game.plays)

    assert [play.kind for play in lucky] == [LuckKind.PASS_DEFENDED_INCOMPLETE] * 2
    assert all(play.retained == 0.80 for play in lucky)
    assert all(play.changed_possession is False for play in lucky)


def test_the_nfl_writes_the_defender_in_punctuation():
    """
    Gamebook text names the defender in a trailing parenthetical rather than
    in words. On an incompletion there is nobody to tackle, so that credit is
    unambiguous -- which is the only case this reads it in.
    """
    game = _pass_game(
        (
            "(12:11) (Shotgun) J.Goff pass incomplete deep left to C.Kupp (J.Ramsey).",
            False,
        ),
        ("(9:02) T.Brady pass incomplete short right to J.White (D.White).", False),
    )

    lucky = find_lucky_plays(game.plays)

    assert [play.kind for play in lucky] == [LuckKind.PASS_DEFENDED_INCOMPLETE] * 2


def test_a_pass_nobody_touched_is_not_a_coin_flip():
    """An overthrow is a throw, not a bounce. Only the ones a defender reached."""
    game = _pass_game(
        ("Pass incomplete deep left to Brown, broken up by Slay", False),
        ("Pass incomplete short right to Smith, broken up by Bates", False),
    )

    lucky = find_lucky_plays(game.plays)

    assert len(lucky) == 2, "only the two the text says were defended"


def test_an_interception_in_a_game_that_records_nothing_is_left_alone():
    """
    The gate. A feed that doesn't write down broken-up passes still writes
    down every interception, and adjusting those alone would charge a defense
    for its picks and credit it for none of the balls it got a hand on.
    """
    game = _pass_game(("Pass intercepted by Slay at PHI 40", True))

    assert find_lucky_plays(game.plays) == []


def test_an_interception_that_was_later_fumbled_is_still_an_interception():
    """
    Both words are in the sentence. The play is the interception -- reading it
    as a fumble would price the wrong branch, and get the direction of the
    possession flip backwards.
    """
    game = _pass_game(
        ("Pass intercepted by Slay, FUMBLE on the return, recovered by PHI", True),
        ("Pass incomplete deep left to Brown, broken up by Slay", False),
        ("Pass incomplete short right to Smith, broken up by Bates", False),
    )

    (pick, *_) = find_lucky_plays(game.plays)

    assert pick.kind is LuckKind.PASS_DEFENDED_INTERCEPTION


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


def _possession(*plays) -> GamePlays:
    """
    A game of snaps that say who had the ball, for the possession witness.

    Each argument is `(offense, text, is_turnover)`. `_snaps` leaves every
    play with the same offense, which is the right default for the tests that
    are about text -- these are the ones about what happened after.
    """
    return _game(
        [
            make_play(
                play_id=f"p{number}",
                play_number=number,
                clock_seconds=900 - 30 * number,
                offense_team_id=offense,
                defense_team_id="away" if offense == "home" else "home",
                text=text,
                is_turnover=is_turnover,
            )
            for number, (offense, text, is_turnover) in enumerate(plays, start=1)
        ]
    )


# --- Who ended up with the ball ----------------------------------------


def test_a_sack_fumble_the_defense_recovered_is_a_lost_fumble():
    """
    The bug the second witness is here for. `is_turnover` is false on a sack
    the defense stripped and recovered -- systematically, in both leagues --
    and taking it at its word prices the wrong branch: it would build the
    counterfactual where the offense *lost* the ball it already lost.
    """
    game = _possession(
        ("home", None, False),
        ("home", "A.Dalton sacked. FUMBLES, RECOVERED by SEA-A.Woods.", False),
        ("away", None, False),
    )

    (lucky,) = find_lucky_plays(game.plays)

    assert lucky.kind is LuckKind.FUMBLE_LOST
    assert lucky.changed_possession is True


def test_a_fumble_the_offense_kept_still_reads_as_kept():
    """The other half: the next snap is the same team's, so nothing changed."""
    game = _possession(
        ("home", None, False),
        ("home", "K.Cousins FUMBLES (Aborted) at MIN 27, and recovers.", False),
        ("home", None, False),
    )

    (lucky,) = find_lucky_plays(game.plays)

    assert lucky.kind is LuckKind.FUMBLE_KEPT
    assert lucky.changed_possession is False


def test_a_turnover_flag_that_says_yes_is_taken_at_its_word():
    """
    Only the false is checked. The column has no false positives worth
    guarding against, and trusting the yes is what lets a fumble on the last
    play of a game -- nothing after it to witness -- still be classified.
    """
    game = _possession(("home", "FUMBLE, recovered by DAL.", True))

    (lucky,) = find_lucky_plays(game.plays)

    assert lucky.kind is LuckKind.FUMBLE_LOST


def test_a_null_turnover_flag_is_answered_by_the_next_snap():
    """
    Previously dropped outright. NCAAFB before 2014 is the case: the column
    is no help, and the next snap says plainly who has the ball.
    """
    game = _possession(
        ("home", "Shai Werts fumbled, recovered by LSU Patrick Queen", None),
        ("away", None, False),
    )

    (lucky,) = find_lucky_plays(game.plays)

    assert lucky.kind is LuckKind.FUMBLE_LOST


def test_a_fumble_no_witness_can_call_is_still_dropped():
    """Nothing after it and a null column -- guessing is worse than skipping."""
    game = _possession(("home", "Barkley rush for 3 yards. FUMBLE.", None))

    assert find_lucky_plays(game.plays) == []


def test_the_plays_between_do_not_confuse_the_witness():
    """
    A timeout has no offense at all, so the witness looks past it to the next
    real snap rather than reading the gap as a change of possession.
    """
    game = _game(
        [
            make_play(play_id="p1", play_number=1, offense_team_id="home"),
            make_play(
                play_id="p2",
                play_number=2,
                offense_team_id="home",
                text="Hurts rush. FUMBLE, recovered by PHI.",
                is_turnover=False,
            ),
            make_play(
                play_id="p3", play_number=3, offense_team_id=None, play_type="Timeout"
            ),
            make_play(play_id="p4", play_number=4, offense_team_id="home"),
        ]
    )

    (lucky,) = find_lucky_plays(game.plays)

    assert lucky.kind is LuckKind.FUMBLE_KEPT


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


# --- Lucky WP ----------------------------------------------------------


def test_a_coin_flip_hands_out_half_of_what_it_swung():
    """
    Dallas comes up with the ball and the model drops Philadelphia from 0.5 to
    0.2. Had Philadelphia recovered it, the fit says 0.6. Half of that gap is
    the recovery being a coin flip, and half of it is Dallas playing the down.
    """
    game = _snaps(
        (None, False),
        ("Barkley rush for 3 yards. FUMBLE, recovered by DAL.", True),
        (None, False),
    )

    breaks = lucky_wp(_FixedModel([0.5, 0.5, 0.2], counterfactual=0.6), game)

    (swing,) = breaks.swings
    assert swing.kind is LuckKind.FUMBLE_LOST
    assert (swing.realized, swing.counterfactual) == (0.2, 0.6)
    # 0.5 * 0.2 + 0.5 * 0.6, and the realized 0.2 is 0.2 below it.
    assert swing.expected == pytest.approx(0.4)
    assert swing.home_delta == pytest.approx(-0.2)
    assert (breaks.home, breaks.away) == pytest.approx((0.0, 0.2))
    assert breaks.net == pytest.approx(-0.2)


def test_an_interception_hands_out_four_fifths_of_its_swing():
    """
    The same swing off a less likely branch is more of a gift. Four balls in
    five that a defender gets to fall incomplete, so the one that was caught
    was mostly not the defense's doing.
    """
    game = _pass_game(
        (None, False),
        ("Pass intercepted by Slay at PHI 40", True),
        ("Pass incomplete deep left to Brown, broken up by Slay", False),
        ("Pass incomplete short right to Smith, broken up by Bates", False),
    )

    breaks = lucky_wp(_FixedModel([0.5, 0.5, 0.2], counterfactual=0.6), game)

    pick = next(
        swing
        for swing in breaks.swings
        if swing.kind is LuckKind.PASS_DEFENDED_INTERCEPTION
    )
    assert pick.retained == 0.20
    # 0.2 * 0.2 + 0.8 * 0.6, and the realized 0.2 is 0.32 below it.
    assert pick.expected == pytest.approx(0.52)
    assert pick.home_delta == pytest.approx(-0.32)


def test_both_teams_can_get_a_break_in_the_same_game():
    """
    Why the two sides are totalled separately rather than netted. This game
    swung twice, hard, in opposite directions -- `net` alone would report it as
    the same game as one where nothing bounced at all.
    """
    game = _snaps(
        ("Prescott rush. FUMBLE, recovered by PHI.", True),
        (None, False),
        ("Barkley rush. FUMBLE, recovered by DAL.", True),
        (None, False),
    )

    breaks = lucky_wp(_FixedModel([0.5, 0.9, 0.9, 0.1], counterfactual=0.5), game)

    assert (breaks.home, breaks.away) == pytest.approx((0.2, 0.2))
    assert breaks.net == pytest.approx(0.0)
    # And the trap this type exists to avoid: it is not a share of anything.
    assert breaks.home + breaks.away != pytest.approx(1.0)


def test_a_game_with_no_bounces_in_it_hands_out_nothing():
    game = _snaps((None, False), ("Barkley rush for 3 yards", False), (None, False))

    breaks = lucky_wp(_FixedModel([0.4, 0.6, 0.7]), game)

    assert breaks.swings == []
    assert (breaks.home, breaks.away, breaks.net) == (0.0, 0.0, 0.0)


def test_retaining_everything_hands_out_nothing_either():
    """
    The same identity setting the adjusted curve has, on the other number: a
    bounce that was always going to happen that way was not a bounce. The
    plays are still reported -- only their share of the swing is zero.
    """
    game = _pass_game(
        ("FUMBLE, recovered by DAL.", True),
        ("Pass intercepted by Slay at PHI 40", True),
        ("Pass incomplete deep left to Brown, broken up by Slay", False),
        ("Pass incomplete short right to Smith, broken up by Bates", False),
        (None, False),
    )
    everything = {kind: 1.0 for kind in LuckKind}

    breaks = lucky_wp(
        _FixedModel([0.4, 0.55, 0.7, 0.6], counterfactual=0.01),
        game,
        retained=everything,
    )

    assert len(breaks.swings) >= 3
    assert all(swing.home_delta == pytest.approx(0.0) for swing in breaks.swings)
    assert (breaks.home, breaks.away) == pytest.approx((0.0, 0.0))


def test_the_pass_pair_can_be_left_out_of_the_total():
    """
    The dial, on a whole coin rather than one side of it. Turning the pass
    pair off has to turn off *both* of its kinds -- zeroing only the
    interception would leave the credit without the charge, which is the same
    error as the gate exists to prevent, pointed the other way.
    """
    game = _pass_game(
        ("FUMBLE, recovered by DAL.", True),
        ("Pass intercepted by Slay at PHI 40", True),
        ("Pass incomplete deep left to Brown, broken up by Slay", False),
        ("Pass incomplete short right to Smith, broken up by Bates", False),
        (None, False),
    )
    fumbles_only = {
        **DEFAULT_RETAINED,
        LuckKind.PASS_DEFENDED_INTERCEPTION: 1.0,
        LuckKind.PASS_DEFENDED_INCOMPLETE: 1.0,
    }

    both = lucky_wp(_FixedModel([0.5, 0.3, 0.1, 0.2], counterfactual=0.7), game)
    fumbles = lucky_wp(
        _FixedModel([0.5, 0.3, 0.1, 0.2], counterfactual=0.7),
        game,
        retained=fumbles_only,
    )

    # 0.5 * (0.3 - 0.7) from the fumble, and nothing from the passes.
    assert fumbles.away == pytest.approx(0.2)
    assert both.home + both.away > fumbles.home + fumbles.away


def test_a_bounce_on_the_last_snap_has_nothing_to_price_it_against():
    """
    The same play the adjusted curve drops, dropped for the same reason: the
    branch that happened is read off the *next* snap, and there isn't one.
    """
    game = _snaps((None, False), ("FUMBLE, recovered by DAL.", True))
    model = _FixedModel([0.4, 0.6], counterfactual=0.9)

    breaks = lucky_wp(model, game)

    assert breaks.swings == []
    assert (breaks.home, breaks.away) == (0.0, 0.0)


def test_both_numbers_are_talking_about_the_same_plays():
    """
    The contract `_priced_branches` exists for. The two metrics are free to
    disagree about what a bounce was worth; a game where they disagree about
    which plays bounced is a bug in one of them.
    """
    game = _snaps(
        ("FUMBLE, recovered by DAL.", True),
        (None, False),
        ("Pass intercepted by Slay (tipped by Sweat)", True),
        ("FUMBLE, recovered by PHI.", False),
    )
    probabilities = [0.5, 0.4, 0.4, 0.8]

    adjusted = luck_adjusted_curve(_FixedModel(probabilities, 0.3), game)
    breaks = lucky_wp(_FixedModel(probabilities, 0.3), game)

    assert [swing.play_id for swing in breaks.swings] == [
        lucky.play_id for lucky in adjusted.lucky_plays
    ]


def test_states_a_caller_already_has_give_the_same_total():
    """The backend path, for the same reason `adjusted_curve_from_states` is."""
    game = _snaps(
        (None, False),
        ("FUMBLE, recovered by DAL.", True),
        (None, False),
    )
    probabilities = [0.5, 0.5, 0.8]

    walked = lucky_wp(_FixedModel(probabilities, 0.2), game)
    from_states = lucky_wp_from_states(
        _FixedModel(probabilities, 0.2),
        list(iter_states(game)),
        find_lucky_plays(game.plays),
    )

    assert walked == from_states


def test_the_total_is_win_probability_not_a_share_of_the_clock():
    """
    A bounce late in a close game is worth more than the same bounce in the
    first quarter, and this number says so directly -- there is no weighting
    by how long the situation stood, because nothing here is an average.
    """
    early = _snaps(
        ("FUMBLE, recovered by DAL.", True),
        (None, False),
        (None, False),
    )
    late = _snaps(
        (None, False),
        (None, False),
        ("FUMBLE, recovered by DAL.", True),
        (None, False),
    )

    # Same shape of swing in both, so the only difference is when it happened
    # -- and it isn't a difference this number makes.
    assert lucky_wp(_FixedModel([0.5, 0.2, 0.2], 0.6), early).away == pytest.approx(
        lucky_wp(_FixedModel([0.5, 0.5, 0.5, 0.2], 0.6), late).away
    )


# --- Whether a game's feed says enough ---------------------------------


def test_a_feed_that_records_defended_passes_opens_the_gate():
    game = _pass_game(
        ("Pass incomplete deep left to Brown, broken up by Slay", False),
        ("Pass incomplete short right to Smith, broken up by Bates", False),
    )

    assert records_defended_passes(game.plays) is True


def test_a_feed_that_records_none_of_them_keeps_it_shut():
    """
    NCAAFB 2022-24 for most venues, and the NFL before 2009. Every
    interception is still in the text; none of them is adjustable.
    """
    game = _pass_game()

    assert records_defended_passes(game.plays) is False


def test_recording_a_few_of_them_is_not_enough():
    """
    Presence isn't the test, the rate is. A venue that writes down a third of
    its breakups would leak in proportion to the two thirds it didn't -- the
    charge on interceptions is whole either way, and only the credit shrinks.
    """
    game = _pass_game(
        ("Pass incomplete deep left to Brown, broken up by Slay", False),
        incompletions=40,
    )

    assert records_defended_passes(game.plays) is False


def test_a_game_with_barely_any_passing_says_nothing_either_way():
    """
    Below `MIN_INCOMPLETIONS` the absence of a marker is not evidence of a
    feed that doesn't write them, just of a game that didn't throw.
    """
    game = _snaps(
        ("Pass incomplete short right to Smith, broken up by Bates", False),
        (None, False),
    )

    assert records_defended_passes(game.plays) is False


def test_the_floor_is_a_keyword_like_the_weights_are():
    game = _pass_game(
        ("Pass incomplete deep left to Brown, broken up by Slay", False),
        incompletions=40,
    )

    assert records_defended_passes(game.plays, floor=0.02) is True


def test_the_gate_decides_both_sides_of_the_coin_together():
    """
    The property the whole design rests on. A shut gate drops the defended
    incompletions *and* the interceptions; it can never drop one and keep the
    other, because that is the biased half.
    """
    shut = _pass_game(("Pass intercepted by Slay at PHI 40", True))
    open_ = _pass_game(
        ("Pass intercepted by Slay at PHI 40", True),
        ("Pass incomplete deep left to Brown, broken up by Slay", False),
        ("Pass incomplete short right to Smith, broken up by Bates", False),
    )

    assert [lucky.kind for lucky in find_lucky_plays(shut.plays)] == []
    kinds = Counter(lucky.kind for lucky in find_lucky_plays(open_.plays))
    assert kinds[LuckKind.PASS_DEFENDED_INTERCEPTION] == 1
    assert kinds[LuckKind.PASS_DEFENDED_INCOMPLETE] == 2


def test_a_shut_gate_still_adjusts_the_fumbles():
    """
    The gate is about the pass coin only. A game whose feed says nothing about
    defended passes still has both branches of every fumble in its text, so it
    gets the smaller adjustment rather than none.
    """
    game = _pass_game(
        ("Barkley rush for 3 yards. FUMBLE, recovered by DAL.", True),
        ("Pass intercepted by Slay at PHI 40", True),
    )

    assert [lucky.kind for lucky in find_lucky_plays(game.plays)] == [
        LuckKind.FUMBLE_LOST
    ]
