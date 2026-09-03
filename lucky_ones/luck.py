"""
The coin flips: finding them, and the two ways of reporting them.

`lucky_ones.curve.game_control` measures what happened. That is the honest
answer to "who controlled this game", and it is also the answer that hands a
team full credit for a fumble that bounced their way -- an outcome nobody on
the field decided. This module is what to say about that, and there are two
things worth saying, so there are two numbers:

- `luck_adjusted_game_control` -- *how much of the game would this team have
  controlled if the fifty-fifty balls had gone fifty-fifty?* A rewrite of the
  game: the curve is redrawn with each bounce replaced by its average, and
  `game_control` is taken over the result. Read next to the real number, and
  in the same units.
- `lucky_wp` -- *how much win probability did the bounces hand each team?* An
  accounting of the game rather than a rewrite: each bounce contributes once,
  the curve is left alone, and the totals are win probability rather than a
  share of anything. Read on its own, as the size of the breaks.

They are the same three steps up to the last one, and they share the first
two exactly (`_priced_branches`), so they can differ about what a bounce was
worth but never about which plays bounced:

1. **Find the coin flips.** `find_lucky_plays` reads `Play.text` for the
   handful of plays whose result was decided by a bounce -- a fumble, and an
   interception off a tipped ball. See `LuckKind` for the whole list and why
   it is short.
2. **Ask the model what the other branch was worth.** Every play here has
   exactly two outcomes that matter: the ball changed hands or it didn't. The
   one that didn't happen is a `GameState` we can build (`_counterfactual`)
   and the model can price, so the counterfactual is the model's own number
   rather than a constant someone picked.
3. **Then either replace the swing with the average of the two** -- the play's
   win probability move becomes `retained * what happened + (1 - retained) *
   the other branch`, and every later point on the curve carries the
   difference forward -- **or bank the difference between the two** as the
   part of the swing the bounce is responsible for, and total it per team.

`retained` is a probability: how often the outcome that happened is the one
that happens. A fumble is recovered by either side about half the time, so
0.5 -- half the credit for the recovery, and half of the branch where it went
the other way. A tipped pass is intercepted a good deal less often than it
falls incomplete, so a tipped interception keeps 0.25 of its swing.

Three things this deliberately is not:

- **A prediction.** The adjusted curve does not end at 0 or 1 the way the
  real one does. The game had a winner; this is a counterfactual, and its
  last point is where the game *would* have stood.
- **Symmetric in what it can see.** A fumble the offense recovered is in the
  play text, so the team that got away with one is charged for it. A pass
  that was tipped and *not* intercepted mostly isn't in the text at all, so
  it can't be. How much that costs depends on which number you are reading,
  and it is not the same answer for both -- see the second bullet in
  `lucky_wp`, where it costs the most.
- **An adjustment to the model.** The fit is untouched. Both numbers are read
  off the curve that fit produces; retrain nothing.
"""

import re
from enum import StrEnum
from typing import Iterable, Mapping, NamedTuple, Sequence

import numpy as np

from .curve import CurvePoint, GameControl, curve_from_states, game_control
from .game import GamePlays
from .model import WinProbabilityModel
from .plays import Play
from .state import GameState, iter_states


class LuckKind(StrEnum):
    """
    The kinds of play this discounts.

    A short list on purpose. Every entry has to clear three bars: the play
    text says it happened, the outcome that didn't happen is one this module
    can build a `GameState` for, and **its opposite is on the list too**.

    That last one is what keeps this honest. Each pair is one coin: a fumble
    lost and a fumble kept, an interception and the pass that was defended
    and fell. Adjust one side without the other and the metric stops netting
    to zero in expectation and becomes a levy on whoever the recorded side
    happens to fall against -- every defense charged for its interceptions
    and none credited for the balls it got a hand on. Both sides or neither.

    What that rules out, and why each is left in the realized number rather
    than guessed at:

    - **Muffed punts and kickoff fumbles.** `iter_states` keeps scrimmage
      snaps only, so a kickoff isn't on the curve to adjust, and a punt's
      counterfactual isn't "the punting team keeps it on the next down".
    - **Blocked kicks.** The block is a play someone made; only the bounce
      after it is luck, and the two aren't separable from the text.
    - **A field goal off the upright.** No branch to build: the counterfactual
      to a 51-yarder that went in is a 51-yarder that didn't, which is the
      same state with three fewer points and a different spot.
    - **A pass nobody touched.** An overthrow is not a coin flip. Only the
      passes a defender is recorded as having reached are on the list, which
      is also the only population whose two branches are both observable.
    """

    FUMBLE_LOST = "fumble_lost"
    """A fumble the other team came up with."""

    FUMBLE_KEPT = "fumble_kept"
    """A fumble the offense recovered -- the one that got away with it."""

    PASS_DEFENDED_INTERCEPTION = "pass_defended_interception"
    """An interception: the ball a defender got to and came down with."""

    PASS_DEFENDED_INCOMPLETE = "pass_defended_incomplete"
    """The same ball, broken up instead -- the interception that wasn't."""


DEFAULT_RETAINED: Mapping[LuckKind, float] = {
    # Measured, and it really is a coin: over the plays this module classifies,
    # the ball changed hands on 0.477 of NFL fumbles (2008-2025) and 0.540 of
    # NCAAFB ones (2006-2026), each stable to a few points every season.
    # `train.py rates` is the measurement.
    LuckKind.FUMBLE_LOST: 0.5,
    LuckKind.FUMBLE_KEPT: 0.5,
    # Measured too, and the same in both leagues: of the passes a defender is
    # recorded as having reached, 0.20 are intercepted and 0.80 fall. NFL
    # 2009-2025 gives 0.2015 and NCAAFB 2026 -- the first season its feed
    # records them completely -- gives 0.188. Every NCAAFB season back to 2006
    # agrees once corrected for how much its feed wrote down. Complements, for
    # the same reason the fumbles are.
    LuckKind.PASS_DEFENDED_INTERCEPTION: 0.20,
    LuckKind.PASS_DEFENDED_INCOMPLETE: 0.80,
}
"""
How much of each kind's swing is real, as `P(the outcome that happened)`.

The two fumble entries are a **pair, not two numbers**: they are the same coin
from its two sides, so whatever they are they have to sum to 1. 0.5/0.5
satisfies that for free, and it is also within four points of the measurement
in both leagues, which is closer than the difference between the leagues
themselves (NFL 0.477, NCAAFB 0.540). Per-league values would be more honest
and would have to be complements; a single shared pair is the reason this is
one mapping rather than one per release.

The pass pair is `P(interception | a defender reached the ball)` and its
complement, and it is measured the same way -- with the wrinkle that the two
leagues write "a defender reached it" in different alphabets. The NFL says it
in gamebook punctuation, the parenthetical on an *incompletion*, and has said
it completely since 2009: 0.093 to 0.109 of pass attempts every season.
NCAAFB says it in words ("broken up by") and said it unevenly until 2026,
when its feed went gamebook-shaped and its coverage reached the NFL's, 0.107
against 0.109. Where both are complete they agree -- 0.2015 against 0.188 --
and correcting the incomplete seasons for how much their feed wrote down
makes twenty seasons of both leagues agree too, 0.19 to 0.23 against raw
shares spanning 0.18 to 0.69.

What that pair is not is a statement about *tipped* balls specifically. A
pass defended is any ball a defender got to, and no interception in either
league records whether one was deflected on the way -- 88 of 8,980 NFL picks
and 0 of 47,543 NCAAFB ones carry a marker. The neighbouring question turned
out to be the answerable one, and `retained` answers that instead.

Estimates, then, not fits. They are a keyword argument everywhere they are
used so that a caller who disagrees can say so at the call site rather than by
editing this file, and so that `retained=1.0` for every kind reproduces the
unadjusted curve exactly, which is what `luck_test` checks. `retained=1.0` for
one kind is also how a caller drops that kind without dropping it from the
report -- the escape hatch `lucky_wp` points at.
"""


class LuckyPlay(NamedTuple):
    """One play whose result was decided by a bounce."""

    play_id: str
    play_number: int
    kind: LuckKind

    retained: float
    """`P(this outcome)` -- the weight its realized swing keeps."""

    changed_possession: bool
    """
    Whether the ball actually changed hands on it.

    What picks the branch to price: the counterfactual to a fumble that was
    lost is the offense keeping it, and the counterfactual to one that was
    recovered is the other team taking over. Getting it backwards prices the
    wrong branch, which is worse than not adjusting the play, so it is read
    from two witnesses rather than one -- see `_changed_possession`.
    """


class AdjustedCurve(NamedTuple):
    """
    A game's curve both ways, and what moved it.

    Both curves rather than just the adjusted one because every use of this
    wants the pair: a chart draws them together, and the number worth
    reporting is the gap between their two `game_control`s. Computing the
    realized curve is most of the work here, so handing it back costs nothing
    and saves the caller a second `predict` over the whole game.
    """

    points: list[CurvePoint]
    """The adjusted curve. Same points, same labels, reweighted probability."""

    realized: list[CurvePoint]
    """What actually happened -- identical to `win_probability_curve`."""

    lucky_plays: list[LuckyPlay]
    """
    The plays that were actually discounted, in game order.

    A subset of what `find_lucky_plays` found: a lucky play only counts if it
    is a snap on the curve *and* something comes after it, since the last
    snap of a game has no later probability to move.
    """


class LuckySwing(NamedTuple):
    """
    One lucky play, priced both ways.

    `realized` and `counterfactual` are the two branches as home win
    probability *after* the play: what the model made of the snap that
    followed, and what it makes of the snap that would have followed. The gap
    between them is everything the bounce was worth; `home_delta` is the share
    of that gap the bounce handed out rather than the offense earning.
    """

    play_id: str
    play_number: int
    kind: LuckKind

    retained: float
    """`P(the outcome that happened)` -- see `DEFAULT_RETAINED`."""

    realized: float
    """Home win probability at the next snap: the branch that happened."""

    counterfactual: float
    """
    Home win probability of the branch that didn't, as the fit prices it.

    Built from the snap rather than from the play (`_counterfactual`), so it
    carries the snap's score and clock while `realized` carries the next
    snap's. That asymmetry is deliberate and it is not free: see
    `lucky_wp`.
    """

    @property
    def expected(self) -> float:
        """
        Where the play left the game in expectation, before the ball bounced:
        `retained * what happened + (1 - retained) * the other branch`.
        """
        return (
            self.retained * self.realized + (1.0 - self.retained) * self.counterfactual
        )

    @property
    def home_delta(self) -> float:
        """
        `realized - expected`: win probability the bounce handed the home team,
        negative when it went the other way.

        Equal to `(1 - retained) * (realized - counterfactual)`, which is the
        reading worth carrying: a bounce hands out the whole gap between its
        two branches *less* the part that was going to happen anyway. A coin
        flip hands out half the gap, a one-in-four bounce three quarters of it.
        """
        return self.realized - self.expected


class LuckyWP(NamedTuple):
    """
    Win probability the bounces handed each team over a game.

    Not a share of anything, which is where this differs most sharply from its
    neighbour `GameControl`: **`home` and `away` do not sum to 1**. Each is a
    total of win probability in the units the curve is drawn in, so 0.18 for
    the home team reads as "the breaks that went their way were worth eighteen
    points of win probability more than those same plays were worth before the
    ball landed".

    Counted as two non-negative totals rather than one signed number because
    "both teams got a big break" and "neither team got one" are different
    games and `net` alone can't tell them apart. `net` is there for the caller
    who only wants the one number.
    """

    home: float
    """Win probability the bounces handed the home team. >= 0."""

    away: float
    """Win probability the bounces handed the away team. >= 0."""

    swings: list[LuckySwing]
    """Every play in the total, in game order. Empty when nothing bounced."""

    @property
    def net(self) -> float:
        """`home - away`: who came out ahead on the bounces, and by how much."""
        return self.home - self.away


def luck_adjusted_curve(
    model: WinProbabilityModel,
    game: GamePlays,
    *,
    retained: Mapping[LuckKind, float] = DEFAULT_RETAINED,
) -> AdjustedCurve:
    """The game's curve, with each coin flip replaced by its average."""
    return adjusted_curve_from_states(
        model, list(iter_states(game)), find_lucky_plays(game.plays, retained=retained)
    )


def luck_adjusted_game_control(
    model: WinProbabilityModel,
    game: GamePlays,
    *,
    retained: Mapping[LuckKind, float] = DEFAULT_RETAINED,
) -> GameControl | None:
    """
    `game_control` over the adjusted curve: the share of the game each team
    would have controlled with the bounces split evenly.

    None on the same games `game_control` is None on -- no elapsed regulation
    clock to weight by. A game with no lucky plays in it returns exactly the
    unadjusted number, not something near it.
    """
    return game_control(luck_adjusted_curve(model, game, retained=retained).points)


def adjusted_curve_from_states(
    model: WinProbabilityModel,
    states: Sequence[GameState],
    lucky_plays: Iterable[LuckyPlay],
) -> AdjustedCurve:
    """
    The adjusted curve for states and lucky plays a caller already has.

    The one a backend calls, for the same reason `curve_from_states` exists:
    it walked the game once already. Two `predict` calls for the whole game --
    one for the curve, one for every counterfactual at once -- rather than one
    per lucky play.

    Adjusting is a walk over the curve accumulating deltas:

        adjusted[0]   = realized[0]
        adjusted[i+1] = adjusted[i] + (logit(target) - logit(realized[i]))

    where `target` is `realized[i + 1]` for an ordinary snap, and the blend of
    the two branches for a lucky one. Two scales, each for a reason. The
    *blend* is in probability space, because it is an expectation over
    outcomes and that is where an expectation lives. The *accumulation* is in
    log-odds, because deltas there are the model's own additive scale -- the
    coefficients are log-odds -- and because a shifted curve stays inside
    (0, 1) on its own, with no clamp to bend the endgame back into range.

    The shift persists to the final whistle, which is the point: a fumble
    returned for a touchdown in the first quarter is still on the scoreboard
    in the fourth, so a curve that discounts the recovery has to keep
    discounting it.
    """
    realized = curve_from_states(model, states)
    branches = _priced_branches(model, states, realized, lucky_plays)
    if not branches:
        # No bounce with a snap after it -- including the one-snap game, which
        # has no delta to adjust and no elapsed time either. The curve is the
        # curve.
        return AdjustedCurve(points=realized, realized=realized, lucky_plays=[])

    counterfactual = {branch.index: branch.counterfactual for branch in branches}
    lucky_at = {branch.index: branch.lucky for branch in branches}

    logit = _logit(realized[0].home_win_probability)
    points = [realized[0]]
    for index, point in enumerate(realized[:-1]):
        after = realized[index + 1].home_win_probability
        lucky = lucky_at.get(index)
        if lucky is not None:
            after = (
                lucky.retained * after + (1.0 - lucky.retained) * counterfactual[index]
            )
        logit += _logit(after) - _logit(point.home_win_probability)
        points.append(
            realized[index + 1]._replace(home_win_probability=_sigmoid(logit))
        )
    return AdjustedCurve(
        points=points,
        realized=realized,
        lucky_plays=[branch.lucky for branch in branches],
    )


def lucky_wp(
    model: WinProbabilityModel,
    game: GamePlays,
    *,
    retained: Mapping[LuckKind, float] = DEFAULT_RETAINED,
) -> LuckyWP:
    """
    How much win probability the bounces handed each team.

    The other number in this module, and the one that answers "who got the
    breaks" rather than "who would have controlled this game without them".
    For every lucky play it takes the gap between the branch that happened and
    the branch that didn't, and keeps the share of that gap the bounce was
    responsible for -- `(1 - retained)` of it. Those add up, per team.

    Where `luck_adjusted_game_control` is a *rewrite* of the game, this is an
    *accounting* of it: nothing is carried forward, nothing is re-averaged
    over the clock, and the game's realized curve is left exactly as it is.
    Each play contributes once, and the total is in win probability, so it
    reads on the same scale as the curve a chart draws. The two are meant to
    be read together -- one says how big the bounces were, the other says what
    the game looks like without them.

    Two things to hold onto when reading the number:

    - **The tipped balls are counted one-sidedly.** A tipped pass that *was*
      intercepted is in the play text; one that fell incomplete mostly isn't,
      so the defense is charged for the breaks it got and the offense is never
      credited for the ones it got. That asymmetry is survivable in
      `luck_adjusted_game_control`, where a missing play would shift a curve
      slightly, and it is not survivable here: a dropped tipped ball is worth
      `0.25` of a full turnover swing to the offense, and none of that is in
      this total. Fumbles have no such gap -- both branches are in the text.
      A caller who wants the clean half of the number passes
      `retained={**DEFAULT_RETAINED, LuckKind.PASS_DEFENDED_INTERCEPTION: 1.0,
      LuckKind.PASS_DEFENDED_INCOMPLETE: 1.0}`,
      which zeroes every tipped interception's contribution.
    - **The two branches are read at different moments.** `realized` is the
      next snap, after the clock ran and any points went up; the
      counterfactual is built from the snap itself (see `_counterfactual`).
      A play's worth of clock is small against a change of possession, but it
      is a floor under how precise this can be.
    """
    return lucky_wp_from_states(
        model, list(iter_states(game)), find_lucky_plays(game.plays, retained=retained)
    )


def lucky_wp_from_states(
    model: WinProbabilityModel,
    states: Sequence[GameState],
    lucky_plays: Iterable[LuckyPlay],
) -> LuckyWP:
    """
    `lucky_wp` for states and lucky plays a caller already has.

    The backend path, for the same reason `adjusted_curve_from_states` is:
    two `predict` calls for the whole game rather than one per lucky play.

    Counts the same plays `adjusted_curve_from_states` adjusts -- a bounce on
    the last snap of the game has no next snap to price it against, so it is
    left out of both.
    """
    realized = curve_from_states(model, states)
    swings = [
        LuckySwing(
            play_id=branch.lucky.play_id,
            play_number=branch.lucky.play_number,
            kind=branch.lucky.kind,
            retained=branch.lucky.retained,
            realized=realized[branch.index + 1].home_win_probability,
            counterfactual=branch.counterfactual,
        )
        for branch in _priced_branches(model, states, realized, lucky_plays)
    ]
    deltas = [swing.home_delta for swing in swings]
    return LuckyWP(
        home=float(sum(delta for delta in deltas if delta > 0.0)),
        away=float(-sum(delta for delta in deltas if delta < 0.0)),
        swings=swings,
    )


class _Branch(NamedTuple):
    """A lucky play that lands on the curve, and what the other branch is worth."""

    index: int
    """
    Where the play sits in both `states` and the realized curve.

    One index for the two because `curve_from_states` is one point per state,
    in order -- the curve carries no filtering of its own.
    """

    lucky: LuckyPlay
    counterfactual: float
    """The model's home win probability for the branch that didn't happen."""


def _priced_branches(
    model: WinProbabilityModel,
    states: Sequence[GameState],
    realized: Sequence[CurvePoint],
    lucky_plays: Iterable[LuckyPlay],
) -> list[_Branch]:
    """
    The lucky plays that can actually move a number, with the branch that
    didn't happen priced -- in one `predict` for the whole game rather than
    one per play.

    Both metrics in this module go through here, which is what keeps them
    talking about the same football: `luck_adjusted_curve` and `lucky_wp` are
    free to disagree about what a bounce was worth, but not about which plays
    were bounces or what the other branch looked like.

    A lucky play qualifies only if it is a snap on the curve *and* something
    comes after it. `[:-1]` is the second half of that: the last snap of a
    game has no later probability to move and no next snap to read "what
    happened" off. The first half is the dictionary lookup -- `find_lucky_plays`
    reads every play, including the kickoffs and the clock stoppages
    `iter_states` drops, and those never appear in `realized` to be found.
    """
    if len(realized) < 2:
        return []
    by_play_id = {lucky.play_id: lucky for lucky in lucky_plays}
    applied = [
        (index, by_play_id[point.play_id])
        for index, point in enumerate(realized[:-1])
        if point.play_id in by_play_id
    ]
    if not applied:
        return []
    prices = model.predict(
        [
            _counterfactual(states[index], lucky.changed_possession)
            for index, lucky in applied
        ]
    )
    return [
        _Branch(index=index, lucky=lucky, counterfactual=float(price))
        for (index, lucky), price in zip(applied, prices)
    ]


# The two ways a feed says a defender got to a pass that stayed incomplete.
#
# In words, which is NCAAFB's way and the NFL's only before it stopped:
DEFENDED_MARKERS: tuple[str, ...] = (
    "broken up",
    "broke up",
    "break up",
    "defensed",
    "pass defended",
    "deflect",
    "tipped",
    "tip by",
    "batted",
    "knocked down",
    "knocked away",
    "swatted",
)

# ...and in punctuation, which is the NFL's. Its text is gamebook text, where
# a trailing parenthetical is the defender credited with the play:
#
#     J.Garcia pass incomplete short middle to E.Graham (B.James) [S.Bowen]
#                                                        ^^^^^^^^  ^^^^^^^^
#                                                        defensed   QB hit
#
# Only on incompletions. On a completion that same parenthetical is the
# tackler, and on an interception it is the tackler on the return -- an
# incomplete pass is the one case with nobody to tackle, which is what makes
# the credit there unambiguous. The check that settles it rather than assuming
# it: completions ending in a touchdown have no tackler either, and carry a
# parenthetical 20% of the time against 82% for every other completion.
#
# The bracket is a QB hit and is deliberately not read: pressure is not a hand
# on the ball. NCAAFB's "qb hurried by" is the same thing in words, and is
# likewise absent from DEFENDED_MARKERS.
_PAREN = re.compile(r"\(([^)]*)\)")
_LEADING_CLOCK = re.compile(r"^\(\d+:\d+\)\s*")
_DEFENDER_NAME = re.compile(r"^#?\d*\s?[A-Z][A-Za-z'\-]*\.\s?[A-Za-z'\-. ]+$")

DEFENDED_RATE_FLOOR = 0.15
"""
The share of a game's incompletions that must name a defender for the pass
pair to be adjusted in it. See `records_defended_passes`.
"""

MIN_INCOMPLETIONS = 10
"""Below this a game has too few incompletions to say anything about its feed."""


def records_defended_passes(
    plays: Iterable[Play], *, floor: float = DEFENDED_RATE_FLOOR
) -> bool:
    """
    Whether this game's play-by-play records defended passes at all.

    The gate on `PASS_DEFENDED_*`, and the reason those kinds can be trusted
    where they fire. Whether a feed writes down a broken-up pass is a property
    of whoever entered the play-by-play, not of the football: in NCAAFB it
    follows the *home venue*, and between 2019 and 2023 the number of venues
    doing it fell from 168 to 41 -- of 224, with 127 dropping it and **none**
    taking it up. The NFL has done it every season since 2009 and not at all
    before.

    A game where it isn't recorded still has all its interceptions, and
    adjusting those alone is the one thing this module must not do: it would
    charge every defense for its picks and credit none for the balls it got a
    hand on. So the pair is adjusted where the game can support both sides and
    left alone where it can't, and `luck_adjusted_game_control` is a smaller
    adjustment on a game that says less. That is the honest shape for it --
    the number is what the game can support, not a constant pretending the
    evidence is even.

    The test is a rate rather than presence because a venue that records
    *some* of them would leak in proportion to what it missed. What says the
    rate is enough: among games that pass this, defended passes outnumber
    interceptions 3.5 to 4.9 to one in every league and season measured --
    including NCAAFB 2022-24, where only a fifth of games pass. The balance
    point is `(1 - retained) / retained` = 4. So the seasons that collapsed
    lost whole games rather than fidelity within them, and a game that passes
    in 2023 is worth what one in 2019 is.
    """
    incompletions = defended = 0
    for play in plays:
        text = (play.text or "").lower()
        if not text or "no play" in text or "intercept" in text:
            continue
        if "incomplete" not in text:
            continue
        incompletions += 1
        defended += is_defended_pass(play.text)
    if incompletions < MIN_INCOMPLETIONS:
        return False
    return defended / incompletions >= floor


def is_defended_pass(text: str | None) -> bool:
    """
    Whether an incomplete pass's text names the defender who got to the ball.

    Public because measuring a feed and classifying a play have to agree about
    what "defended" means -- `train.py rates` profiles the same predicate this
    classifies on, so a season's reported coverage is the coverage the gate
    will actually see.
    """
    lowered = (text or "").lower()
    if any(marker in lowered for marker in DEFENDED_MARKERS):
        return True
    for inner in _PAREN.findall(_LEADING_CLOCK.sub("", text or "")):
        if any(_DEFENDER_NAME.match(part.strip()) for part in inner.split(",")):
            return True
    return False


def find_lucky_plays(
    plays: Iterable[Play],
    *,
    retained: Mapping[LuckKind, float] = DEFAULT_RETAINED,
) -> list[LuckyPlay]:
    """
    Every play in `plays` whose result was decided by a bounce, in order.

    Classification is substring matching over `Play.text`, which is what
    there is: ESPN ships a sentence, and the manner of a play is in the
    sentence or nowhere. That makes this a filter with a false-negative rate
    rather than a parser -- a fumble phrased in a way these words miss is
    simply not adjusted, which leaves the realized number, which is the safe
    direction to be wrong in.

    A play with no text is skipped, and so is one whose text says "no play":
    a fumble wiped out by a penalty never happened, and its win probability
    swing is a penalty's, not a bounce's.

    Whether the ball changed hands is `_changed_possession`, which needs the
    play after this one -- so `plays` is materialised here rather than
    streamed. A fumble no witness can call is dropped instead of guessed at.
    """
    ordered = list(plays)
    defended = records_defended_passes(ordered)
    lucky = []
    for index, next_offense in enumerate(_after(ordered)):
        play = ordered[index]
        classified = _classify(play, next_offense, defended)
        if classified is None:
            continue
        kind, changed_possession = classified
        lucky.append(
            LuckyPlay(
                play_id=play.play_id,
                play_number=play.play_number,
                kind=kind,
                retained=retained.get(kind, DEFAULT_RETAINED[kind]),
                changed_possession=changed_possession,
            )
        )
    return lucky


def _after(plays: Sequence[Play]) -> list[str | None]:
    """
    For each play, the offense of the next play that has one.

    One backward pass rather than a scan per play, and it skips the plays
    with no offense at all (a timeout, the end of a quarter) so that the
    witness below sees the next real snap rather than a gap.
    """
    following: list[str | None] = [None] * len(plays)
    seen: str | None = None
    for index in range(len(plays) - 1, -1, -1):
        following[index] = seen
        if plays[index].offense_team_id is not None:
            seen = plays[index].offense_team_id
    return following


def _changed_possession(play: Play, next_offense: str | None) -> bool | None:
    """
    Whether the ball changed hands, from the two witnesses there are.

    `is_turnover` is trusted when it says yes and checked when it says no,
    which is not a hedge -- it is the shape of the column's error. Over
    NFL 2008-2025 and NCAAFB 2006-2026 it disagrees with the possession the
    next snap actually shows on 17% and 34% of fumbles respectively, and
    almost every one of those is a false *negative*: a sack-fumble the
    defense recovered, written `FUMBLES ... RECOVERED by <the other team>`
    in the text and `is_turnover = false` in the column. In NCAAFB 2006-2013
    the column is uniformly false, so on its own it calls every fumble in
    eight seasons a recovery.

    The second witness is the next snap's offense, which is what possession
    means. It has one failure of its own -- a fourth-down fumble the offense
    recovered short of the sticks hands the ball over on downs, so the next
    snap is the other team's and this reads it as lost. Rare enough against
    a column that is wrong on a fifth of the population.

    None when neither witness can say: no offense on the play, nothing after
    it, and a null column.
    """
    if play.is_turnover:
        return True
    if play.offense_team_id is not None and next_offense is not None:
        return next_offense != play.offense_team_id
    return play.is_turnover


def _classify(
    play: Play, next_offense: str | None, defended_recorded: bool
) -> tuple[LuckKind, bool] | None:
    """The play's `LuckKind` and whether it turned the ball over, or None."""
    text = (play.text or "").lower()
    if not text or "no play" in text:
        return None
    if "intercept" in text:
        # Returning here rather than falling through matters: an interception
        # returned and fumbled has both words in it, and the interception is
        # the play. Unadjustable without the other side of its coin -- see
        # `records_defended_passes`.
        if defended_recorded:
            return LuckKind.PASS_DEFENDED_INTERCEPTION, True
        return None
    if "incomplete" in text:
        if defended_recorded and is_defended_pass(play.text):
            return LuckKind.PASS_DEFENDED_INCOMPLETE, False
        return None
    if "fumble" in text:
        changed = _changed_possession(play, next_offense)
        if changed is None:
            return None
        return (LuckKind.FUMBLE_LOST if changed else LuckKind.FUMBLE_KEPT), changed
    return None


def _counterfactual(state: GameState, changed_possession: bool) -> GameState:
    """
    The state the branch that didn't happen would have led to.

    Built from the snap itself rather than from the next one, which is what
    keeps it a counterfactual: the realized play's yards and any points it put
    on the board belong to the branch that happened, and would be wrong on the
    other one -- a fumble returned for a touchdown didn't score in the branch
    where it was recovered by the offense.

    So the ball is at the spot of the snap, and the only thing that moves is
    who has it:

    - **It changed hands, so the other branch is keeping it.** Same team, same
      spot, next down. On fourth down that's a first: a fourth-down snap the
      offense held onto is one they converted.
    - **It didn't, so the other branch is losing it.** The other team, first
      and ten, at the mirror of the same yardline -- `yardline` is measured
      from the offense's own goal line, so possession flipping flips it.

    An approximation, and the shape of the error is worth knowing: it prices
    a turnover at the line of scrimmage, ignoring the yards the play gained
    before the ball came loose and the yards a return added after. Against the
    thing being measured -- who has the ball -- that is small. A 25-yard
    change in field position is worth a few points of win probability; the
    possession itself is worth several times that.
    """
    if changed_possession:
        down = state.down + 1
        if down > 4:
            return state._replace(down=1, distance=10)
        return state._replace(down=down)
    return state._replace(
        offense_is_home=not state.offense_is_home,
        yardline=max(1, min(99, 100 - state.yardline)),
        down=1,
        distance=10,
    )


# Far enough from 0 and 1 that a log is finite, close enough that clipping
# never bites a probability a fit would actually produce. The endgame is where
# it would matter -- `predict` returns genuine 0.9999s in the last minute of a
# blowout -- and 1e-9 leaves those alone.
_EPSILON = 1e-9


def _logit(probability: float) -> float:
    clipped = min(max(probability, _EPSILON), 1.0 - _EPSILON)
    return float(np.log(clipped / (1.0 - clipped)))


def _sigmoid(logit: float) -> float:
    """
    The inverse, written the overflow-free way for the same reason
    `model._sigmoid` is: a three-score lead in the last minute is a genuinely
    enormous log-odds, and `1 / (1 + exp(-x))` returns garbage for it.
    """
    exponentiated = float(np.exp(-abs(logit)))
    if logit >= 0:
        return 1.0 / (1.0 + exponentiated)
    return exponentiated / (1.0 + exponentiated)
