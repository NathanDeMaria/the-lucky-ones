"""
Expected points: what a snap is worth on the scoreboard.

Win probability asks who is going to win. Expected points asks a smaller and
much older question -- *how many points is this situation worth to the team
holding the ball?* -- and it is the question you need if you want to say
anything about a single play. A win probability model can tell you a 12-yard
gain on third and 8 helped; it can't tell you whether it helped as much as a
12-yard gain on third and 8 usually does, because the answer depends on the
scoreboard and the clock, which have nothing to do with the play.

The definition is the standard one, and it is a definition about *drives*
rather than about plays:

    EP(state) = the expected value of the next score in this half,
                signed from the point of view of the team with the ball.

So a first and goal at the 2 is worth nearly a touchdown; a third and 15 on
your own 4 is worth slightly *less* than zero, because the next points in the
half are more often the other team's. Nobody has to decide what a play is
worth -- the football decides, and this module counts.

Three pieces, in the order they run:

- `score_events` reads the scores out of the score columns, with the feed's
  own scoring flag as a second witness. How much the score went up by says
  which of the three things happened; why it takes two witnesses to say that
  it happened at all is the interesting part, and it is in that function.
- `next_scores` turns that into one `ScoreKind` per snap: the label.
- `MultinomialExpectedPoints` fits `P(kind | situation)` over the seven kinds
  and takes the expectation. Seven probabilities and a dot product with
  `SCORE_VALUES`, so the scoring path is numpy and nothing else, exactly like
  `LogisticWinProbability`.

`lucky_ones.epa` is what this is for: the difference between what a snap was
worth and what the next one is worth is the play, and that difference is the
one football number that is about a play rather than about a game.
"""

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Protocol, Sequence

import numpy as np

from .features import EP_FEATURE_NAMES, ep_feature_matrix
from .game import GamePlays
from .plays import Play
from .state import GameState, half_of, is_a_down, is_scrimmage_play, is_try, iter_states

TOUCHDOWN_POINTS = 7.0
"""
A touchdown and the try after it, as one number.

Flat 7 rather than the six on the board plus whatever the kick did. The try
is a near-certainty that says nothing about the snap being priced -- nobody
runs a better third and 6 because their kicker is good -- so folding its
variance into the label makes the class noisier without moving its mean far.
Both leagues convert well over 90% of extra points, and the two-point tries
and the misses roughly cancel what is left.
"""

FIELD_GOAL_POINTS = 3.0
SAFETY_POINTS = 2.0

# What a play's score change is allowed to be, and what it means. A change of
# 1 is an extra point, which belongs to the touchdown two plays ago and is
# folded into it above rather than counted as a scoring event of its own; a
# change of 8 is a touchdown whose two-point try the feed put on the same row.
# Anything else -- 4, 5, 9, a correction going the other way -- is the feed
# being wrong about a score, and is skipped rather than guessed at.
_POINTS_BY_CHANGE: Mapping[int, float] = {
    2: SAFETY_POINTS,
    3: FIELD_GOAL_POINTS,
    6: TOUCHDOWN_POINTS,
    7: TOUCHDOWN_POINTS,
    8: TOUCHDOWN_POINTS,
}

# How many rows after a flagged play its jump may arrive and still be its.
# Measured on NCAAFB 2010 and 2013: of the displaced jumps that follow a
# flagged row, 27 of 31 land on the very next row and 3 more within three;
# past that the pairings are as rare as coincidence would make them.
_PENDING_REACH = 3

# The other direction is looser: a jump that arrives *early* can sit on the
# timeout or the incompletion half a dozen rows before the play the feed
# flags, so a held jump waits longer for its flag than a flag waits for its
# jump.
_EARLY_REACH = 6


FIRST_LEGIBLE_SEASON = 2014
"""
The first season whose scoring can be placed on the play that produced it, in
both leagues.

Not a rule about football -- a fact about the feed, and the one that decides
what `train.py train_ep` should be pointed at. Before it the feed's two
witnesses to a score -- the columns and the flag -- disagree on about one
score in six: the jump lands on the try or the kickoff, or a row early, or
the touchdown play is missing altogether and only its try is there.
`score_events` pairs most of that back up, and what it can't place it
leaves off: read through it, a game's witnessed scoring is within two points
of the final for 70-83% of NCAAFB games in 2010-2013, against 91-97% from
2014 on. That is enough to *score* those seasons -- an EPA index over them
reads on the same scale as the later ones -- and not enough to train on
them, where a label attached to the wrong drive is a label, so this is
still where the expected points fit starts.

It is also where NCAAFB's `is_turnover` stops being uniformly false (see
`luck._changed_possession`), which is a different column with the same
boundary -- the play-by-play got better in both leagues at once.
"""


class ScoreKind(StrEnum):
    """
    The next thing that happens on the scoreboard, from the point of view of
    the team with the ball right now.

    Seven kinds: three ways to score, three ways to be scored on, and the
    half running out with neither. "Offense" and "defense" are the sides *at
    the snap being priced*, not at the score -- `OFFENSE_SAFETY` is the team
    currently holding the ball tackling somebody in the end zone later in the
    half, which is a perfectly ordinary way for a possession to end up worth
    two points.
    """

    OFFENSE_TOUCHDOWN = "offense_touchdown"
    OFFENSE_FIELD_GOAL = "offense_field_goal"
    OFFENSE_SAFETY = "offense_safety"
    NO_SCORE = "no_score"
    DEFENSE_SAFETY = "defense_safety"
    DEFENSE_FIELD_GOAL = "defense_field_goal"
    DEFENSE_TOUCHDOWN = "defense_touchdown"


SCORE_VALUES: Mapping[ScoreKind, float] = {
    ScoreKind.OFFENSE_TOUCHDOWN: TOUCHDOWN_POINTS,
    ScoreKind.OFFENSE_FIELD_GOAL: FIELD_GOAL_POINTS,
    ScoreKind.OFFENSE_SAFETY: SAFETY_POINTS,
    ScoreKind.NO_SCORE: 0.0,
    ScoreKind.DEFENSE_SAFETY: -SAFETY_POINTS,
    ScoreKind.DEFENSE_FIELD_GOAL: -FIELD_GOAL_POINTS,
    ScoreKind.DEFENSE_TOUCHDOWN: -TOUCHDOWN_POINTS,
}
"""
What each kind is worth to the team with the ball. Signed, and symmetric:
the same points cost the same as they pay.
"""


class ScoreEvent(NamedTuple):
    """One score, as the score columns record it."""

    play_id: str
    play_index: int
    """Where it sits in the play list it was read from."""

    half: int
    """The half it happened in. A score doesn't count for the half before it."""

    home_scored: bool
    """Which side of the scoreboard moved."""

    points: float
    """One of `TOUCHDOWN_POINTS`, `FIELD_GOAL_POINTS`, `SAFETY_POINTS`."""


class ExpectedPointsModel(Protocol):
    """
    Anything that can put a number of points on a situation.

    Vector in, vector out, for the same reason `WinProbabilityModel` is:
    scoring a season is millions of snaps and the baseline's `predict` is one
    matrix multiply for all of them.
    """

    def predict(self, states: Sequence[GameState]) -> np.ndarray:
        """
        Expected points for the team with the ball, one per state.

        Signed: negative where the next score in the half is more often the
        other team's, which is most of third and long from inside your own 10.
        """


def score_events(plays: Sequence[Play]) -> list[ScoreEvent]:
    """
    Every score in `plays`, from the two witnesses to one: the score columns
    went up, *and* the feed flagged the play as a scoring play.

    endgame's scores are cumulative after the play, so a score is a play
    where one of them went up, and the size of the jump says which of the
    three scoring events it was -- more robust than reading the text, where a
    touchdown is spelled a dozen ways and scored exactly one.

    Two details about the extra point that the jump has to get right:

    - The try is a separate row, and its single point is *not* an event of
      its own -- it belongs to the touchdown before it, already priced at
      `TOUCHDOWN_POINTS`. So a change of 1 is skipped.
    - A missed try leaves the touchdown at 6 and a two-point conversion takes
      it to 8. Both are the same event, so both map to the same value.

    **And why `scoring_play` has to agree.** Before 2014 the score columns
    jitter in both leagues: the points appear on a play, come off again a
    snap later, and reappear where they belong, so a bare reading of the
    column finds a touchdown and then an untouchdown. Counting the points it
    finds against the final score is the check, and the columns alone are
    wrong by 21% in the NFL of 2013 and 28% in the NCAAFB of 2010 -- 67 points
    of scoring in a 52-point game. Requiring the flag as well takes
    that to within a couple of points of the real total in every season of
    both leagues, and to within half a point from 2014 on.

    It is the same shape of fix as `luck._changed_possession`, and it errs the
    same way on purpose. What is left is a *false negative* -- a score neither
    witness can place -- and a score this can't find leaves its snaps
    labelled with whatever scores next. That is a quieter error than
    inventing a touchdown, which is what the alternative does. `train.py
    train_ep` prints the class shares for the same reason: a feed that
    stopped flagging would show up there as a season that never scored.

    **When the two witnesses are a row apart.** Before 2014, about one score
    in seven has the flag on the touchdown and the jump on the row after it
    -- the try, or the kickoff -- and read strictly that is a flagged play
    that didn't score followed by a score nobody flagged. Both halves of it
    are visible, so this pairs them: a flagged row whose columns didn't move
    is held as *pending*, and a jump within the next few rows of the same
    period is credited to it. The reach is short (`_PENDING_REACH`) and a
    correction in between forfeits it, because the other thing a flagged
    row without a jump can be is a touchdown called back, and a jitter
    landing on that would invent the score the flag was wrong about.

    **The jitter itself is handled by remembering it.** The columns take a
    score off and put it back, and the row they put it back on is not
    always the row they took it from: a touchdown witnessed on its own row
    comes off on the next snap and returns on the try, or on the snap after
    that. A withdrawal is held, and the same points returning to the same
    side within reach of it -- with that score already an event -- is the
    jitter closing, not a second touchdown. Without this, every rule below
    that places a score on a row would place the returning copy somewhere.

    The displacement runs the other way too, less often: the columns move
    on a row *before* the touchdown -- the snap before it, a timeout, the
    kickoff -- and the flag arrives on the touchdown. An unflagged jump is
    held the same way, and a flagged row without a jump within reach of it
    (`_EARLY_REACH`, the longer of the two) is the play it belongs to.
    Unclaimed, it is the jitter and is dropped as before.

    **When the touchdown isn't in the feed at all.** The larger case before
    2014, about one touchdown in ten: the row before the try is an ordinary
    snap from wherever the drive had got to, the touchdown play is simply
    absent, and the try carries the flag and the whole 7. A try is not a
    snap (`state.NOT_A_SNAP`), so a score credited to it is a score no
    state carries, and `lucky_ones.epa` would price the drive as ending at
    the other team's first snap -- minus seven and change against the
    offense that just scored. So a score on a try is credited to the last
    snap within reach: the drive still adds up to the points it scored
    minus what its first snap was worth, and only the play count is one
    short. That holds when the missing play was a pick-six, too -- the
    points are signed to the side of the scoreboard that moved, and
    `lucky_ones.epa` reads them against whoever had the ball at the snap
    they land on, which is the offense that threw it. Only the try gets
    this. A kickoff returned for a touchdown is scored on the kickoff row
    in every season, and the snap before it belongs to the team that just
    got scored on.
    """
    events: list[ScoreEvent] = []
    home, away = 0, 0
    period = 1
    # A flagged row whose score columns didn't move, as (index, play,
    # period), waiting for the jump the feed put on a later row.
    pending: tuple[int, Play, int] | None = None
    # The mirror image: a jump on an unflagged snap, as (index, (home_scored,
    # points), period), waiting for the flagged row the feed put it before.
    orphan: tuple[int, tuple[bool, float], int] | None = None
    # A score the columns took back, in the same shape. The jitter puts it
    # back within a row or two, usually somewhere else -- on the try, on the
    # next snap -- and that is the same score returning, not a new one.
    withdrawn: tuple[int, tuple[bool, float], int] | None = None
    # The most recent scrimmage snap, the same way -- where a score the feed
    # put on a try is credited when no flagged row is waiting for it.
    last_snap: tuple[int, Play, int] | None = None
    for index, play in enumerate(plays):
        if play.period is not None:
            period = play.period
        if is_scrimmage_play(play.play_type) and is_a_down(play.down):
            last_snap = (index, play, period)
        if play.home_score is None or play.away_score is None:
            continue
        change_home, change_away = play.home_score - home, play.away_score - away
        # Reset the running total either way, including on a change that goes
        # backwards: a correction is the feed telling us the score, and the
        # next jump has to be measured from the corrected one.
        home, away = play.home_score, play.away_score
        if change_home < 0 or change_away < 0:
            # A correction between a flag and its jump means the jump is
            # measured from a number the feed withdrew; nothing after it can
            # be paired with confidence.
            pending = orphan = None
            taken_back = _POINTS_BY_CHANGE.get(-min(change_home, change_away))
            if taken_back is not None:
                withdrawn = (index, (change_home < 0, taken_back), period)
        # Both sides moving up on one play is a feed that skipped a row, not
        # a play; there's no honest way to say which score came first. One
        # side corrected down while the other went up is one score, and the
        # side that went up is the one that scored.
        if change_home > 0 and change_away > 0:
            continue
        if change_home <= 0 and change_away <= 0:
            if play.scoring_play and change_home == 0 and change_away == 0:
                # Flagged, and the columns didn't move: either the jump
                # arrived early and is held, or it is still to come.
                if (early := _owner(orphan, index, period, _EARLY_REACH)) is not None:
                    home_scored, points = early[1]
                    orphan = None
                    events.append(
                        ScoreEvent(
                            play_id=play.play_id,
                            play_index=index,
                            half=half_of(period),
                            home_scored=home_scored,
                            points=points,
                        )
                    )
                else:
                    pending = (index, play, period)
            continue
        home_scored = change_home > 0
        change = change_home if home_scored else change_away
        points = _POINTS_BY_CHANGE.get(change)
        if points is None:
            # A change of 1 is the try, and it neither scores nor disturbs a
            # touchdown still waiting for its jump; anything else unlisted
            # is the feed being wrong about a score.
            continue
        if (
            (returning := _owner(withdrawn, index, period)) is not None
            and returning[1] == (home_scored, points)
            and any(
                (event.home_scored, event.points) == (home_scored, points)
                and index - event.play_index <= 2 * _PENDING_REACH
                for event in events[-2:]
            )
        ):
            # The withdrawn score, back on the board: already an event.
            withdrawn = None
            continue
        scored_at = index, play
        if not play.scoring_play or not is_scrimmage_play(play.play_type):
            # The jump is on a row that can't have scored it: unflagged, or
            # not a snap. It belongs to the flagged snap just before it if
            # one is in reach; failing that, a flagged try's points belong
            # to the possession before it. Anything else is an unflagged
            # jump -- the jitter the flag exists to refuse -- or a score on
            # a kickoff, which is where the feed says it happened.
            if (owner := _owner(pending, index, period)) is not None:
                scored_at = owner
            elif (
                play.scoring_play
                and is_try(play.play_type)
                and (owner := _owner(last_snap, index, period)) is not None
            ):
                scored_at = owner
            elif not play.scoring_play:
                orphan = (index, (home_scored, points), period)
                continue
        pending = orphan = None
        if events and events[-1].play_index == scored_at[0]:
            # The jitter's second copy: the columns withdrew this score and
            # put it back on a later row, and both readings landed here.
            continue
        events.append(
            ScoreEvent(
                play_id=scored_at[1].play_id,
                play_index=scored_at[0],
                half=half_of(period),
                home_scored=home_scored,
                points=points,
            )
        )
    return events


def _owner[T](
    held: tuple[int, T, int] | None,
    index: int,
    period: int,
    reach: int = _PENDING_REACH,
) -> tuple[int, T] | None:
    """The held row, if it is close enough behind `index` to be paired with it."""
    if held is None or held[2] != period or index - held[0] > reach:
        return None
    return held[0], held[1]


def scoring_plays(plays: Sequence[Play]) -> dict[str, float]:
    """
    `play_id` -> the points that play put on the board, positive for the home
    team and negative for the away team.

    What `lucky_ones.epa` needs to know where a play ended: a snap that
    scored has no next snap to price it against, it has the points. Signed to
    the home team rather than to the offense because the caller has the state
    and the state knows who had the ball -- a fumble returned for a touchdown
    is the defense's seven, and the two signs disagree about it.
    """
    return {
        event.play_id: event.points if event.home_scored else -event.points
        for event in score_events(plays)
    }


def next_scores(game: GamePlays) -> list[ScoreKind]:
    """
    The label for every snap of `game`: what the next score in the half was,
    from the point of view of the team with the ball.

    One per state of `iter_states(game)`, in the same order, so a caller can
    zip the two together. `NO_SCORE` when the half ends first, which is not a
    missing label -- it is the outcome that makes expected points finite, and
    roughly one snap in six carries it.

    Two rules decide the rest:

    - **A score counts for the half it happened in.** The clock running out
      is an absorbing state: a drive that stalls at the two-minute warning
      was worth nothing, however many points the third quarter went on to
      produce, and a model told otherwise would price the end of a half like
      the middle of one.
    - **A play's own score is its own next score.** The state before a
      touchdown is labelled with that touchdown. That is the whole point --
      it is what makes first and goal at the 2 worth what it is worth.
    """
    plays = list(game.plays)
    events = score_events(plays)
    index_of = {play.play_id: index for index, play in enumerate(plays)}

    labels: list[ScoreKind] = []
    # Both walks run forwards, so the search for the next event resumes where
    # the last one landed rather than scanning the game again per snap.
    cursor = 0
    for state in iter_states(game):
        index = index_of[state.play_id]
        while cursor < len(events) and events[cursor].play_index < index:
            cursor += 1
        event = events[cursor] if cursor < len(events) else None
        if event is None or event.half != state.half:
            labels.append(ScoreKind.NO_SCORE)
            continue
        labels.append(_kind(event.points, event.home_scored == state.offense_is_home))
    return labels


def _kind(points: float, by_offense: bool) -> ScoreKind:
    """A score's size and side as one of the six that aren't `NO_SCORE`."""
    if points == TOUCHDOWN_POINTS:
        return (
            ScoreKind.OFFENSE_TOUCHDOWN if by_offense else ScoreKind.DEFENSE_TOUCHDOWN
        )
    if points == FIELD_GOAL_POINTS:
        return (
            ScoreKind.OFFENSE_FIELD_GOAL if by_offense else ScoreKind.DEFENSE_FIELD_GOAL
        )
    return ScoreKind.OFFENSE_SAFETY if by_offense else ScoreKind.DEFENSE_SAFETY


class MultinomialExpectedPoints:
    """
    Multinomial logistic regression over `EP_FEATURE_NAMES`, read out as
    points.

    `predict` is `softmax(X @ W.T + b) @ values` -- the seven probabilities
    and their dot product with `SCORE_VALUES`. Which is the shape worth
    having: the model is a *classifier* over what happens next, so it can
    only ever return a number between -7 and +7, and it can't be pushed
    outside that by a strange situation the way a regression on points could.
    The intermediate is also the interesting object. `predict_proba` hands
    back the seven, and "this snap is 31% a touchdown and 18% a punt into a
    scoring drive the other way" is a great deal more legible than the 2.1
    they average out to.

    Holds `(kinds, coefficients, intercepts)` and nothing else, so an instance
    is a few kilobytes, `save` writes readable JSON, and scoring needs numpy
    alone. `fit` imports scikit-learn inside the method for the same reason
    `LogisticWinProbability.fit` does -- see the `fit` extra in pyproject.
    """

    def __init__(
        self,
        coefficients: np.ndarray,
        intercepts: np.ndarray,
        kinds: Sequence[ScoreKind],
        feature_names: Sequence[str] = EP_FEATURE_NAMES,
    ) -> None:
        coefficients = np.asarray(coefficients, dtype=float)
        intercepts = np.asarray(intercepts, dtype=float).reshape(-1)
        if coefficients.ndim != 2:
            raise ValueError(
                f"Coefficients are one row per class: got shape {coefficients.shape}"
            )
        if coefficients.shape != (len(kinds), len(feature_names)):
            raise ValueError(
                f"Got coefficients of shape {coefficients.shape} for "
                f"{len(kinds)} classes and {len(feature_names)} features"
            )
        if len(intercepts) != len(kinds):
            raise ValueError(
                f"Got {len(intercepts)} intercepts for {len(kinds)} classes"
            )
        self.coefficients = coefficients
        self.intercepts = intercepts
        self.kinds = tuple(ScoreKind(kind) for kind in kinds)
        self.feature_names = tuple(feature_names)

    @property
    def values(self) -> np.ndarray:
        """`SCORE_VALUES` in this fit's class order, for the dot product."""
        return np.array([SCORE_VALUES[kind] for kind in self.kinds])

    @classmethod
    def fit(
        cls,
        states: Sequence[GameState],
        next_score: Sequence[ScoreKind],
        *,
        regularization: float = 1.0,
    ) -> "MultinomialExpectedPoints":
        """
        Fit on snaps labelled with what scored next -- what
        `lucky_ones.training.build_expected_points_set` produces.

        `regularization` is sklearn's `C`, so larger means less of it.

        Raises `ImportError` naming the extra if scikit-learn isn't installed,
        which is the expected state of a serving environment.
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as error:  # pragma: no cover - depends on the install
            raise ImportError(
                "Fitting needs scikit-learn, which isn't an install dependency "
                "of lucky-ones -- scoring a snap never calls a solver. Install "
                "the fit extra: `pip install 'lucky-ones[fit]'`, or `uv sync` "
                "in this repo."
            ) from error

        if len(states) != len(next_score):
            raise ValueError(
                f"Got {len(states)} states and {len(next_score)} labels; they "
                "have to line up row for row"
            )
        labels = [ScoreKind(kind).value for kind in next_score]
        if len(set(labels)) < 2:
            raise ValueError(
                "Every snap in the training set was followed by the same thing; "
                "there's nothing to fit"
            )
        fitted = LogisticRegression(C=regularization, max_iter=1000)
        fitted.fit(ep_feature_matrix(states), labels)
        coefficients, intercepts = fitted.coef_, fitted.intercept_
        if len(fitted.classes_) == 2:
            # sklearn drops to binary logistic for two classes and returns one
            # row, for the second class. softmax over [0, z] is that same
            # sigmoid, so a zero row in front of it is the same model written
            # as a multinomial -- which is what everything below expects.
            coefficients = np.vstack([np.zeros_like(coefficients[0]), coefficients[0]])
            intercepts = np.array([0.0, intercepts[0]])
        return cls(
            coefficients=coefficients,
            intercepts=intercepts,
            kinds=[ScoreKind(label) for label in fitted.classes_],
        )

    def predict_proba(self, states: Sequence[GameState]) -> np.ndarray:
        """
        `(len(states), len(self.kinds))` of P(kind | situation), rows summing
        to 1 and columns in `self.kinds` order.
        """
        features = ep_feature_matrix(states)
        if features.shape[1] != self.coefficients.shape[1]:
            raise ValueError(
                f"This fit has {self.coefficients.shape[1]} coefficients per "
                f"class but the current features produce {features.shape[1]} "
                "columns"
            )
        return _softmax(features @ self.coefficients.T + self.intercepts)

    def predict(self, states: Sequence[GameState]) -> np.ndarray:
        """Expected points for the team with the ball. See the protocol."""
        return self.predict_proba(states) @ self.values

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "kinds": [kind.value for kind in self.kinds],
            "coefficients": self.coefficients.tolist(),
            "intercepts": self.intercepts.tolist(),
        }

    @classmethod
    def from_dict(cls, saved: Mapping[str, Any]) -> "MultinomialExpectedPoints":
        """
        Rehydrate a saved fit, refusing one whose features aren't the ones
        this code builds.

        Same check and the same reason as `LogisticWinProbability.from_dict`:
        coefficients are positional, so a feature added to `EP_FEATURE_NAMES`
        silently reinterprets every number in an older file. The classes are
        stored for the same reason -- a fit that never saw a safety has six
        columns, and which six is not something to infer.
        """
        names = tuple(saved["feature_names"])
        if names != EP_FEATURE_NAMES:
            raise ValueError(
                "This fit was made against different features than the current "
                f"code builds.\n  saved:   {names}\n  current: {EP_FEATURE_NAMES}"
            )
        return cls(
            coefficients=np.asarray(saved["coefficients"], dtype=float),
            intercepts=np.asarray(saved["intercepts"], dtype=float),
            kinds=[ScoreKind(kind) for kind in saved["kinds"]],
            feature_names=names,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "MultinomialExpectedPoints":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """
    Row-wise softmax, shifted by the row max first.

    The shift is the whole implementation: `exp` of a raw multinomial logit
    overflows on the situations that matter most -- first and goal at the 1 is
    a genuinely lopsided distribution -- and subtracting the max leaves the
    ratios untouched while keeping every exponent at or below zero.
    """
    shifted = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return shifted / np.sum(shifted, axis=1, keepdims=True)
