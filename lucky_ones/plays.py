"""
What a play is, as far as a win probability model is concerned.

Everything here is a Protocol. Nothing in this module imports pyarrow, boto,
or endgame, and nothing that models plays should import anything that does --
the point is that `lucky_ones` depends on the *shape* of a play, not on where
plays come from. `lucky_ones.arrow` is the one module that knows about
endgame's storage, and it's an adapter: it turns a table into `Play`s and
stops there.

The shape itself is endgame's. Its processed play-by-play layer
(`endgame_aws.pbp_transform.PLAY_SCHEMA`) flattens ESPN's drive JSON into one
row per play with every column a scalar, and `Play` below is the subset of
those columns a win probability model reads, under the same names. Two
consequences worth knowing:

- A row of that table satisfies `Play` structurally. That's deliberate --
  `lucky_ones.arrow.table_to_plays` produces something that matches rather
  than something that has to be mapped.
- Almost every field is optional. In the source data down and distance are
  absent on a kickoff, yardline is absent on an administrative play
  (END QUARTER, END GAME), and a clock that didn't parse is None. A model has
  to decide what to do about that; it does not get to assume it away, so the
  types say so.

The one thing the schema does *not* carry is which team is home. Scores come
as `home_score`/`away_score` and possession as `offense_team_id`, with
nothing tying the two together -- see `lucky_ones.game` for what to do about
that.
"""

from datetime import datetime
from typing import Iterable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Play(Protocol):
    """
    One play, in the columns `PLAY_SCHEMA` spells out.

    Read-only properties rather than attributes so that a class with
    `__slots__`, a NamedTuple, a frozen dataclass and a row view over an
    Arrow table all satisfy it -- a Protocol declaring a mutable attribute
    would rule the last two out.

    `runtime_checkable` so `isinstance` works in a test. Note what that does
    and doesn't buy: it checks that the names exist, never their types.
    """

    # --- Identity ------------------------------------------------------
    @property
    def league(self) -> str: ...

    @property
    def season(self) -> int: ...

    @property
    def week(self) -> int: ...

    @property
    def game_id(self) -> str: ...

    @property
    def play_id(self) -> str: ...

    @property
    def play_number(self) -> int:
        """1-based position in the game, in the order ESPN sent the drives."""

    # --- Clock and score -----------------------------------------------
    @property
    def period(self) -> int | None:
        """Quarter, 1-4, then 5+ for overtime."""

    @property
    def clock_seconds(self) -> int | None:
        """Seconds left in `period`, not in the game."""

    @property
    def wallclock(self) -> datetime | None: ...

    @property
    def home_score(self) -> int | None:
        """Cumulative *after* this play. See `lucky_ones.state`."""

    @property
    def away_score(self) -> int | None: ...

    # --- Situation at the snap -----------------------------------------
    @property
    def offense_team_id(self) -> str | None: ...

    @property
    def defense_team_id(self) -> str | None: ...

    @property
    def down(self) -> int | None: ...

    @property
    def distance(self) -> int | None:
        """
        Yards to a first down. Can be negative -- a penalty enforced from
        behind the spot -- which is why the source column is an int16.
        """

    @property
    def yardline(self) -> int | None:
        """
        Field position on one absolute scale: 1 is the offense's own 1, 99 is
        the defense's. Already normalized by endgame, which matters because
        ESPN's own `yardLine` means different things in the NFL and college.
        """

    # --- The play itself -----------------------------------------------
    @property
    def play_type(self) -> str | None: ...

    @property
    def text(self) -> str | None:
        """
        ESPN's sentence about the play ("Hurts pass intercepted by Slay").

        The only column here that isn't a number or a flag, and the only place
        the *manner* of a play is recorded: `is_turnover` says the ball
        changed hands, and nothing but this says it changed hands off a tipped
        ball. `lucky_ones.luck` is what reads it. Nothing in the model does --
        a feature over free text is not what the eight coefficients are.
        """

    @property
    def scoring_play(self) -> bool | None: ...

    @property
    def is_penalty(self) -> bool | None: ...

    @property
    def is_turnover(self) -> bool | None: ...

    # --- The drive it belongs to ---------------------------------------
    @property
    def drive_id(self) -> str | None: ...

    @property
    def drive_number(self) -> int: ...

    @property
    def drive_team_id(self) -> str | None: ...

    @property
    def drive_result(self) -> str | None: ...

    @property
    def drive_is_score(self) -> bool | None: ...


class PlaySource(Protocol):
    """
    Where plays come from.

    Three methods, matching the three questions this package asks: one game
    (scoring a game that's happening), a range of weeks (assembling a
    training set), and one whole week.

    Async because the implementation that matters reads S3, and a sync
    signature would either block an event loop or force every caller into a
    thread. An in-memory source satisfies this too -- see
    `lucky_ones.arrow.TablePlaySource` -- it just returns immediately.

    Every method returns plays sorted by `play_number` within a game, which
    is the order `lucky_ones.state` walks them in. A source that can't
    promise that should sort before returning rather than leaving each
    caller to.

    A game with no play-by-play comes back empty rather than raising. That is
    the normal case for most of an NCAAFB week, not an error: ESPN simply has
    no plays for a D2 opponent.
    """

    async def load_game(
        self, league: str, season: int, week: int, game_id: str
    ) -> Sequence[Play]:
        """One game's plays. Empty if there aren't any."""

    async def load_week(self, league: str, season: int, week: int) -> Sequence[Play]:
        """Every play of a week, across games."""

    async def load_weeks(
        self, league: str, season: int, weeks: Iterable[int]
    ) -> Sequence[Play]:
        """
        Several weeks of one season, concatenated.

        A week that hasn't been processed is skipped, not an error -- asking
        for a whole season in September is normal.
        """
