"""
The adapter between endgame's stored play-by-play and `lucky_ones.plays`.

This is the only module in the package that imports pyarrow or knows what a
bucket is. Everything downstream of it takes `Play` and `PlaySource`, so
swapping the storage layer -- or handing the model plays out of a fixture, a
CSV, or a live feed -- is a new adapter here and nothing else.

Two sources live here:

- `TablePlaySource`, over a table already in memory. What tests and notebooks
  use, and what you get after reading a parquet file yourself.
- `StorePlaySource`, over endgame's `ProcessedPlaysStore`. It takes the store
  as a constructor argument typed against `ProcessedPlaysStoreLike` below, so
  this package doesn't depend on `endgame_aws` -- import the store where you
  wire the application up and pass it in.
"""

from datetime import datetime
from functools import reduce
from operator import and_
from typing import Any, Iterable, NamedTuple, Protocol, Sequence

import pyarrow as pa
import pyarrow.dataset as ds

from .plays import Play


class ArrowPlay(NamedTuple):
    """
    A `Play` as a value.

    A NamedTuple rather than a view onto the table: a row view keeps the
    whole table alive for as long as one play is referenced, and the win
    probability code reads nearly every field of nearly every play, so the
    lazy access a view buys never pays for itself.

    Field order matches `PLAY_SCHEMA`'s so that the two read side by side.
    """

    league: str
    season: int
    week: int
    game_id: str
    play_id: str
    play_number: int
    period: int | None
    clock_seconds: int | None
    wallclock: datetime | None
    home_score: int | None
    away_score: int | None
    offense_team_id: str | None
    defense_team_id: str | None
    down: int | None
    distance: int | None
    yardline: int | None
    play_type: str | None
    scoring_play: bool | None
    is_penalty: bool | None
    is_turnover: bool | None
    drive_id: str | None
    drive_number: int
    drive_team_id: str | None
    drive_result: str | None
    drive_is_score: bool | None


# The columns `ArrowPlay` needs, which is a subset of what endgame writes:
# `text`, `sequence_number`, the end-of-play situation and the drive's yardage
# are all in the parquet and none of them are read here. Selecting explicitly
# rather than taking the whole row keeps the conversion off columns nothing
# uses, and turns a renamed column into an error naming the column instead of
# a TypeError about arguments.
PLAY_COLUMNS: tuple[str, ...] = ArrowPlay._fields


def table_to_plays(table: pa.Table) -> list[ArrowPlay]:
    """
    Every row of `table` as an `ArrowPlay`, in the order the table holds
    them.

    Raises `KeyError` naming what's missing if the table doesn't carry
    `PLAY_COLUMNS` -- which is what reading a raw (drive JSON) week instead
    of a processed one looks like.
    """
    missing = [name for name in PLAY_COLUMNS if name not in table.column_names]
    if missing:
        raise KeyError(f"Table is missing play columns: {', '.join(missing)}")
    return [ArrowPlay(**row) for row in table.select(PLAY_COLUMNS).to_pylist()]


def sort_plays(plays: Iterable[Play]) -> list[Play]:
    """
    Plays in the order a game happened: by game, then by `play_number`.

    `PlaySource` promises this ordering and `lucky_ones.state` relies on it,
    so every source here ends with this rather than trusting its input.
    """
    return sorted(plays, key=lambda play: (play.game_id, play.play_number))


class TablePlaySource:
    """
    A `PlaySource` over one in-memory table.

    Filters with dataset expressions rather than pandas so that nothing here
    pulls in a DataFrame library, and so that a table spanning several
    league-weeks (what `ProcessedPlaysStore.load_weeks` returns) can be
    handed straight in.
    """

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    async def load_game(
        self, league: str, season: int, week: int, game_id: str
    ) -> Sequence[Play]:
        return self._filter(
            _matches(league=league, season=season, week=week, game_id=game_id)
        )

    async def load_week(self, league: str, season: int, week: int) -> Sequence[Play]:
        return self._filter(_matches(league=league, season=season, week=week))

    async def load_weeks(
        self, league: str, season: int, weeks: Iterable[int]
    ) -> Sequence[Play]:
        wanted = list(weeks)
        if not wanted:
            return []
        expression = _matches(league=league, season=season)
        return self._filter(expression & ds.field("week").isin(wanted))

    def _filter(self, expression: ds.Expression) -> list[Play]:
        return sort_plays(table_to_plays(self._table.filter(expression)))


def _matches(**equals: Any) -> ds.Expression:
    """`column == value` for every keyword, and-ed together."""
    terms = [ds.field(column) == value for column, value in equals.items()]
    return reduce(and_, terms)


class ProcessedPlaysStoreLike(Protocol):
    """
    The part of endgame's `ProcessedPlaysStore` that `StorePlaySource` uses.

    Declared here rather than imported so that `endgame_aws` -- and through
    it aiobotocore, and a bucket to point at -- stays out of this package's
    dependencies. `ProcessedPlaysStore` satisfies it structurally; so does a
    stub in a test.

    `load_week_or_empty` rather than `load_week`: the store's plain
    `load_week` raises for a week nobody has processed, and this package
    treats "not processed yet" as empty. See `PlaySource`.
    """

    async def load_single_game(
        self, league: str, season: int, week: int, game_id: str
    ) -> pa.Table: ...

    async def load_week_or_empty(
        self, league: str, season: int, week: int
    ) -> pa.Table: ...

    async def load_weeks(
        self, league: str, season: int, weeks: Iterable[int]
    ) -> pa.Table: ...


class StorePlaySource:
    """
    A `PlaySource` backed by endgame's processed parquet layer.

    Wire it up with the store from `endgame_aws.pbp_parquet`:

        async with get_processed_plays_store() as store:
            source = StorePlaySource(store)

    Each call is one read of one week's object -- the store pushes the
    `game_id` filter into the parquet reader, so `load_game` moves a fraction
    of the week rather than all of it.
    """

    def __init__(self, store: ProcessedPlaysStoreLike) -> None:
        self._store = store

    async def load_game(
        self, league: str, season: int, week: int, game_id: str
    ) -> Sequence[Play]:
        table = await self._store.load_single_game(league, season, week, game_id)
        return sort_plays(table_to_plays(table))

    async def load_week(self, league: str, season: int, week: int) -> Sequence[Play]:
        table = await self._store.load_week_or_empty(league, season, week)
        return sort_plays(table_to_plays(table))

    async def load_weeks(
        self, league: str, season: int, weeks: Iterable[int]
    ) -> Sequence[Play]:
        table = await self._store.load_weeks(league, season, list(weeks))
        return sort_plays(table_to_plays(table))
