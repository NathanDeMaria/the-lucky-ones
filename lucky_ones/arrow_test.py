import pyarrow as pa
import pytest

from .arrow import (
    ArrowPlay,
    DatasetPlaySource,
    StorePlaySource,
    TablePlaySource,
    table_to_plays,
)
from .conftest import make_play, make_table
from .plays import Play, PlaySource


def _accepts_source(source: PlaySource) -> PlaySource:
    """
    A static check with a runtime body. Passing a class here is how `ty`
    reports a source that has drifted from `PlaySource` -- at this line,
    rather than at whichever caller happens to pass one first.
    """
    return source


def test_table_to_plays_reads_a_row():
    table = make_table([make_play(down=3, distance=7, yardline=61)])

    (play,) = table_to_plays(table)

    assert isinstance(play, ArrowPlay)
    assert (play.down, play.distance, play.yardline) == (3, 7, 61)
    # Not 3.0: the whole reason the schema is arrow rather than pandas dtypes
    assert isinstance(play.down, int)


def test_table_to_plays_keeps_nulls_as_none():
    table = make_table([make_play(down=None, distance=None, yardline=None)])

    (play,) = table_to_plays(table)

    assert (play.down, play.distance, play.yardline) == (None, None, None)


def test_a_row_satisfies_the_play_protocol():
    (play,) = table_to_plays(make_table([make_play()]))

    assert isinstance(play, Play)


def test_table_to_plays_names_the_missing_column():
    table = make_table([make_play()]).drop_columns(["yardline"])

    with pytest.raises(KeyError, match="yardline"):
        table_to_plays(table)


def test_table_to_plays_on_an_empty_table():
    assert table_to_plays(make_table([])) == []


async def test_table_source_filters_to_one_game():
    source = TablePlaySource(
        make_table(
            [
                make_play(game_id="a", play_id="a1"),
                make_play(game_id="b", play_id="b1"),
            ]
        )
    )

    plays = await source.load_game("nfl", 2025, 1, "b")

    assert [play.play_id for play in plays] == ["b1"]


async def test_table_source_sorts_by_play_number():
    source = TablePlaySource(
        make_table(
            [
                make_play(play_id="p3", play_number=3),
                make_play(play_id="p1", play_number=1),
                make_play(play_id="p2", play_number=2),
            ]
        )
    )

    plays = await source.load_week("nfl", 2025, 1)

    assert [play.play_id for play in plays] == ["p1", "p2", "p3"]


async def test_table_source_load_weeks():
    source = TablePlaySource(
        make_table(
            [
                make_play(week=1, play_id="w1"),
                make_play(week=2, play_id="w2"),
                make_play(week=3, play_id="w3"),
            ]
        )
    )

    plays = await source.load_weeks("nfl", 2025, [1, 3])

    assert {play.play_id for play in plays} == {"w1", "w3"}
    assert await source.load_weeks("nfl", 2025, []) == []


class _FakeStore:
    """Enough of `ProcessedPlaysStore` for `StorePlaySource`."""

    def __init__(self, table: pa.Table) -> None:
        self.table = table
        self.calls: list[tuple] = []

    async def load_single_game(self, league, season, week, game_id) -> pa.Table:
        self.calls.append(("load_single_game", league, season, week, game_id))
        return self.table

    async def load_week_or_empty(self, league, season, week) -> pa.Table:
        self.calls.append(("load_week_or_empty", league, season, week))
        return self.table

    async def load_weeks(self, league, season, weeks) -> pa.Table:
        self.calls.append(("load_weeks", league, season, list(weeks)))
        return self.table


async def test_store_source_pushes_the_game_filter_down():
    """
    The single-game read has to reach the store, not be done here: that
    filter is what turns a 927KB week into a ~150KB byte-range read.
    """
    store = _FakeStore(make_table([make_play(game_id="g7")]))

    plays = await StorePlaySource(store).load_game("nfl", 2025, 4, "g7")

    assert store.calls == [("load_single_game", "nfl", 2025, 4, "g7")]
    assert [play.game_id for play in plays] == ["g7"]


async def test_store_source_treats_an_unprocessed_week_as_empty():
    """
    `load_week_or_empty`, not `load_week`: the latter raises for a week
    nobody has processed, and that's the normal state of next week.
    """
    store = _FakeStore(make_table([]))

    assert await StorePlaySource(store).load_week("nfl", 2025, 40) == []
    assert store.calls[0][0] == "load_week_or_empty"


def _write_tree(root, rows) -> None:
    """A local copy of endgame's processed layout: one parquet per
    league-week, under the hive-style directories it writes."""
    import pyarrow.parquet as pq

    by_week: dict[tuple[str, int, int], list] = {}
    for row in rows:
        by_week.setdefault((row["league"], row["season"], row["week"]), []).append(row)
    for (league, season, week), week_rows in by_week.items():
        directory = root / f"league={league}" / f"season={season}" / f"week={week}"
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(make_table(week_rows), directory / "data.parquet")


async def test_dataset_source_reads_a_local_tree(tmp_path):
    _write_tree(
        tmp_path,
        [
            make_play(week=1, play_id="w1"),
            make_play(week=2, play_id="w2", game_id="g2"),
        ],
    )
    source = DatasetPlaySource(tmp_path)

    assert [play.play_id for play in await source.load_week("nfl", 2025, 2)] == ["w2"]
    assert [play.play_id for play in await source.load_game("nfl", 2025, 1, "g1")] == [
        "w1"
    ]
    assert len(await source.load_weeks("nfl", 2025, [1, 2])) == 2


async def test_dataset_source_on_a_directory_nobody_has_synced(tmp_path):
    """
    Pointing a training run at a season that isn't downloaded should say "no
    games", not raise from inside pyarrow.
    """
    source = DatasetPlaySource(tmp_path / "not-there")

    assert await source.load_week("nfl", 2025, 1) == []
    assert await source.load_weeks("nfl", 2025, [1, 2]) == []


def test_every_source_is_a_play_source():
    _accepts_source(TablePlaySource(make_table([])))
    _accepts_source(StorePlaySource(_FakeStore(make_table([]))))
    _accepts_source(DatasetPlaySource("plays"))
