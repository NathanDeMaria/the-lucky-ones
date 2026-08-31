"""
Fixtures for building plays that look like endgame's, without a bucket.

`play` writes one row of `PLAY_SCHEMA` with sane defaults so a test only has
to state the fields it cares about, and `plays_table` turns a list of them
into the Arrow table `lucky_ones.arrow` reads -- typed exactly as the real
one is, so a test catches the int8/None problems a dict of python objects
would hide.
"""

from typing import Any, Sequence

import pyarrow as pa
import pytest

# The subset of endgame's PLAY_SCHEMA that `lucky_ones.arrow.PLAY_COLUMNS`
# reads, with its types. A test table has to be typed like the real thing:
# `down` is an int8 that is genuinely null on a kickoff, and a python dict
# would let a float through where the model will see an int.
TEST_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("week", pa.int8(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("play_id", pa.string(), nullable=False),
        pa.field("play_number", pa.int16(), nullable=False),
        pa.field("period", pa.int8()),
        pa.field("clock_seconds", pa.int16()),
        pa.field("wallclock", pa.timestamp("ms", tz="UTC")),
        pa.field("home_score", pa.int16()),
        pa.field("away_score", pa.int16()),
        pa.field("offense_team_id", pa.string()),
        pa.field("defense_team_id", pa.string()),
        pa.field("down", pa.int8()),
        pa.field("distance", pa.int16()),
        pa.field("yardline", pa.int8()),
        pa.field("play_type", pa.string()),
        pa.field("scoring_play", pa.bool_()),
        pa.field("is_penalty", pa.bool_()),
        pa.field("is_turnover", pa.bool_()),
        pa.field("drive_id", pa.string()),
        pa.field("drive_number", pa.int16(), nullable=False),
        pa.field("drive_team_id", pa.string()),
        pa.field("drive_result", pa.string()),
        pa.field("drive_is_score", pa.bool_()),
    ]
)

_DEFAULTS: dict[str, Any] = {
    "league": "nfl",
    "season": 2025,
    "week": 1,
    "game_id": "g1",
    "play_id": "p1",
    "play_number": 1,
    "period": 1,
    "clock_seconds": 900,
    "wallclock": None,
    "home_score": 0,
    "away_score": 0,
    "offense_team_id": "home",
    "defense_team_id": "away",
    "down": 1,
    "distance": 10,
    "yardline": 25,
    "play_type": "Rush",
    "scoring_play": False,
    "is_penalty": False,
    "is_turnover": False,
    "drive_id": "d1",
    "drive_number": 1,
    "drive_team_id": None,
    "drive_result": None,
    "drive_is_score": False,
}


def make_play(**overrides: Any) -> dict[str, Any]:
    """
    One play row. `drive_team_id` follows `offense_team_id` unless it's
    overridden, since that's true of every play that isn't a turnover return.
    """
    row = {**_DEFAULTS, **overrides}
    if "drive_team_id" not in overrides:
        row["drive_team_id"] = row["offense_team_id"]
    return row


def make_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=TEST_SCHEMA)


@pytest.fixture(name="make_play")
def make_play_fixture():
    return make_play


@pytest.fixture(name="make_table")
def make_table_fixture():
    return make_table
