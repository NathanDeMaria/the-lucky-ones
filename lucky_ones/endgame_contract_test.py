"""
The play data really is endgame's.

`lucky_ones.plays.Play` names columns from `endgame_aws.pbp_transform`'s
PLAY_SCHEMA and nothing enforces that but this file. A column renamed
upstream would otherwise surface as an empty table filter or a KeyError
during a training run, at the point where someone is least able to guess
why.

Skipped where endgame_aws isn't installed -- it's a `fit` group dependency,
so a serving install genuinely doesn't have it and shouldn't fail its tests
over it.
"""

import pytest

from .arrow import PLAY_COLUMNS
from .conftest import TEST_SCHEMA

pbp_transform = pytest.importorskip(
    "endgame_aws.pbp_transform", reason="endgame_aws is a fit-group dependency"
)


def test_every_column_this_package_reads_exists_upstream():
    upstream = set(pbp_transform.PLAY_SCHEMA.names)

    assert set(PLAY_COLUMNS) <= upstream, (
        f"endgame's schema no longer has {sorted(set(PLAY_COLUMNS) - upstream)}"
    )


def test_the_test_fixtures_are_typed_like_the_real_thing():
    """
    The fixture schema in conftest is a hand-copied subset. If a type drifts,
    every test in this suite is exercising something the real data isn't --
    a float64 down, say, where production has an int8 that is genuinely null
    on a kickoff.
    """
    upstream = pbp_transform.PLAY_SCHEMA

    mismatched = {
        name: (str(TEST_SCHEMA.field(name).type), str(upstream.field(name).type))
        for name in PLAY_COLUMNS
        if TEST_SCHEMA.field(name).type != upstream.field(name).type
    }

    assert not mismatched


def test_a_real_transformed_game_satisfies_the_play_protocol():
    """
    End to end against endgame's own transform: ESPN-shaped drive JSON in,
    `Play`s out, with no mapping layer in between. This is the claim the
    protocol makes, checked rather than asserted in a docstring.
    """
    from .arrow import table_to_plays
    from .plays import Play

    drives = [
        {
            "id": "1",
            "team": {"id": "12"},
            "result": "TD",
            "yards": 75,
            "offensivePlays": 8,
            "isScore": True,
            "plays": [
                {
                    "id": "4016717890001",
                    "sequenceNumber": "1",
                    "period": {"number": 1},
                    "clock": {"displayValue": "12:17"},
                    "homeScore": 0,
                    "awayScore": 0,
                    "start": {
                        "team": {"id": "12"},
                        "down": 1,
                        "distance": 10,
                        "yardsToEndzone": 75,
                    },
                    "end": {
                        "team": {"id": "12"},
                        "down": 2,
                        "distance": 6,
                        "yardsToEndzone": 71,
                    },
                    "type": {"id": "5", "text": "Rush"},
                    "text": "Run for 4 yards",
                    "statYardage": 4,
                    "scoringPlay": False,
                    "teamParticipants": [{"type": "defense", "id": "33"}],
                }
            ],
        }
    ]

    table = pbp_transform.transform_game_to_table(
        drives, game_id="401671789", league="nfl", season=2025, week=1
    )
    (play,) = table_to_plays(table)

    assert isinstance(play, Play)
    assert (play.down, play.distance) == (1, 10)
    # endgame normalizes to "distance from the offense's own goal line"
    assert play.yardline == 25
    assert play.offense_team_id == "12"
    assert play.defense_team_id == "33"
    assert play.clock_seconds == 12 * 60 + 17
    assert play.drive_is_score is True
