# the-lucky-ones

In-game win probability for football: given where a game stands at a snap,
how likely is the home team to win?

The name is the point. A win probability model doesn't say who is better --
it says who is currently ahead of where they need to be, which over a season
is mostly a record of who got the bounces.

## Layout

The pipeline, in the order the modules run:

| module | what it does |
| --- | --- |
| `plays` | what a play is, and where plays come from — protocols only |
| `arrow` | the adapter from endgame's stored parquet to those protocols |
| `game` | plays grouped into games, each with its home side worked out |
| `state` | one snap as the model sees it, pre-snap score and all |
| `features` | a state as a row of numbers |
| `training` | games as a labelled matrix, split by game rather than by row |
| `model` | `WinProbabilityModel`, and a logistic baseline |
| `metrics` | scoring the result |
| `curve` | a game's win probability over time, and game control |
| `release` | the artifact a consumer reads |

### The play data

Plays come from [endgame][endgame]'s processed play-by-play layer: ESPN's
drive JSON, flattened into one parquet row per play under
`endgame_aws.pbp_transform.PLAY_SCHEMA`, partitioned by league / season /
week. `lucky_ones.plays.Play` is the subset of those columns a win
probability model reads, under the same names, so a row of that table
satisfies it structurally.

It is a Protocol, and so is `PlaySource`, because nothing downstream should
care where a play came from: `lucky_ones` depends on the shape of a play, not
on a bucket. `lucky_ones.arrow` is the only module that imports pyarrow or
knows what endgame is, and `endgame_aws` is deliberately *not* a dependency of
this package — `StorePlaySource` takes the store as an argument, typed against
a local Protocol. Wire it up where the application is wired up:

```python
from endgame_aws.pbp_parquet import get_processed_plays_store
from lucky_ones import StorePlaySource

source = StorePlaySource(get_processed_plays_store())
plays = await source.load_weeks("nfl", 2025, range(1, 19))
```

Tests, notebooks, and anything with a table already in hand use
`TablePlaySource` instead, and never touch S3.

Two things the schema doesn't give you, both handled here rather than left to
each caller:

- **which team is home.** Scores arrive as `home_score` / `away_score` and
  possession as `offense_team_id`, with nothing tying them together, and
  endgame's `Game.home` is a display name rather than the numeric team id the
  plays carry. `lucky_ones.game.infer_home_team_id` votes on it from the
  scoring drives; pass `home_team_ids` to `group_by_game` when you know it
  from somewhere authoritative. A game it can't call is dropped, not guessed.
- **the scores are cumulative *after* the play.** The touchdown play already
  carries its touchdown, so training on it leaks the result into the
  features. `iter_states` takes the score from the previous play.

## Training a model

```sh
make plays                                        # aws s3 sync, once
make train ARGS="--league nfl --seasons 2022-2025 --root ./plays"
```

That writes `models/nfl.json` — a `WinProbabilityRelease`: the coefficients,
what they were fit on, and how they scored on held-out games.

```
nfl 2025: 20328 plays
Fitting on 134 games (16080 snaps), holding out 34 games
  score_margin           +0.0736
  margin_per_root_time   +5.5357
  ...
Holdout: brier 0.1344, log loss 0.4120 over 34 games
Wrote models/nfl.json
```

Two ways to get the plays. `--root` reads a local copy of endgame's
processed tree, which is reproducible and needs no credentials; without it
the script reads the bucket through `endgame_aws`, which needs `AWS_PROFILE`
and `~/.aws-batch/config.json`. Everything downstream sees the same
`PlaySource` either way — the script is the only place that decides.

Then eyeball a game:

```sh
make curve ARGS="401671789 --week 3"
```

```
{"game_id": "401671789", "home_team_id": "...", "away_team_id": "...",
 "game_control": {"home": 0.804, "away": 0.196, "seconds": 3600},
 "points": [{"period": 1, "clock_seconds": 900, "home_score": 0,
             "home_win_probability": 0.464}, ...]}
```

No AWS and no data at all? `synthetic.py` writes a football-shaped tree in
the same layout, which is what the tests train on:

```sh
uv run python synthetic.py --root ./plays --weeks 12 --games-per-week 14
```

It is a fixture, not a simulator — never fit a model you mean to use on it.

## Reading a release from another service

A release is JSON, and rehydrating it needs neither the bucket nor
scikit-learn — which is what lets invisible-string draw a curve from a fit it
didn't make. Same arrangement as cassandra's `ModelRelease`, and for the same
reason: depend on this package by rev, read the artifact, never run the
fitting stack.

```python
from lucky_ones import WinProbabilityRelease, game_control, win_probability_curve

release = WinProbabilityRelease.model_validate_json(raw)
points = win_probability_curve(release.to_model(), game)  # the graph
control = game_control(points)  # the number under it
```

`points` is one `CurvePoint` per snap, carrying period, clock, both scores
and the home team's win probability — everything an axis label or a tooltip
needs, so the consumer never goes back to the plays.

Publish somewhere the consumer can reach:

```sh
make train ARGS="--league nfl --seasons 2022-2025 --out releases/nfl/latest.json"
aws s3 cp releases/nfl/latest.json s3://BUCKET/win_probability/nfl/latest.json
```

A sibling prefix to invisible-string's `models/` rather than a key inside it:
its release reader validates everything under that prefix against cassandra's
`ModelRelease`, and a win probability release is a different artifact, not a
malformed one.

## Game control

`game_control` is the average win probability over a game, weighted by how
long each one was on the board.

An unweighted mean over snaps counts a two-minute drill's fifteen plays the
same as a quarter of grinding, so a team that trailed all game and won on the
last drive comes out looking like it was in control throughout. Weighting by
elapsed clock is what makes the number mean "most of the game".

Read it as a share of the game controlled, not as a win probability: 0.80
doesn't say the home team was ever 80% to win, it says that averaged over
sixty minutes, that's where the model had them. Both sides sum to 1, and
`seconds` says what the average covers — regulation only, since college
overtime has no clock to weight by.

The synthetic game above is the case worth having it for: 0.80 control for a
team that led 20-17 with a minute left and lost.

## Development

Open the repo in the devcontainer (`.devcontainer/`) — python 3.14, uv, the
GitHub and AWS CLIs, and Claude Code, with `uv sync` run on create. Outside
it, `uv` is the only prerequisite; every target below syncs from `uv.lock`
before it runs.

```
make test     # pytest
make lint     # ruff (fix + format) and ty
make check    # the same checks, reporting instead of fixing -- what CI runs
make train    # fit a model, write models/{league}.json
make curve    # one game's curve and its game control, as JSON
make plays    # aws s3 sync the processed play-by-play down for offline fits
```

Machine-local settings go in `.devcontainer/local.env` (gitignored;
`local.env.example` is the tracked reference) — `AWS_PROFILE`, mainly, since
the plays live in a bucket.

[endgame]: https://github.com/NathanDeMaria/endgame
