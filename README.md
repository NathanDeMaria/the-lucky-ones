# the-lucky-ones

In-game win probability for football: given where a game stands at a snap,
how likely is the home team to win?

The name is the point. A win probability model doesn't say who is better --
it says who is currently ahead of where they need to be, which over a season
is mostly a record of who got the bounces.

```python
from lucky_ones import MODELS, group_by_game

(game,) = group_by_game(plays)
points = MODELS.NCAAFB.curve(game)  # the graph
control = MODELS.NCAAFB.game_control(game)  # the number under it
```

The fits ship inside the package, so that is the whole setup: no bucket, no
credentials, no scikit-learn. See [Using a model](#using-a-model).

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
| `release` | the artifact a fit is stored as |
| `bundled` | the fits that ship with the package, and `MODELS` |

`lucky_ones` itself exports five names — `MODELS`, `group_by_game`, and the
three types on the boundary (`GamePlays`, `CurvePoint`, `GameControl`).
That's what scoring a game needs. Everything else is one import deeper, in
the module that owns it: `lucky_ones.model` for the fit, `lucky_ones.arrow`
for stored plays, `lucky_ones.training` for building a training set. The
split isn't tidiness — the deeper half is where pyarrow and scikit-learn
live, and keeping it out of the top-level import is what makes the small
install work.

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
from lucky_ones.arrow import StorePlaySource

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

That rewrites `lucky_ones/releases/nfl.json` — a `WinProbabilityRelease`: the
coefficients, what they were fit on, and how they scored on held-out games —
which is the file `MODELS.NFL` serves. **Commit it.** The fit ships with the
code, so a retrain is a reviewable diff of eight coefficients and a holdout
score rather than a file that appeared in a bucket. `--out PATH` writes
somewhere else without touching the shipped one.

```
nfl 2025: 20328 plays
Fitting on 134 games (16080 snaps), holding out 34 games
  score_margin           +0.0736
  margin_per_root_time   +5.5357
  ...
Holdout: brier 0.1344, log loss 0.4120 over 34 games
Wrote lucky_ones/releases/nfl.json
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

## Using a model

The fits live in `lucky_ones/releases/*.json` and are packaged into the
wheel, so a consumer gets the model by depending on the package:

```python
from lucky_ones import MODELS, group_by_game

(game,) = group_by_game(plays)

points = MODELS.NFL.curve(game)  # one CurvePoint per snap
control = MODELS.NFL.game_control(game)  # who controlled the game
MODELS.NFL.metrics.brier_score  # how the fit scored on a holdout
MODELS.NFL.trained_on.seasons  # what it was fit on
```

`MODELS.NFL` and `MODELS.NCAAFB` are attributes, so an editor and a type
checker both know them; `MODELS["nfl"]` is there too for a league name that
arrived in a request. Each loads and validates its JSON on first use and
caches it, so importing the package reads nothing.

That's the trade this makes against fetching a release from a bucket:
**pinning the package by rev pins the model with it.** "Which fit is
invisible-string drawing?" has one answer — the rev — instead of depending on
what was at an S3 key when the process started. The cost is that a retrain is
a commit and a release rather than a file copy, which for a model that
changes a few times a season is the right way round. A consumer that does
want to load a release from somewhere else still can:

```python
from lucky_ones.curve import game_control, win_probability_curve
from lucky_ones.release import WinProbabilityRelease

release = WinProbabilityRelease.model_validate_json(raw)
points = win_probability_curve(release.to_model(), game)
control = game_control(points)
```

`points` is one `CurvePoint` per snap, carrying period, clock, both scores
and the home team's win probability — everything an axis label or a tooltip
needs, so the consumer never goes back to the plays.

### The two installs

Scoring a game needs neither the bucket nor scikit-learn, and that's enforced
by the install rather than by convention:

| install | you get | for |
| --- | --- | --- |
| `lucky-ones` | numpy, pydantic | `MODELS`, scoring a game, drawing a curve |
| `lucky-ones[train]` | + pyarrow, scikit-learn | reading stored plays, fitting a model |

The extra is about 250MB — pyarrow is 152MB and scikit-learn ~100MB — against
a scoring path that is nine floats and a logistic function, so a service that
only draws graphs should take the bare install. Everything the small install
can't do fails where it's used, naming the extra, rather than at import:
`LogisticWinProbability.fit` defers its scikit-learn import, and `lucky_ones`
doesn't export anything from `lucky_ones.arrow`, so merely importing the
package can't pull Arrow in. `make wheel` checks the other half — that the
fits are really in the build, which the repo's blanket `*.json` ignore would
otherwise quietly undo.

Nothing currently tests the pyarrow-free import, which is the one property
here that a one-line edit can break silently: the dev environment always has
pyarrow, so every other test passes either way.

In this repo you always have both — `uv sync` installs the `fit` group, which
self-references `lucky-ones[train]`.

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
make wheel    # build, and list the fits that made it into the wheel
make train    # fit a model, rewrite lucky_ones/releases/{league}.json
make curve    # one game's curve and its game control, as JSON
make plays    # aws s3 sync the processed play-by-play down for offline fits
```

Machine-local settings go in `.devcontainer/local.env` (gitignored;
`local.env.example` is the tracked reference) — `AWS_PROFILE`, mainly, since
the plays live in a bucket.

[endgame]: https://github.com/NathanDeMaria/endgame
