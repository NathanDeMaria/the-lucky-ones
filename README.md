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

async with get_processed_plays_store() as store:
    plays = await StorePlaySource(store).load_weeks("nfl", 2025, range(1, 19))
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

## Fitting

```python
from lucky_ones import (
    LogisticWinProbability,
    build_training_set,
    group_by_game,
    iter_states,
    split_games,
)

games = group_by_game(plays)
train, holdout = split_games(games)  # by game, never by row
training = build_training_set(train)
model = LogisticWinProbability.fit(training.states, training.home_won)
model.save("nfl.json")
```

The baseline is logistic regression on eight features, and it is a baseline:
readable coefficients, a fit that takes a second, and the thing anything more
elaborate has to beat before it earns its complexity. Scoring a game needs
neither the store nor scikit-learn — a saved fit is a JSON file of
coefficients, and `predict` is one matrix multiply:

```python
model = LogisticWinProbability.load("nfl.json")
probabilities = model.predict(list(iter_states(game)))
```

That split is what the `fit` dependency group in `pyproject.toml` is for:
scikit-learn is imported inside `fit`, so a serving install never pays for it.

## Development

Open the repo in the devcontainer (`.devcontainer/`) — python 3.14, uv, the
GitHub and AWS CLIs, and Claude Code, with `uv sync` run on create. Outside
it, `uv` is the only prerequisite; every target below syncs from `uv.lock`
before it runs.

```
make test     # pytest
make lint     # ruff (fix + format) and ty
make check    # the same checks, reporting instead of fixing -- what CI runs
```

Machine-local settings go in `.devcontainer/local.env` (gitignored;
`local.env.example` is the tracked reference) — `AWS_PROFILE`, mainly, since
the plays live in a bucket.

[endgame]: https://github.com/NathanDeMaria/endgame
