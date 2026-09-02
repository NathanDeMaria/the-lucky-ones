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
earned = MODELS.NCAAFB.luck_adjusted_game_control(game)  # minus the bounces
breaks = MODELS.NCAAFB.lucky_wp(game)  # and what the bounces were worth
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
| `luck` | the coin flips: the curve without them, and what they were worth |
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

One column stands apart from the rest: `text`, ESPN's sentence about the play.
Nothing the model sees is built from it — the eight features are numbers and
flags — but it's the only place the *manner* of a play is recorded.
`is_turnover` says the ball changed hands; nothing but the text says it changed
hands off a tipped ball. `lucky_ones.luck` is what reads it.

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
 "luck_adjusted_game_control": {"home": 0.731, "away": 0.269, "seconds": 3600},
 "lucky_wp": {"home": 0.041, "away": 0.187, "net": -0.146},
 "lucky_plays": [{"play_id": "...", "kind": "fumble_lost", "retained": 0.5,
                  "changed_possession": true, "realized": 0.62,
                  "counterfactual": 0.81, "expected": 0.715,
                  "home_delta": -0.095}, ...],
 "points": [{"period": 1, "clock_seconds": 900, "home_score": 0,
             "home_win_probability": 0.464,
             "luck_adjusted_win_probability": 0.464}, ...]}
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
earned = MODELS.NFL.luck_adjusted_game_control(game)  # who earned it
breaks = MODELS.NFL.lucky_wp(game)  # what the bounces were worth
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
| `lucky-ones[data]` | + pyarrow (152MB) | reading endgame's stored plays |
| `lucky-ones[fit]` | + scikit-learn (~100MB) | fitting a model |
| `lucky-ones[data,fit]` | both | this repo, and anything that trains |

**The two extras are separate because the jobs are.** A service that draws
curves for games it didn't fit needs Arrow and no solver; a caller with a
training set already in hand needs the solver and no Arrow. One `[train]`
extra covering both made the first case carry ~100MB of scikit-learn to
satisfy an import it never makes — reading a stored fit should never require
the ability to produce one.

Together they are about 250MB against a scoring path that is nine floats and a
logistic function, so a service that only draws graphs should take
`[data]` and a bare install should be the default assumption. Everything the
small install can't do fails where it's used, naming the extra that fixes it,
rather than at import: `LogisticWinProbability.fit` defers its scikit-learn
import, and `lucky_ones` doesn't export anything from `lucky_ones.arrow`, so
merely importing the package can't pull Arrow in. `make wheel` checks the
other half — that the fits are really in the build, which the repo's blanket
`*.json` ignore would otherwise quietly undo.

Nothing currently tests the pyarrow-free import, which is the one property
here that a one-line edit can break silently: the dev environment always has
pyarrow, so every other test passes either way.

In this repo you always have both — `uv sync` installs the `train` group,
which self-references `lucky-ones[data,fit]`.

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

## Luck-adjusted game control

`game_control` measures what happened, which is also what hands a team full
credit for a fumble that bounced their way. `luck_adjusted_game_control`
answers the neighbouring question: **how much of the game would this team have
controlled if the fifty-fifty balls had gone fifty-fifty?**

```python
control = MODELS.NFL.game_control(game)  # 0.68 -- what happened
earned = MODELS.NFL.luck_adjusted_game_control(game)  # 0.59 -- on purpose
```

Report them as a pair. The gap between them is the bounces.

### How it works

Three steps, in `lucky_ones.luck`:

1. **Find the coin flips.** `find_lucky_plays` reads `Play.text` — ESPN's
   sentence about the play, the only column that records the *manner* of one —
   for the plays whose result was decided by a bounce.
2. **Ask the model what the other branch was worth.** Every play it finds has
   two outcomes that matter: the ball changed hands or it didn't. The one that
   didn't happen is a `GameState` the code can build and the fit can price, so
   the counterfactual is the model's own number rather than a constant someone
   picked. The ball sits at the spot of the snap and the only thing that moves
   is who has it — same team on the next down, or the other team first and ten
   at the mirrored yardline.
3. **Replace the swing with the average of the two.** The play's win
   probability move becomes `retained × what happened + (1 − retained) × the
   other branch`, and every later point on the curve carries the difference
   forward.

That last part is the point rather than a rounding decision: a fumble returned
for a touchdown in the first quarter is still on the scoreboard in the fourth,
so a curve that discounts the recovery has to keep discounting it. The blend is
done in probability space, because it's an expectation over outcomes; the
accumulation is in log-odds, because that's the fit's own additive scale and
because a shifted curve then stays inside (0, 1) with no clamp bending the
endgame back into range.

### What counts, and what it's worth

`retained` is a probability: how often the outcome that happened is the one
that happens.

| kind | `retained` | why |
| --- | --- | --- |
| `fumble_lost` | 0.5 | loose-ball recovery is a coin flip; neither side recovers its own at a rate that holds up year over year |
| `fumble_kept` | 0.5 | the same coin, seen from the side that got away with one |
| `tipped_interception` | 0.25 | a ball off a hand or a helmet falls incomplete far more often than a defender comes down with it |

Those are estimates, not fits — there's no labelled set of counterfactual
fumbles to fit them on. They're a keyword argument everywhere they're used, so
a caller who disagrees says so at the call site:

```python
from lucky_ones.luck import DEFAULT_RETAINED, LuckKind

MODELS.NFL.luck_adjusted_game_control(
    game, retained={**DEFAULT_RETAINED, LuckKind.TIPPED_INTERCEPTION: 0.4}
)
```

`retained=1.0` for every kind reproduces the unadjusted curve exactly, which
is what the tests check — so if the two numbers ever differ by the arithmetic
rather than by the luck, that fails.

The list is short because each entry has to clear two bars: the play text says
it happened, and the outcome that *didn't* happen is one this code can build a
`GameState` for. Left out, deliberately: muffed punts and kickoff fumbles
(`iter_states` keeps scrimmage snaps only, so a kickoff isn't on the curve to
adjust); blocked kicks (the block is a play someone made, and only the bounce
after it is luck); a field goal off the upright (no branch to build);
untipped interceptions (a throw into coverage is a decision).

### Three things it isn't

- **A prediction.** The adjusted curve doesn't end at 0 or 1 the way the real
  one does. The game had a winner; this is a counterfactual, and its last point
  is where the game *would* have stood.
- **Symmetric in what it can see.** A fumble the offense recovered is in the
  play text, so a team that got away with one is charged for it. A pass that
  was tipped and *not* intercepted mostly isn't in the text at all, so it
  can't be. That's a real gap; what keeps it from being fatal is that a
  near-miss's realized swing is already near zero — what's missing is the
  discount on the other side of the same coin, which is small.
- **A change to the model.** The fit is untouched. This reweights the curve
  that fit produces; nothing here is retrained, and no release changes.

The counterfactual is an approximation, and the shape of its error is worth
knowing: it prices a turnover at the line of scrimmage, ignoring the yards the
play gained before the ball came loose and the yards a return added after.
Against the thing being measured — who has the ball — that's small. Twenty-five
yards of field position is worth a few points of win probability; the
possession itself is worth several times that.

## Lucky WP

The other thing worth saying about a bounce. `luck_adjusted_game_control`
rewrites the game without it; `lucky_wp` just adds up what it was worth:
**how much win probability did the bounces hand each team?**

```python
breaks = MODELS.NFL.lucky_wp(game)
breaks.home  # 0.04 -- win probability the bounces gave the home team
breaks.away  # 0.19 -- and the away team
breaks.net  # -0.15
```

For each lucky play there are two branches and a gap between them. The part of
that gap the *bounce* decided, rather than the team, is `(1 − retained)` of it:

```
home_delta = realized − expected
           = realized − (retained × realized + (1 − retained) × counterfactual)
           = (1 − retained) × (realized − counterfactual)
```

A coin flip hands out half the gap between its two branches. A one-in-four
bounce — a tipped ball that got caught — hands out three quarters of it. Those
deltas total per team, and `swings` carries each one with both branches on it,
so a table of "the plays this game turned on" is a list comprehension.

### Reading it next to the other number

|  | `luck_adjusted_game_control` | `lucky_wp` |
| --- | --- | --- |
| what it does | rewrites the game | totals the breaks |
| units | share of the game, sums to 1 | win probability, sums to nothing |
| a bounce counts | once, and every point after it | once |
| when it happened | matters — it's time-weighted | doesn't |
| answers | who was ahead of this game on merit | who got the breaks, and how big |

The middle two rows are the substance. Control is an average over the clock, so
a first-quarter fumble is discounted for the rest of the game and a bounce with
a minute left barely moves it. `lucky_wp` is a sum over plays, so the late
bounce that decided the game counts for exactly what it swung, and counts once.
Neither is more correct — they're answers to different questions, and a team
can be near the top of one and the middle of the other.

`home` and `away` are kept as two non-negative totals rather than netted
because "both teams got a big break" and "neither got one" are different games
that `net` alone reports identically.

### The one thing to watch

**The tipped balls are counted one-sidedly, and here that costs more than it
does next door.** A tipped pass that *was* intercepted is in ESPN's play text;
one that bounced off a helmet and fell incomplete mostly isn't. So a defense is
charged for the tips that went its way, and an offense is never credited for
the ones that didn't — and a dropped tip is worth 0.25 of a full turnover swing,
which is not a rounding error.

In `luck_adjusted_game_control` a missing play shifts a curve a little. In a
running total it's a one-way leak. Fumbles have no such gap: both branches are
in the text, so `fumble_lost` and `fumble_kept` are both found and both
charged.

If you want only the half with both branches observed, turn the tipped balls
off — `retained=1.0` zeroes a kind's contribution without removing it from the
report:

```python
from lucky_ones.luck import DEFAULT_RETAINED, LuckKind

MODELS.NFL.lucky_wp(
    game, retained={**DEFAULT_RETAINED, LuckKind.TIPPED_INTERCEPTION: 1.0}
)
```

One more approximation worth knowing, shared with the number next door: the
realized branch is read off the *next* snap, after the clock ran and any points
went up, while the counterfactual is built from the snap itself. A play's worth
of clock is small against a change of possession, but it's the floor under how
precise either number can be.

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
