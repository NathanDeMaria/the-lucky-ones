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
epa = MODELS.NCAAFB.epa_per_play(game)  # and how they actually played
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
| `features` | a state as a row of numbers, for either model |
| `training` | games as labelled matrices, split by game rather than by row |
| `model` | `WinProbabilityModel`, and a logistic baseline |
| `points` | `ExpectedPointsModel`: what a snap is worth on the scoreboard |
| `metrics` | scoring the result |
| `curve` | a game's win probability over time, and game control |
| `luck` | the coin flips: the curve without them, and what they were worth |
| `epa` | expected points added per play, bounded and weighted |
| `release` | the artifacts a fit is stored as |
| `bundled` | the fits that ship with the package, and `MODELS` |

There are two fits per league and they answer different questions: win
probability is about the game, expected points is about the snap.
`epa_per_play` is the one call that reads both.

`lucky_ones` itself exports six names — `MODELS`, `group_by_game`, and the
four types on the boundary (`GamePlays`, `CurvePoint`, `GameControl`,
`EpaPerPlay`). That's what scoring a game needs. Everything else is one
import deeper, in the module that owns it: `lucky_ones.model` for the fit,
`lucky_ones.arrow` for stored plays, `lucky_ones.training` for a training set. The
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
- **and before 2014 they jitter.** Points appear on a play, come off again a
  snap later and reappear where they belong, so reading the column alone
  finds a touchdown and then an untouchdown. Win probability never notices —
  it needs the final score, which is right in every season — but expected
  points needs to know *when* the points went up, so
  `lucky_ones.points.score_events` requires `scoring_play` to agree. See
  [Reading the scores](#reading-the-scores).
- **a timeout looks exactly like a snap.** It arrives with a down, a distance,
  a yardline, a clock and a possession team, because the feed carries the
  situation around it — so no test over the columns can see it, and only
  `play_type` says it wasn't a play. With college kickoffs, which usually do
  carry a down, that was 6.7% of NFL states and 9.1% of NCAAFB ones.
  `lucky_ones.state.NOT_A_SNAP` is the filter; [the validation
  report](docs/epa-validation.md#what-it-caught) is how it was found and what
  it was costing.

One column stands apart from the rest: `text`, ESPN's sentence about the play.
Nothing either model sees is built from it — every feature in both lists is a
number or a flag — but it's the only place the *manner* of a play is recorded.
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

The shipped fits are the same command over 2006–2025: NFL brier 0.160 over 990
held-out games, NCAAFB 0.139 over 5,188.

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

The other model is a second command and a second file:

```sh
make train-ep ARGS="--league nfl --seasons 2014-2025 --root ./plays"
make epa ARGS="401671490 --season 2024 --week 3"
```

`make rates` and `make bounds` are the two that answer "why those numbers?"
for the constants — `DEFAULT_RETAINED` in [What counts, and what it's
worth](#what-counts-and-what-its-worth), `DEFAULT_CLIP` in [The
bound](#the-bound).

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
epa = MODELS.NFL.epa_per_play(game)  # how each offense actually played
MODELS.NFL.metrics.brier_score  # how the fit scored on a holdout
MODELS.NFL.expected_points_metrics.log_loss  # and the other fit
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

**Each league is two files**, `nfl.json` and `nfl-ep.json`, read lazily and
separately. A service that only draws curves never opens the second one, and
a league with a win probability fit and no expected points fit works for
everything except `epa_per_play` — which fails naming `make train-ep`, rather
than at import.

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
| `pass_defended_interception` | 0.20 | of the passes a defender is recorded as reaching, one in five is caught |
| `pass_defended_incomplete` | 0.80 | the other four — the interception that wasn't |

The pass pair is the fumble pair in a different hat. Every *contested* pass is
the denominator, exactly as every fumble is the denominator above, and the
kind records which side came up — caught or not caught, just as the fumble
kinds record recovered or not recovered.

The fumble pair is measured; the tipped one can't be. `make rates` is the
measurement:

```sh
make rates ARGS="--league ncaafb --seasons 2006-2025 --root ./plays"
```

It counts through `find_lucky_plays`, so what comes back is the rate over the
population the model actually adjusts rather than a second opinion about which
plays those are. Over the corpus in `./plays`:

| | fumbles classified | changed hands | per-season range |
| --- | --- | --- | --- |
| NFL 2008–2025 | 10,262 | **0.477** | 0.427 – 0.508 |
| NCAAFB 2006–2026 | 61,806 | **0.540** | 0.519 – 0.581 |

So the coin really is close to fair, and 0.5 is within four points of both —
closer than the leagues are to each other. The two fumble entries are a
**pair**: one coin seen from its two sides, so they have to sum to 1. 0.5/0.5
gets that for free; any measured pair has to be held to it, and would have to
be per-league.

The pass pair is measured the same way, with one wrinkle: the two leagues
write "a defender reached it" in different alphabets, and one of them writes
it unevenly. That's what the rest of this section is about, and what the
[per-game gate](#the-gate) exists for.

Note what the pair is *not* about: whether the ball was tipped. No
interception in either league records how it got there — 88 of 8,980 NFL picks
and 0 of 47,543 NCAAFB ones carry a deflection marker — so "interceptions off
a tip" isn't a population that can be selected. "A defender got to it" is, and
turned out to be the better question anyway.

`by_family` splits that denominator by what caught it, and the split is the
interesting part: **the two leagues say it in different alphabets.**

NCAAFB says it in words, unreliably. `broken_up` is its entire union — every
other phrase family is flat zero in every season — so there is no rename to
find, and the 2021–24 collapse from 8.0% to 2.2% coverage is annotation
genuinely going away:

| NCAAFB | 2018 | 2023 | 2025 |
| --- | --- | --- | --- |
| coverage | 0.080 | 0.024 | 0.058 |
| `interception_share` | 0.266 | 0.534 | 0.303 |

The NFL says it in punctuation, and steadily. Its text is gamebook text, where
a trailing parenthetical is the defender credited with the play and a bracket
is the one who hit the passer:

```
J.Garcia pass incomplete short middle to E.Graham (B.James) [S.Bowen]
                                                   ^^^^^^^^  ^^^^^^^^
                                                   defensed    QB hit
```

On a *completion* that parenthetical is the tackler, so only incompletions are
read — an incomplete pass has nobody to tackle, and the credit there is the
pass defensed. The check that settles it: touchdown completions have no
tackler either, and they carry a parenthetical 20% of the time against 82% for
every other completion.

That gives a denominator worth the name. `defensed_syntax` runs **0.093 to
0.109 of pass attempts every season from 2009 on** — a real pass-defensed
rate, moving less in seventeen years than college moves between consecutive
seasons:

| NFL | 2012 | 2019 | 2025 |
| --- | --- | --- | --- |
| coverage | 0.109 | 0.100 | 0.109 |
| `interception_share` | 0.203 | 0.190 | 0.177 |

**0.1931 combined over 2009–2025**, drifting only as slowly as the
interception rate itself. 2006–08 are missing or partial, and 2008 is the
adoption year.

### The two leagues agree

NCAAFB switched to a gamebook-shaped format in 2026 — leading clock,
formation, jersey numbers, parenthetical tacklers — but kept saying the pass
defensed in words:

```
(12:20) Shotgun #9 B.Edwards Jr. pass incomplete short left to #4 M.Humphrey
        thrown to TCU00 broken up by #1 V.Glover
```

What changed isn't the alphabet, it's the completeness. College coverage hits
**0.107 in 2026** against the NFL's 0.109.

And once the ratio is counted only over games that record both sides, every
season of both leagues agrees. That gating is not a refinement, it's the
correctness of the number: a game whose feed records none of its breakups
still records all of its interceptions, so leaving it in puts a numerator into
the ratio with nothing underneath it. In NCAAFB 2023 that's four games in five.

| | games recording | ungated | **gated share** |
| --- | --- | --- | --- |
| NCAAFB 2018 | 967 / 1,452 | 0.266 | **0.209** |
| NCAAFB 2022 | 250 / 1,401 | 0.560 | **0.186** |
| NCAAFB 2023 | 270 / 1,498 | 0.534 | **0.192** |
| NCAAFB 2026 | 68 / 72 | 0.187 | **0.179** |
| NFL 2012 | 219 / 241 | 0.204 | **0.194** |
| NFL 2025 | 234 / 252 | 0.177 | **0.170** |

Ungated, those span 0.18 to 0.56. Gated, twenty seasons and two leagues sit in
**0.170–0.209** — NFL 2009–2025 combined is 0.1931, NCAAFB 2014–2026 is
0.2069. So `P(intercepted | contested)` is about **0.20**, it's the same in
both leagues, and the instability was never in the football.

What that number is *not* is the tipped share `retained` is nominally about. A
pass defensed is any ball a defender reached, and no interception in either
league records whether one was tipped on the way. So 0.20 answers a
neighbouring question — better founded, on a stable population, and the one
worth building on if this kind ever grows into an adjustment over all
interceptions. It costs little that the shipped 0.25 is a guess, for the
reason under [Lucky WP](#the-one-thing-to-watch).

One thing the same evidence rules out: making the numerator *all*
interceptions rather than tipped ones. That population is stable — the
interception rate is the one thing here that doesn't move — but it forces the
other branch with it. An interception charged at `1 − retained` needs the
pass broken up at `retained`, and breakups run three to four times as common,
so the two sides are near equal in aggregate and the metric nets to zero in
expectation only if both are counted. Counting one is not a small leak; it is
half the effect, always pointed the same way.

They're a keyword argument everywhere they're used, so a caller who disagrees
says so at the call site:

```python
from lucky_ones.luck import DEFAULT_RETAINED, LuckKind

MODELS.NFL.luck_adjusted_game_control(
    game, retained={**DEFAULT_RETAINED, LuckKind.PASS_DEFENDED_INTERCEPTION: 0.3}
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

### The gate

The pass pair needs both its sides, and whether a game's feed records the
*breakup* side is a property of whoever typed the play-by-play. In NCAAFB it
follows the **home venue**: between 2019 and 2023 the venues doing it fell from
168 to 41 of 224 — 127 dropped it and **none** took it up. The NFL has done it
every season since 2009 and not at all before.

So `records_defended_passes` asks each game, and the pair is adjusted only
where the game can support both sides:

```python
from lucky_ones.luck import records_defended_passes

records_defended_passes(game.plays)  # True -> the pass coin is in play
```

A game that fails still gets its fumbles adjusted. It just gets a *smaller*
adjustment, which is the honest shape: the number is what the game can support,
not a constant pretending the evidence is even.

The test is a rate — 15% of a game's incompletions, over at least ten of them —
rather than presence, because a venue recording only some of its breakups would
leak in proportion to what it missed. What says the rate is enough is the ratio
inside passing games:

| | 2018 | 2022 | 2023 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- |
| NCAAFB games passing | 68% | 19% | 18% | 50% | 94% |
| breakups per interception | 3.8 | 4.4 | 4.2 | 4.6 | 4.6 |

The balance point is `(1 − retained) / retained` = **4**, and passing games sit
on it in every season — including 2022–24, where only a fifth of games pass. The
collapse took whole games, not fidelity within them, so a 2023 game that passes
is worth what a 2019 one is. In the NFL 90–95% of games pass every season from
2009.

### Three things it isn't

- **A prediction.** The adjusted curve doesn't end at 0 or 1 the way the real
  one does. The game had a winner; this is a counterfactual, and its last point
  is where the game *would* have stood.
- **Symmetric in what it can see.** A fumble the offense recovered is in the
  play text, so a team that got away with one is charged for it — both
  branches are recorded, which is what makes the fumble half of this sound. A
  tipped pass isn't: the interception is in the text and the drop mostly
  isn't. What keeps that from mattering is how rarely it comes up at all —
  tip markers appear on 88 NFL interceptions across twenty seasons, none in
  NCAAFB ever, and the NFL's annotation stopped (zero in 2025).
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

**A kind is only ever adjusted with its opposite.** That's the rule the whole
design turns on, and the reason for [the gate](#the-gate). An interception
charged `1 − retained` needs the pass that was broken up credited `retained`,
or the metric stops netting to zero in expectation and becomes a levy on
whoever the recorded side happens to fall against — every defense charged for
its picks, none credited for the balls it got a hand on. Breakups outnumber
interceptions about four to one, which is exactly `(1 − retained) / retained`,
so the two sides are the same size in aggregate: counting one is not a small
leak, it's half the effect pointed one way.

The same rule applies to the dial. If you want the pass coin out of the
number, turn off **both** of its kinds — zeroing only the interception leaves
the credit without the charge, which is the same error with the sign flipped:

```python
from lucky_ones.luck import DEFAULT_RETAINED, LuckKind

MODELS.NFL.lucky_wp(
    game,
    retained={
        **DEFAULT_RETAINED,
        LuckKind.PASS_DEFENDED_INTERCEPTION: 1.0,
        LuckKind.PASS_DEFENDED_INCOMPLETE: 1.0,
    },
)
```

Fumbles are the half that needs no gate: both branches are in every feed's
text, in every season, in both leagues. The rate is measured rather than
assumed, and which branch happened is settled by two witnesses rather than one.
That last part matters more than it sounds. `is_turnover` alone is wrong on 17%
of NFL fumbles and 34% of NCAAFB ones — nearly always a false negative on a
sack the defense stripped and recovered, and in NCAAFB before 2014 the column
is uniformly false — so the code trusts it when it says yes and checks the next
snap's offense when it says no. Getting that backwards prices the branch that
already happened.

One more approximation worth knowing, shared with the number next door: the
realized branch is read off the *next* snap, after the clock ran and any points
went up, while the counterfactual is built from the snap itself. A play's worth
of clock is small against a change of possession, but it's the floor under how
precise either number can be.

## EPA per play

The three numbers above are all about the game. This one is about the
football: **how much did each offense move the ball's value, a snap at a
time?**

```python
epa = MODELS.NFL.epa_per_play(game)
epa.home  # +0.09 -- the home offense, in expected points per snap
epa.away  # -0.04 -- the away offense, on its own snaps
epa.net  # +0.13 -- the home team's margin, which is what a table sorts on
epa.home_plays  # 68 snaps
epa.home_weight  # 51.2 of them -- what the average actually covers
```

Expected points added is the difference between what a situation was worth
before a snap and what it's worth after it, signed for the team that had the
ball:

```
EPA = (what the situation is worth now) - (what it was worth at the snap)
```

A 3-yard gain on third and 8 is negative. The punt that follows is only
slightly negative. An interception is worth about as much as a touchdown, in
the other direction. It's the one number play-by-play offers that is about a
play rather than about a game, and unlike `game_control` it doesn't care who
won — which is the whole reason to have it next to the others. A team can
control 0.70 of a game it played worse in, and that gap is most of what
there is to say about a team that keeps winning close ones.

Two things happen between the raw per-play numbers and that average, and both
are things a plain mean gets wrong rather than refinements on it: [a
bound](#the-bound) on how much any one play can contribute, and [a
weighting](#the-weighting) by how much the game was still in doubt when it
happened. Both are keyword arguments at every call site, and both are off at
their identity values:

```python
MODELS.NFL.epa_per_play(game, clip=float("inf"), weight_power=0.0)
```

is the plain unweighted mean of raw EPA, exactly — the same escape hatch
`retained=1.0` is next door, and what the tests check.

### Expected points

`lucky_ones.points` is the second fit, and it answers the older, smaller
question: **how many points is this situation worth to the team holding the
ball?** The standard definition, and it's a definition about drives rather
than about plays —

```
EP(state) = the expected value of the next score in this half,
            signed from the point of view of the team with the ball
```

— so nobody has to decide what a play is worth. The football decides and the
code counts. `MultinomialExpectedPoints` fits `P(kind | situation)` over the
seven things that can happen next (touchdown, field goal or safety, for
either side, or the half running out) and takes the expectation against
`SCORE_VALUES`. Seven probabilities and a dot product, so the scoring path is
numpy and nothing else, exactly like the logistic fit.

Reading `predict_proba` is often better than reading the number: "31% this
offense scores a touchdown, 18% the other one does" says more than the 2.1
they average to.

Being a classifier is also what keeps it honest. It can only ever return a
number between −7 and +7, and no strange situation can push it out — which a
regression on points could.

What the shipped fits say, at four snaps whose value everyone already knows:

| | 1st & 10, own 25 | 1st & 10, midfield | 1st & goal, the 2 | 3rd & 15, own 5 |
| --- | --- | --- | --- | --- |
| NFL | +1.11 | +2.43 | +5.94 | −1.99 |
| NCAAFB | +0.80 | +2.41 | +5.62 | −2.46 |

Note that the deep one is *negative*: on third and long from your own 5, the
next points in the half are more often the other team's.

Two things are deliberately not features, and they're the two you'd reach for
first.

**The score isn't** — not the margin, not the clock left in the game. A team
down 28 in the fourth throws on every down and a team up 28 kneels, so the
score genuinely predicts the next points, which is exactly why it stays out.
Putting it in would fold "this game was over" into the expected value and
leave EPA quietly measuring how far ahead you already were. Blowouts get
handled where the distortion actually is — in [the weighting](#the-weighting)
— and that's a knob a caller can turn off and look at. A coefficient isn't.

**Which team is home isn't.** Home field is worth real points over a season,
and including it would make the same snap worth more to one side than the
other, so a home offense and an away offense could post the same EPA per play
off different football.

What is in it is 13 columns: three down indicators, `log_distance`,
`goal_to_go`, a six-term linear spline in field position, the clock left in
the *half*, and one interaction between the last two. The two unobvious ones
earned their place against measurement rather than taste:

- **The field is a spline, not a polynomial.** A quadratic was the first
  thing tried, and it reads first and ten inside your own 10 as half a point
  better than it is (+0.37 against a measured −0.16) — it has one bend to
  spend, and the middle of the field outvotes both ends for it. A play from
  there is one of the biggest EPAs in football, so the end of the curve is not
  the part to approximate. Knots sit at the two 10-yard lines, the two 20s and
  midfield, which is where football changes what it's doing.
- **The clock is the half's, and it interacts with the field.** Time is what
  stands between a drive and points, and it runs out at the half, not at the
  end of the game. But a minute left isn't worth the same everywhere: a team
  on its own 10 has nothing, a team on the opponent's 15 still has a field
  goal. A clock level with no interaction has to split the difference, and it
  misses both ends the same way — a couple of tenths too generous deep in your
  own end with under a minute left, a couple of tenths too stingy in field
  goal range. The interaction roughly halves both.

```
Fitting on 2433 games (358616 snaps), holding out 609 games
  defense_field_goal       9.7% of snaps
  ...
  offense_touchdown       34.9% of snaps
  1st and 10, own 25     +1.11 points
  1st and 10, midfield   +2.43 points
  1st and goal, the 2    +5.94 points
  3rd and 15, own 5      -1.99 points
Holdout: log loss 1.3189, mean absolute error 3.75 points over 609 games
```

`log(7)` = 1.95 is what a guess at the base rates gets, so 1.32 (NFL) and 1.26
(NCAAFB) are fits that learned the situation. Both are held out by game, and
[the validation report](docs/epa-validation.md) does the same thing more
carefully — refitting on 2014–2022 and measuring on seasons the fit has never
seen. The mean absolute error is large
by construction — the next score is 7 or 0 or −3 and the fit says 2.1 — and its
size is exactly why EPA is a per-play *average* rather than a per-play claim.

The residual worth knowing: first and ten between your own 30 and 39 is
over-priced by about half a point in both leagues, at every point on the clock.
Nearly every snap there is a fresh drive after a kickoff, while a first down at
your own 45 is one the offense earned on the way — so the truth is flat across
that stretch and then jumps. Fixing it means telling the model how the ball got
there, which is the same category of thing as the score, and it stays out for
the same reason.

### Reading the scores

The label needs to know *when* the points went up, and that turns out to be
the hard part. Before 2014, both leagues' score columns jitter: points land on
a play, come off a snap later, and reappear where they belong. Counting the
scoring the columns claim against the final score is the check:

| points per game | the columns alone | with `scoring_play` agreeing | the actual final scores |
| --- | --- | --- | --- |
| NFL 2013 | 56.6 | **42.0** | 46.8 |
| NFL 2024 | 46.3 | **46.0** | 45.8 |
| NCAAFB 2010 | 67.0 | **49.2** | 52.3 |
| NCAAFB 2018 | 58.9 | **56.7** | 56.9 |

So `score_events` takes two witnesses — the column went up, *and* the feed
flagged the play — which is the same shape of fix `luck._changed_possession`
makes for `is_turnover`, and it errs the same way on purpose. What's left
before 2014 is a false negative: about one score in ten where the flag and the
jump landed on different rows, and a score this can't find just leaves its
snaps labelled with whatever scored next. That's a quieter error than
inventing a touchdown, which is what the alternative does.

Ten percent is still too much to fit on, so **the shipped expected points fits
start at 2014** while the win probability fits use all twenty seasons. The two
need different things from the same data: a game's winner is right in every
season, and the minute it was scored in isn't. 2014 is also where NCAAFB's
`is_turnover` column stops being uniformly false — the play-by-play got better
in both leagues at once.

### The bound

EPA has fat tails. Most snaps are worth a fraction of a point and a pick-six
is worth eleven, so a game's mean is largely a report of whether one enormous
play happened. `clip` bounds each play's contribution at ±`DEFAULT_CLIP`,
which is **5.0**.

`make bounds` is the measurement, and it runs through `play_epa`, so what
comes back is the distribution the metric averages rather than a second
opinion about it:

```sh
make bounds ARGS="--league nfl --seasons 2014-2025 --root ./plays"
```

| percentile of absolute EPA | **p99** | across the seasons measured |
| --- | --- | --- |
| NFL 2014–2025 | **5.03 – 5.38** | 1.02% – 1.41% of plays beyond ±5 |
| NCAAFB 2014–2025 | **5.24 – 5.42** | 1.19% – 1.36% beyond |

The 99th percentile is 5.03 to 5.42 across all twenty-four league-seasons from
2014 on, so 5.0 bounds about one snap in a hundred and leaves the other
ninety-nine exactly alone. The reason to believe that number isn't the percentile, though — it's
that the football agrees with it. Ask the fit what the biggest ordinary snaps
are worth:

| the biggest things one snap does | NFL |
| --- | --- |
| 60-yard touchdown from your own 40 | **+5.07** |
| interception at your own 25, returned to the spot | **−5.12** |
| the same interception returned for a touchdown | −8.11 |

The first two are the ceiling of what a team does on purpose, they happen a
few times a game, and a metric that flinches at them is measuring something
else. The clip sits right on them — within a tenth of a point — so what it
actually truncates is the third row: the compound play, where the ball was
turned over *and* returned for a score, which is two events the box score
charges to one snap.

That's also why the first guess was wrong. A clip at 4 is the 97.5th
percentile, and it bites about one snap in forty — which means it truncates
ordinary deep interceptions and ordinary long touchdowns, and reports a team
that threw one as a team that threw something smaller.

Why bound at all, given the tail is real football: a game is only ~130 snaps.
At its full −11 a red-zone pick-six moves a team's average by 0.08, which is
most of the gap between a good offense and a bad one, off one play. Bounded it
moves it by 0.04. The play still dominates the game; it just doesn't get to be
three plays.

### The weighting

A team up 28 in the fourth runs into the line and kneels; a team down 28
throws deep on every down against a defense that has stopped covering. Both
post EPA per play that measures the score rather than the team.

Each play is weighted by how much its game was still in doubt:

```
weight = 4 * p * (1 - p)
```

on the win probability at that snap — which is the variance of the game's
outcome at that moment, scaled to peak at 1. A coin-flip game weighs 1.00, a
three-score game 0.36, a decided one 0.04.

That's chosen over the usual garbage-time *filter* — drop everything outside
0.05 to 0.95 — because a filter puts a cliff in the middle of the fourth
quarter, and two teams whose games hovered on either side of it get
incomparable numbers off comparable football. This fades instead, and it fades
on a quantity that already means something rather than on a curve someone
shaped. It also disposes of the kneel-downs and the clock-killing runs without
naming them: they happen above p = 0.97, where they weigh about nothing.
There's no list of play types in this module and there doesn't need to be.

`home_weight` and `away_weight` report what survived, which is the honest part
of the number in the way `GameControl.seconds` is:

Over every season of both leagues from 2014 (`make bounds` again):

| | effective play share | median shift in a team's game number | p95 |
| --- | --- | --- | --- |
| NFL | 0.64 – 0.68 | 0.05 – 0.07 | 0.22 – 0.28 |
| NCAAFB | 0.53 – 0.55 | 0.06 – 0.07 | 0.31 – 0.37 |

Two readings. A season of NFL snaps is worth about two thirds of its count
once the decided stretches are down-weighted, and a season of college snaps
barely more than half — college has more blowouts, and the metric says so.
And the shift is not ceremony: the bound and the weighting together move a
team's game number by about 0.06 typically and 0.3 at the tail, against a
spread between good and bad offenses of a few tenths.

`weight_power` is the exponent, and there's no measurement that settles it the
way the clip is settled — it's a statement about what you want the number to
mean. 1.0 is the unexaggerated choice: the weight *is* the outcome's variance,
with nothing done to it. 0.0 turns the weighting off.

### Adding games up

**A weighted sum, not a mean of means.** A game contributes
`home * home_weight` to the numerator and `home_weight` to the denominator:

```python
numerator = sum(epa.home * epa.home_weight for epa in games)
denominator = sum(epa.home_weight for epa in games)
season = numerator / denominator
```

Averaging the per-game numbers instead lets a 12-snap weather-shortened game
count as much as a full one, and lets a blowout — whose weight is almost all
gone — count as much as a game that was live throughout. That's what the
weights are on the tuple for.

### Is it any good?

`make validate` refits both models on 2014–2022 and measures them on
2023–2025, which neither fit has seen. The full report with the charts is
[docs/epa-validation.md](docs/epa-validation.md); the headline:

| out of sample | NFL | NCAAFB |
| --- | --- | --- |
| held-out snaps | 109,525 | 715,182 |
| log loss (base rates 1.53 / 1.49) | **1.318** | **1.279** |
| skill over base rates | 13.7% | 14.1% |
| worst calibration bin, in points | 0.33 | 0.35 |
| mean calibration gap | 0.097 | 0.096 |

And the metric built on it, over 824 team-seasons whose games are dealt at
random into two sets 1,500 times over:

| | reliability | predicts scoring | predicts live scoring |
| --- | --- | --- | --- |
| EPA/play (NFL) | **0.589** | **0.549** | **0.471** |
| points/play (NFL) | 0.551 | 0.539 | 0.446 |
| EPA/play (NCAAFB) | **0.511** | **0.470** | **0.249** |
| points/play (NCAAFB) | 0.468 | 0.466 | 0.242 |

EPA beats the box score in six of six cells, though none of those margins
clears significance on its own.

**And what the weighting costs.** A plain sweep of `weight_power` makes it look
bad, but that sweep can't be read: down-weighting a blowout is nearly the same
as dropping it, so the weighted number works from a different pool of games
than the unweighted one. `make weighting` holds the pool still — comparing the
weighted number against the unweighted one *thinned to the same effective
sample* (Kish's `n_eff`), so only the choice of snaps differs:

| at power 1 | NFL | NCAAFB |
| --- | --- | --- |
| effective sample kept | 81% | 70% |
| what that precision costs, in reliability | −0.031 | −0.072 |
| what the weighting's snap choice buys back | +0.026 | +0.004 |

**The whole penalty is the sample.** The snaps the weighting picks are, if
anything, marginally better than a random draw of the same size — positive in
six of six cells, significant in none. So the weighting costs precision and
nothing else.

**And what it buys.** Everything above is predictive; the complaint the
weighting exists for is descriptive — *that 0.31 wasn't real, it was a rout*.
So take the unweighted mean over only the live snaps (0.2–0.8 WP) as what the
team did while it mattered, and see how close each whole-game number lands:

| | gap to the live-only mean | closed | sample kept |
| --- | --- | --- | --- |
| power 0 — no weighting | 0.113 | — | 100% |
| **power 1 — shipped** | **0.065** | **43%** | **86%** |
| power 2 — best | 0.053 | 53% | 76% |
| power 5 | 0.081 | 28% | 60% |

Garbage time is worth about **0.11** to a team's game number (0.26 at the 90th
percentile), against a good-to-bad offense spread of a few tenths — so the
problem is real. The shipped power removes 43% of it. NFL and college trace
almost the same curve.

The curve **bottoms at power 2 and climbs again**, which is what makes this a
test rather than arithmetic: `4p(1−p)` fades inside the live window too, so a
high enough power averages the coin-flip snaps rather than the live game. So
`DEFAULT_CLIP` is measured, and `DEFAULT_WEIGHT_POWER = 1.0` is the
conservative end of a flat stretch — with power 2 a defensible alternative at
roughly one more point of sample per point of gap closed, and anything past 3
strictly worse.

### Three things this one isn't

- **A rating.** A game is ~130 snaps, and 130 snaps is a noisy estimate of
  anything. The `home_weight` on the tuple is there so a reader can see how
  much of a game the number covers.
- **Symmetric with `GameControl`.** `home` and `away` don't sum to anything.
  They're two separate averages over two disjoint sets of snaps, in points,
  and both are positive in a game where everybody moved the ball. `None` means
  there was nothing to average, not 0.0 — which would read as a team that
  played exactly to expectation.
- **Related to the luck adjustment.** `lucky_wp` and `epa_per_play` both have
  opinions about a fumble, and they are different opinions about different
  things: one prices the bounce in win probability and splits it, the other
  prices the change of possession in points and charges all of it. Nothing in
  `lucky_ones.epa` reads `lucky_ones.luck`, and neither fit is retrained by
  any of it.

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
make train    # fit win probability, rewrite releases/{league}.json
make train-ep # fit expected points, rewrite releases/{league}-ep.json
make curve    # one game's curve and its game control, as JSON
make epa      # one game's EPA per play, and every snap behind it
make rates    # the fumble and pass rates behind DEFAULT_RETAINED
make bounds   # the EPA distribution behind DEFAULT_CLIP
make validate # are the models any good? -- docs/epa-validation.md
make weighting # what the WP weighting costs, at a fixed effective sample
make plays    # aws s3 sync the processed play-by-play down for offline fits
```

Machine-local settings go in `.devcontainer/local.env` (gitignored;
`local.env.example` is the tracked reference) — `AWS_PROFILE`, mainly, since
the plays live in a bucket.

[endgame]: https://github.com/NathanDeMaria/endgame
