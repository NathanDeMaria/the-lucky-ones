# Is the expected points fit any good?

Everything here is produced by `make validate`, which refits both models on
**2014–2022** and measures them on **2023–2025** — seasons neither fit has
seen. Numbers quoted off a shipped release would be numbers about games that
release was fitted on, so none are.

```sh
make validate ARGS="--league nfl --root ./plays"
make validate ARGS="--league ncaafb --root ./plays"
```

The JSON it writes (`validation-{league}.json`) is the source for every figure
below, and for the charts, which `make validate` redraws from it. Nothing on
this page was typed by hand.

The short version: the fit has real skill, it is well calibrated, it prices
the field the way football does, and the metric built on it beats the box
score. One of the two knobs is supported by the measurement and the other one
isn't — and the most valuable thing this validation did was find a bug.

---

## What it caught

Validation that finds nothing wasn't looking. This found something worth the
exercise on its own.

`iter_states` promises "one `GameState` per scrimmage play" and filters on the
columns: no down, no clock, no possession team, not a snap. **Timeouts pass
every one of those tests.** They arrive with a down, a distance, a yardline, a
clock and a team, because the feed carries the situation of the snap around
them — or a placeholder first-and-ten at the 35. In the NFL's 2023 feed,
timeouts were **60% of every "snap"** the model saw between the offense's own
33 and 37 with more than ten minutes left in the half.

Over 2014–2025, every row that passes the column tests:

| admitted as a "snap" | NFL | NCAAFB |
| --- | --- | --- |
| `Timeout`, `Official Timeout` | 26,328 | 106,285 |
| `Two-minute warning` | 5,374 | — |
| `Kickoff`, `Kickoff Return`, `Kickoff Return Touchdown` | 14 | 158,526 |
| `End Period`, `End of Game`, `End of Regulation`, `Coin Toss` | 724 | 1,594 |
| **total** | **32,440** | **266,405** |
| **share of everything `iter_states` yielded** | **6.7%** | **9.1%** |

College kickoffs are the other half of it: `iter_states` skipped kickoffs
because they have no down, and in NCAAFB they usually *do*.

This was not harmless padding. It broke the two models in different ways:

- **Expected points, badly.** A timeout on the kickoff after a score is priced
  as first and ten at your own 35 with the possession stale, so it is labelled
  with a next score belonging to the other team. Measured on held-out seasons,
  the fit was **1.58 points wrong** across the whole 33-to-37 slice of the
  field in both 2023 and 2024, and the worst calibration bin missed by 0.79.
- **EPA, more simply.** A timeout is not a play and has no business in a
  per-play denominator.

`state.NOT_A_SNAP` is the fix, and it is a `play_type` test because nothing
else can see these rows. After it, the worst calibration bin misses by **0.33**
and the field is smooth. The college expected points fit moved substantially —
first and ten on your own 25 went from **+0.43 to +0.80** — because 5.4% of that
league's states had been kickoffs, sitting at the 25 and the 35.

All four shipped releases are retrained on the corrected states.

---

## Does it know anything?

Two baselines are worth beating. **Uniform** is `log(7) = 1.946`, the number
everyone knows. **Base rates** is the honest one: how often each of the seven
outcomes happens, ignoring the situation entirely. A model that can't beat
that has learned nothing about football.

| | NFL | NCAAFB |
| --- | --- | --- |
| held-out snaps | 109,525 | 715,182 |
| log loss | **1.318** | **1.279** |
| …base rates | 1.527 | 1.490 |
| …uniform | 1.946 | 1.946 |
| **skill over base rates** | **13.7%** | **14.1%** |
| mean absolute error | **3.64** | **4.21** |
| …base rates | 4.06 | 4.70 |
| **better than base rates** | **10.3%** | **10.4%** |
| bias | −0.020 | −0.049 |

The mean absolute error is large by construction and it is worth saying why
rather than apologising for it: the thing being predicted is 7 or 0 or −3, and
the fit says 2.1. Most of that error is the irreducible spread of a discrete
outcome, not the model missing. It is also exactly why EPA is a per-play
*average* and never a per-play claim.

Variance explained is 0.11 in both leagues, and for the same reason it is the
wrong lens. Calibration is the right one.

---

## Is it right, or only confident?

Held-out snaps in twenty equal-count bins by what the fit predicted, against
what actually happened next.

![NFL calibration](calibration-nfl.svg)

![NCAAFB calibration](calibration-ncaafb.svg)

| | NFL | NCAAFB |
| --- | --- | --- |
| worst bin | 0.33 | 0.35 |
| mean absolute gap | 0.097 | 0.096 |

A tenth of a point across the whole range, on seasons neither fit saw. The one
visible outlier in both leagues is the bottom bin — the most negative
situations, deep in your own end on late downs — where the fit is about a third
of a point too pessimistic. That is the sparsest part of the field and the
hardest to fit, and it is the direction that matters least: it makes a team's
worst snaps look slightly worse than they were.

---

## Is it football?

The check that needs no statistics. Lines are the fit across the field for each
down; dots are the observed mean at the same spots on held-out seasons, over
buckets of 200 snaps or more.

![NFL expected points surface](surface-nfl.svg)

![NCAAFB expected points surface](surface-ncaafb.svg)

And at situations whose value is common knowledge:

| | NFL | NCAAFB |
| --- | --- | --- |
| 1st and 10, own 1 | −0.18 | −0.31 |
| 1st and 10, own 25 | +1.11 | +0.86 |
| 1st and 10, midfield | +2.43 | +2.41 |
| 1st and 10, opponent 25 | +4.01 | +3.87 |
| 1st and goal, the 2 | +5.95 | +5.45 |
| 3rd and 15, own 5 | −2.05 | −2.40 |
| 4th and 1, midfield | +0.01 | −0.16 |

These land on top of every published expected points model, which is the point:
the fit was not tuned towards them and it arrives there anyway. The negative
ones are the ones worth reading twice — on first and ten from your own 1, the
next points in the half really are more often the other team's.

**The residual worth knowing.** First and ten between your own 30 and 39 is
over-priced by about half a point in both leagues, at every point on the clock.
Nearly every snap there is a fresh drive after a kickoff, while a first down at
your own 45 is one the offense earned on the way — so the truth is flat across
that stretch and then jumps, and a model that can't see how the ball got there
splits the difference. Fixing it means giving the fit the drive's history,
which is the same category of thing as the score, and it stays out for the
same reason.

---

## The bound

`DEFAULT_CLIP = 5.0`, and the distribution it is a statement about:

![NFL EPA distribution](distribution-nfl.svg)

| held-out 2023–2025 | NFL | NCAAFB |
| --- | --- | --- |
| 99th percentile of \|EPA\| | 5.20 | 5.24 |
| share of plays beyond ±5 | 1.21% | 1.19% |

One play in a hundred, in both leagues, on seasons neither fit saw — and 5.03
to 5.42 across every league-season from 2014 when `make bounds` measures the
lot. The football argument is in [the README](../README.md#the-bound) and the
split-half evidence is below; both point the same way.

---

## The weighting

`weight = 4p(1−p)` on the win probability at the snap — the variance of the
game's outcome, scaled to peak at 1.

![The weighting curve](weighting-nfl.svg)

| effective play share at power 1 | NFL | NCAAFB |
| --- | --- | --- |
| | 0.67 | 0.55 |

College keeps barely half its snaps' weight, against the NFL's two thirds.
College has more blowouts and the metric says so — and that difference is
exactly what makes the next section unable to settle this knob.

---

## Does the metric measure the team?

Each team-season's games are dealt **at random** into two sets, the number is
computed over each, and the two are correlated across teams. Then it is done
again — 1,500 times, with a fresh deal and a fresh bootstrap resample of the
team-seasons each time — so the intervals cover both which games landed
together and which teams were in the sample.

Randomly, not by alternating games, and that matters: schedules tend to
alternate home and away, so every-other-game can put most of a team's home
games in one set and its away games in the other. That would have shown up as
the metric being unreliable when it was really the design.

Three targets, because they disagree, and the disagreement is the finding:

| target | what it asks | who it favours |
| --- | --- | --- |
| **reliability** | does one set's number match the other's? | sample size — the weighting can only lose |
| **all scoring** | does it predict the other set's points per play? | the unweighted number: garbage-time points are in the target |
| **live scoring** | …counting only games that stayed in doubt? | the weighting — this is what it's for |

![NFL split-half](split-half-nfl.svg)

![NCAAFB split-half](split-half-ncaafb.svg)

### EPA beats the box score

| NCAAFB, 733 team-seasons | reliability | all scoring | live scoring |
| --- | --- | --- | --- |
| EPA/play, shipped | **0.511** | **0.470** | **0.249** |
| EPA/play, unadjusted | 0.567 | 0.489 | 0.261 |
| points per play | 0.468 | 0.466 | 0.242 |

| NFL, 91 team-seasons | reliability | all scoring | live scoring |
| --- | --- | --- | --- |
| EPA/play, shipped | **0.589** | **0.549** | **0.471** |
| EPA/play, unadjusted | 0.617 | 0.555 | 0.483 |
| points per play | 0.551 | 0.539 | 0.446 |

EPA beats points per play on all three targets in both leagues. This is the
comparison the design is cleanest for: both numbers are computed over the same
games and the same deal, so whatever the schedule does, it does to both. The
margins are small in absolute terms (+0.04 on reliability, P = 0.69 in the NFL
and 0.90 in college) and none clears a conventional significance bar on its
own — but the direction is the same in six of six cells, which is the thing to
read.

### The bound earns its place

Against the unadjusted number, in college, where there are enough team-seasons
to resolve a small effect:

| clip (weighting off) | reliability | all scoring | live scoring |
| --- | --- | --- | --- |
| 3.0 | +0.032 (P 1.00) | +0.008 (P 0.94) | +0.002 (P 0.67) |
| **5.0 (shipped)** | **+0.014 (P 1.00)** | **+0.005 (P 0.97)** | **+0.002 (P 0.75)** |

Small, consistent, and positive on every target. A tighter clip scores better
on reliability, but reliability rewards throwing away variance with no natural
stopping point — and the predictive targets flatten out between 2.5 and 5, with
live scoring falling away below 2. So the measurement puts the useful range at
roughly 3 to 5 and cannot separate values inside it; 5.0 is chosen from the
football, where it sits on the ordinary ceiling of what one snap does.

### The weighting is a choice, not a fitted value

| power (clip 5) | reliability | all scoring | live scoring |
| --- | --- | --- | --- |
| 0.25 | +0.003 (P 0.61) | +0.006 (P 0.83) | +0.001 (P 0.54) |
| **1.0 (shipped)** | **−0.056 (P 0.00)** | **−0.020 (P 0.04)** | **−0.013 (P 0.17)** |
| 2.0 | −0.103 (P 0.00) | −0.043 (P 0.00) | −0.025 (P 0.08) |

Read at face value this says the weighting costs accuracy, monotonically in
the power, on every target including the one built to favour it. In the NFL
nothing is significant in either direction (power 1: −0.028 / −0.005 / −0.012,
P ≈ 0.3).

**But this test cannot settle the weighting, and it is important to say so
rather than bank the result.** Down-weighting discards blowouts, and blowouts
are the mismatch games — so the weighted metric's two sets are built from a
smaller and differently-selected pool of games than the unweighted one. The
noise is not common to the two variants, and the effect is worst in exactly
the league that has the statistical power: college schedules are wildly uneven,
and college is where 733 of the 824 team-seasons are. The comparison isn't
clean, and the NFL, whose schedule is balanced, has too few team-seasons to
adjudicate a 0.03 difference.

So `DEFAULT_WEIGHT_POWER = 1.0` stays, and what this section establishes is its
*price*, not its verdict: at power 1 you are working from 55% of a college
season's snaps and 67% of an NFL one, and you should expect the number to be
somewhat noisier than the unweighted one in exchange for describing
competitive football by construction. Turning it off is
`weight_power=0.0` at any call site.

There is a test that would settle it, and it isn't this one: hold the *games*
fixed across variants — score every variant on the same competitive subset —
so that the weighting changes only how snaps are combined and not which games
are in the sample. That is worth building before anyone changes the default in
either direction.

---

## What isn't tested here

- **The win probability model**, except as an input to the weighting. Its own
  holdout scores are in its release and in the README.
- **The luck adjustment.** `lucky_wp` and `luck_adjusted_game_control` are
  measured by `make rates`, on different evidence.
- **Overtime.** Both models are regulation-only, matching `game_control`.
- **The counterfactual in `lucky_ones.luck`**, which shares no code with any
  of this.
- **Anything before 2014**, where the score columns can't place a tenth of the
  scoring. See `points.score_events` and `points.FIRST_LEGIBLE_SEASON`.
