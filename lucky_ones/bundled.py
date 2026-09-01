"""
The fits that ship inside the package, and the `MODELS` that reaches them.

    from lucky_ones import MODELS

    points = MODELS.NCAAFB.curve(game)
    control = MODELS.NCAAFB.game_control(game)

The releases in `lucky_ones/releases/` are data files inside the wheel, not
something a consumer fetches. That is a deliberate trade: a fit is ~1.3KB of
coefficients, and pinning the package by rev now pins the model with it, so
"which fit is invisible-string drawing" has one answer -- the rev -- instead
of depending on what happened to be at an S3 key when the process started.
The cost is that a retrain is a commit and a release rather than a file copy.
`train.py` writes into that directory by default, so the retrain loop is
`make train` and then `git add`.

`BundledModel` wraps the release rather than being it, so the thing you reach
for by name is the thing you want to call. `MODELS.NFL.predict(states)`
satisfies `WinProbabilityModel`, so a bundled fit can go anywhere a fit can.
Everything else -- the coefficients, the seasons behind them, the holdout
score -- is on `.release` and proxied here where it answers a question a
caller actually asks ("how good is this?", "was my game in the training
set?").

Loading is lazy and cached. Reading two JSON files at import is not the cost
worth avoiding; validating them is pydantic's, and a process that only ever
touches one league shouldn't pay for the other.
"""

from dataclasses import dataclass, fields
from functools import cached_property
from importlib.resources import files
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .curve import CurvePoint, GameControl, curve_from_states, game_control
from .game import GamePlays
from .model import LogisticWinProbability
from .release import Metrics, TrainedOn, WinProbabilityRelease
from .state import GameState, iter_states

# Where the shipped fits live, as a filesystem path, for the one caller that
# writes them: `train.py`. Reading goes through `importlib.resources` below
# instead, which also works when the package isn't a directory on disk.
RELEASE_DIR = Path(__file__).parent / "releases"


class BundledModel:
    """
    One league's shipped fit: the release, and the calls you make with it.

    A `WinProbabilityModel` in its own right -- `predict` is the protocol's
    method, with the release's coefficients behind it.
    """

    def __init__(self, league: str) -> None:
        self.league = league

    def __repr__(self) -> str:
        # Deliberately doesn't touch `.release`: a repr that reads two files
        # off disk makes a debugger session mysteriously slow, and makes an
        # exception traceback fail while it's formatting itself.
        return f"{type(self).__name__}({self.league!r})"

    @cached_property
    def release(self) -> WinProbabilityRelease:
        """
        The parsed release, read from the package's own data on first use.

        Raises `FileNotFoundError` if the league's file isn't in the install,
        which means either a build that dropped the data files (see the
        wheel's `artifacts` in pyproject) or a league nobody has fit yet.
        """
        resource = files(__package__).joinpath("releases", f"{self.league}.json")
        if not resource.is_file():
            raise FileNotFoundError(
                f"No bundled fit for {self.league!r}. Fit one with "
                f"`make train ARGS='--league {self.league} ...'`, which writes "
                f"into {RELEASE_DIR}, and commit it."
            )
        return WinProbabilityRelease.model_validate_json(resource.read_text())

    @cached_property
    def model(self) -> LogisticWinProbability:
        """
        The fit itself, for a caller that wants the coefficients.

        Raises `ValueError` through `to_model` if the shipped release was
        written against different features than this build produces -- which
        can only happen if someone changed `FEATURE_NAMES` without retraining,
        and is exactly when you want the loud failure. `bundled_test` asserts
        it for every shipped league so that break lands in CI, not in a
        consumer.
        """
        return self.release.to_model()

    # The release's own fields, forwarded because they answer the questions a
    # caller has at the point of use -- "is this fit any good", "did it see
    # this season" -- and `MODELS.NFL.metrics` reads better than
    # `MODELS.NFL.release.metrics` for something asked that often.
    @property
    def metrics(self) -> Metrics:
        """How the fit scored on games it wasn't fit on. Lower is better."""
        return self.release.metrics

    @property
    def trained_on(self) -> TrainedOn:
        """The seasons, weeks and game count behind the coefficients."""
        return self.release.trained_on

    @property
    def run_id(self) -> str:
        """The training run that produced this fit."""
        return self.release.run_id

    def predict(self, states: Sequence[GameState]) -> np.ndarray:
        """P(home team wins), one per state. See `WinProbabilityModel`."""
        return self.model.predict(states)

    def curve(self, game: GamePlays) -> list[CurvePoint]:
        """The game's win probability at every snap -- what a chart draws."""
        return curve_from_states(self.model, list(iter_states(game)))

    def curve_from_states(self, states: Sequence[GameState]) -> list[CurvePoint]:
        """
        The same curve for states a caller already walked the game to build.

        The one a backend serving a game calls: `curve` walks it again, and
        for a service that has the states in hand that is double the work.
        """
        return curve_from_states(self.model, states)

    def game_control(self, game: GamePlays) -> GameControl | None:
        """
        The share of `game` each team spent winning it, time-weighted.

        None for a game with no elapsed regulation clock to weight by. See
        `lucky_ones.curve.game_control`.
        """
        return game_control(self.curve(game))


@dataclass(frozen=True)
class BundledModels:
    """
    Every fit in the package, by league.

    Attributes rather than a dict lookup so that `MODELS.NCAAFB` is a name a
    type checker and an editor both know; `MODELS["ncaafb"]` is here too for
    the caller holding a league string out of a request. A new league is a
    field here plus its JSON -- `bundled_test` fails if the two disagree, so
    a fit that ships without being reachable can't happen quietly.
    """

    NFL: BundledModel
    NCAAFB: BundledModel

    def __getitem__(self, league: str) -> BundledModel:
        """The fit for a league name, case-insensitively. KeyError if none."""
        for model in self:
            if model.league == league.lower():
                return model
        raise KeyError(
            f"No bundled fit for {league!r}. Have: "
            f"{', '.join(model.league for model in self)}"
        )

    def __iter__(self) -> Iterator[BundledModel]:
        return (getattr(self, field.name) for field in fields(self))

    def __len__(self) -> int:
        return len(fields(self))


MODELS = BundledModels(NFL=BundledModel("nfl"), NCAAFB=BundledModel("ncaafb"))
