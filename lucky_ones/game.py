"""
Grouping plays into games, and working out which team was at home.

The second half of that is the awkward one. endgame's play schema carries
`home_score`, `away_score` and `offense_team_id`, but nothing that says which
team id the home score belongs to -- and a win probability model is useless
without it, because the sign of the score margin is the model's single
strongest feature.

It can't be joined in from endgame's `Game` either: `Game.home` is a cleaned
display name ("Ohio State"), while the plays carry ESPN's numeric team ids
("194"). Nothing in the two datasets connects the two, so the mapping is
either inferred from the plays themselves or supplied by the caller. This
module does the first and lets you do the second.
"""

from collections import Counter
from logging import getLogger
from typing import Iterable, Mapping, NamedTuple, Sequence

from .plays import Play

logger = getLogger(__name__)


class GamePlays(NamedTuple):
    """
    One game's plays, with the two things the plays don't say themselves:
    which side is home, and which is away.
    """

    game_id: str
    league: str
    season: int
    week: int
    home_team_id: str
    away_team_id: str
    plays: Sequence[Play]


class _Votes(NamedTuple):
    """How often a team's own drives moved each side of the scoreboard."""

    home: int = 0
    away: int = 0


def infer_home_team_id(plays: Sequence[Play]) -> str | None:
    """
    Which of the two team ids in `plays` the `home_score` column belongs to,
    or None if the plays don't say.

    Works off scoring drives. Scores are cumulative *after* a play, so a play
    where `home_score` went up and `away_score` didn't is a play that scored
    for the home side; the team that had the ball on it is usually the team
    that scored, and that's one vote for "this team is home".

    "Usually", because a pick-six or a safety scores for the defense -- so
    this is a vote rather than a single reading. Both assignments are scored
    against every vote in the game and the better one wins, which a handful
    of defensive scores can't flip: a game has one or two of them against a
    typical eight to twelve scoring drives.

    None when there's nothing to count -- a scoreless game, a game where
    every play is missing its scores, or one where the two assignments come
    out equal. That last one is genuinely ambiguous rather than unlikely-but-
    fine, and a coin flip here would silently sign-flip every feature in the
    game.
    """
    votes: dict[str, _Votes] = {}
    previous_home, previous_away = 0, 0
    for play in plays:
        home, away = play.home_score, play.away_score
        if home is None or away is None:
            continue
        home_gained, away_gained = home > previous_home, away > previous_away
        previous_home, previous_away = home, away
        # A play that moved both sides isn't a scoring play, it's a gap in
        # the data -- the plays either side of a missing one look like that.
        if home_gained == away_gained:
            continue
        team = play.drive_team_id or play.offense_team_id
        if team is None:
            continue
        current = votes.get(team, _Votes())
        votes[team] = _Votes(
            home=current.home + home_gained, away=current.away + away_gained
        )

    teams = _two_team_ids(plays)
    if teams is None or not votes:
        return None
    first, second = teams
    first_votes, second_votes = votes.get(first, _Votes()), votes.get(second, _Votes())
    # How well each assignment explains the votes: if `first` is home, then
    # its drives should have moved the home score and `second`'s the away one.
    first_is_home = first_votes.home + second_votes.away
    second_is_home = second_votes.home + first_votes.away
    if first_is_home == second_is_home:
        logger.warning(
            "Couldn't tell which of %s / %s was home in %s; the scoring "
            "drives split evenly",
            first,
            second,
            plays[0].game_id if plays else "?",
        )
        return None
    return first if first_is_home > second_is_home else second


def _two_team_ids(plays: Iterable[Play]) -> tuple[str, str] | None:
    """
    The game's two team ids, most-used first.

    Counted over possession rather than taken as a set: a stray id from a
    malformed play would otherwise turn a normal game into "not two teams"
    and drop it. The two that actually ran plays win.
    """
    counts = Counter(
        play.offense_team_id for play in plays if play.offense_team_id is not None
    )
    if len(counts) < 2:
        return None
    (first, _), (second, _) = counts.most_common(2)
    return first, second


def group_by_game(
    plays: Iterable[Play],
    home_team_ids: Mapping[str, str] | None = None,
) -> list[GamePlays]:
    """
    Plays, split into games and each labelled with its home side.

    `home_team_ids` maps game id to home team id and wins wherever it has an
    entry; anything it doesn't cover falls back to `infer_home_team_id`. Pass
    it when you have the mapping from somewhere authoritative -- inference is
    the fallback, not the design.

    A game whose home side can't be determined is dropped with a warning
    rather than guessed at. Half a training set with its score margins
    sign-flipped is worse than a smaller one.

    Games come out in `game_id` order, and each game's plays stay in the
    order they were given -- which `PlaySource` promises is play order.
    """
    by_game: dict[str, list[Play]] = {}
    for play in plays:
        by_game.setdefault(play.game_id, []).append(play)

    games = []
    for game_id, game_plays in sorted(by_game.items()):
        home_team_id = (home_team_ids or {}).get(game_id) or infer_home_team_id(
            game_plays
        )
        if home_team_id is None:
            logger.warning("Dropping %s: couldn't tell which team was home", game_id)
            continue
        teams = _two_team_ids(game_plays)
        if teams is None or home_team_id not in teams:
            logger.warning(
                "Dropping %s: %s isn't one of the teams that ran a play",
                game_id,
                home_team_id,
            )
            continue
        away_team_id = teams[1] if teams[0] == home_team_id else teams[0]
        first = game_plays[0]
        games.append(
            GamePlays(
                game_id=game_id,
                league=first.league,
                season=first.season,
                week=first.week,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                plays=game_plays,
            )
        )
    return games
