"""Line the stat book up with the film: every play Garrett logged courtside gets a time in the film.

The stat book (tracker/stat-book.html) knows *what* happened: who shot, made or missed, from where, and the
rebounds, assists, steals, turnovers and fouls around it, in the order he tapped them. The film knows *when*
shots happened: each card in events.json is a ball arriving at a rim at a film time. Shots are what both
sides share, so they are the anchors:

1. Split both into periods: the stat book's own period marks, the film's by its scoreboard period digit.
2. Within a period, align the logged shots to the film's cards in order (a sequence alignment: a logged shot
   may have no card, a card may be no logged shot). A match earns more when the made/missed call agrees,
   when the card is near where the logged time says it should be, and when the shot went at the basket that
   team is attacking that period.
3. Fit logged time -> film time through the matched shots and align again, tighter. Logged live, the phone's
   clock runs with the film and the fit is nearly a straight line. Logged while rewatching film, with pauses
   and skips, it is not, and the order of shots does most of the work: that is why time is a soft term.
4. Every other play takes its film time from its shot (a rebound just after the miss, an assist or block at
   the shot) or, failing that, from the fit between the matched shots either side of it.

Pure: no film, no I/O beyond reading the two JSON files. `film statbook` writes the result.
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

MATCH_REWARD = 2.0          # a logged shot that finds its card
MADE_WEIGHT = 2.0           # log-odds weight of the card's made/missed call agreeing with the log
SIDE_REWARD = 0.8           # the card is at the basket this team attacks this period ...
SIDE_PENALTY = 3.0          # ... or at the other one
TIME_CAP = 5.0              # logged live, the time term can veto a match ...
REWATCH_TIME_CAP = 0.5      # ... logged while rewatching it only breaks ties: order, made/missed and basket decide
REWATCH_MIN_SIGMA_S = 15.0  # and a rewatch fit is never trusted closer than this
DIST_SCALE_FT = 7.0         # a logged spot this far from the card's located spot costs one point
STRONG_CARD_SKIP = 0.6      # leaving a confident ball card unmatched costs this; motion-only cards are free
FT_MATCH_REWARD = 0.6       # free throws are rarely on the film as cards; matching one is a small bonus
MIN_SIGMA_S = 3.0           # the time fit is never trusted closer than this
FIRST_SIGMA_FRACTION = 0.3  # before any anchors, the logged position in the period says this much of it
REBOUND_AFTER_S = 2.0       # a rebound lands this long after the shot reaches the rim
CLIP_BEFORE_S = 8.0
CLIP_AFTER_S = 3.0
PASSES = 4
GAP_WEIGHT = 1.0            # see gap_cost
GAP_CAP = 2.5
GAP_SLACK_S = 4.0
GAP_SLACK_FRACTION = 0.35
LINK_PLAYS = 4              # a link may skip up to this many logged shots (no card on film) ...
LINK_CARDS = 10             # ... and this many cards (not shots, or the other team's unlogged ones)
RESTART_COST = 2.0
LIVE_SHARE = 0.5           # this share of shots agreeing on one offset means the log was kept live

FOLLOW_OFFSET_S = {"reb": REBOUND_AFTER_S, "ast": 0.0, "blk": 0.0, "stl": 0.0}


@dataclass
class Play:
    seq: int
    id: str
    ms: float | None
    period: int | None
    type: str
    team: str | None = None
    player: str | None = None
    name: str | None = None
    made: bool | None = None
    pts: int | None = None
    off: bool | None = None
    x: float | None = None
    y: float | None = None
    ref: str | None = None
    group: str | None = None
    on_court: list | None = None
    kind: str | None = None
    action: str | None = None
    score_us: int | None = None
    score_them: int | None = None

    @property
    def is_shot(self) -> bool:
        return self.type == "shot"

    @property
    def is_ft(self) -> bool:
        return self.type == "shot" and self.pts == 1

    @property
    def dist_ft(self) -> float | None:
        if self.x is None or self.y is None:
            return None
        return math.hypot(self.x - 25.0, self.y - 5.25)


@dataclass
class StatBook:
    game: dict
    roster: dict                    # number -> name
    plays: list[Play]


@dataclass
class Card:
    """A film shot card, reduced to what matching needs."""
    id: str
    t: float
    period: int | None
    made_prob: float | None         # the film's belief the shot went in; None when it has no opinion
    strength: float                 # how sure the film is that this is a real shot (0 motion-only .. 1)
    side: str | None                # "left" / "right": which basket
    dist_ft: float | None = None
    clip: dict | None = None


@dataclass
class Placed:
    play: Play
    film_t: float | None
    how: str                        # "shot" (matched a card), "follow" (from its shot), "fit", "none"
    card_id: str | None = None
    score: float | None = None
    uncertainty_s: float | None = None
    clock_s: float | None = None
    film_period: int | None = None

    def to_json(self) -> dict:
        d = {k: v for k, v in asdict(self.play).items()}
        d.update(film_t=None if self.film_t is None else round(self.film_t, 2), how=self.how, card_id=self.card_id,
                 match_score=None if self.score is None else round(self.score, 2),
                 uncertainty_s=None if self.uncertainty_s is None else round(self.uncertainty_s, 1),
                 clock_s=self.clock_s, film_period=self.film_period)
        if self.film_t is not None:
            before, after = clip_window(self)
            d["clip"] = {"start": round(self.film_t - before, 2), "end": round(self.film_t + after, 2)}
        return d


# ---------------------------------------------------------------- loading


def _ms(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000.0


def load_statbook(path: Path) -> StatBook:
    """Either the stat book's own export (format "titans-stat-book") or a raw game from its store/backup."""
    data = json.loads(Path(path).read_text())
    if "game" in data and "events" in data:                      # the export
        g = data["game"]
        roster = {str(r["num"]): r.get("name") or str(r["num"]) for r in g.get("roster", [])}
        plays = [Play(seq=e["seq"], id=e["id"], ms=_ms(e.get("time")), period=e.get("period"), type=e["type"],
                      team=e.get("team"), player=e.get("player"), name=e.get("name"), made=e.get("made"),
                      pts=e.get("pts"), off=e.get("off"), x=e.get("x_ft"), y=e.get("y_ft"), ref=e.get("ref"),
                      group=e.get("group"), on_court=e.get("on_court"), kind=e.get("timeout_kind"),
                      action=e.get("action"), score_us=e.get("score_us"), score_them=e.get("score_them"))
                 for e in data["events"]]
        game = {k: g.get(k) for k in ("id", "date", "us", "them", "quarter_minutes", "final_score")}
    else:                                                        # a raw game object
        roster = {str(r["num"]): r.get("name") or str(r["num"]) for r in data.get("roster", [])}
        plays = []
        us = them = 0
        for i, e in enumerate(data["events"], 1):
            if e["type"] == "shot" and e.get("made"):
                if e.get("team") == "us":
                    us += e.get("pts") or 0
                else:
                    them += e.get("pts") or 0
            plays.append(Play(seq=i, id=e["id"], ms=_ms(e.get("ms", e.get("t"))), period=e.get("p"), type=e["type"],
                              team=None if e["type"] in ("sub", "period", "final") else e.get("team"),
                              player=e.get("player"), name=roster.get(str(e.get("player"))) if e.get("player") else None,
                              made=e.get("made") if e["type"] == "shot" else None, pts=e.get("pts") if e["type"] == "shot" else None,
                              off=e.get("off"), x=e.get("x"), y=e.get("y"), ref=e.get("ref"), group=e.get("grp"),
                              on_court=e.get("on"), kind=e.get("kind"), action=e.get("action"), score_us=us, score_them=them))
        game = {"id": data.get("id"), "date": data.get("date"), "us": data.get("us"), "them": data.get("them"),
                "quarter_minutes": data.get("qlen"), "final_score": {"us": us, "them": them}}
    return StatBook(game=game, roster=roster, plays=plays)


def cards_from_events(events, sides: dict | None = None, rejected: set | None = None) -> list[Card]:
    """Film events -> cards. Cards the coach rejected in review are not shots and are left out.

    `sides` maps an event id (or, failing that, a rim track id) to the basket: "left" or "right"."""
    sides = sides or {}
    rejected = rejected or set()
    out = []
    for e in events:
        if e.id in rejected:
            continue
        sig = e.signals or {}
        source = sig.get("source")
        # The film's own made/missed model, not the fused call: a board whose score was never kept (a
        # scrimmage) reads 0-0 all game and turns every make into a miss in the fused call.
        clf = sig.get("clf")
        if isinstance(clf, (list, tuple)) and len(clf) == 2 and clf[0] in ("made", "missed"):
            p = min(max(float(clf[1]), 0.5), 0.95)
            made_prob = p if clf[0] == "made" else 1.0 - p
        elif e.made is None:
            made_prob = None
        else:
            c = min(max(float(e.confidence or 0.5), 0.5), 0.97)
            made_prob = c if e.made else 1.0 - c
        not_a_shot = isinstance(clf, (list, tuple)) and clf and clf[0] == "none"
        if source == "ball" and "motion-only" not in e.flags:
            strength = 1.0 if (sig.get("arrival_conf") or 0) >= 0.6 else 0.6
        elif source == "scoreboard":
            strength = 0.8
        else:
            strength = 0.0
        if e.type != "shot":
            strength = 0.0
        elif not_a_shot:
            strength *= 0.5
        dist = math.hypot(e.x - 25.0, e.y - 5.25) if e.x is not None and e.y is not None else None
        out.append(Card(id=e.id, t=e.t, period=e.quarter, made_prob=made_prob, strength=strength,
                        side=sides.get(e.id, sides.get(e.rim_id)), dist_ft=dist, clip=e.clip))
    return sorted(out, key=lambda c: c.t)


# ---------------------------------------------------------------- the time fit


class TimeFit:
    """Logged ms -> film seconds, piecewise linear through anchor pairs; outside them, slope 1 (real time)."""

    def __init__(self, pairs: list[tuple[float, float]]):
        pts = sorted(pairs)
        clean: list[tuple[float, float]] = []
        for ms, t in pts:                       # keep it monotone: drop an anchor that would run time backwards
            if clean and (ms <= clean[-1][0] or t <= clean[-1][1]):
                continue
            clean.append((ms, t))
        self.pairs = clean

    def __call__(self, ms: float) -> float | None:
        if not self.pairs or ms is None:
            return None
        xs = [p[0] for p in self.pairs]
        i = bisect.bisect_left(xs, ms)
        if i == 0:
            ms0, t0 = self.pairs[0]
            return t0 + (ms - ms0) / 1000.0
        if i >= len(xs):
            ms1, t1 = self.pairs[-1]
            return t1 + (ms - ms1) / 1000.0
        (ms0, t0), (ms1, t1) = self.pairs[i - 1], self.pairs[i]
        return t0 + (t1 - t0) * (ms - ms0) / (ms1 - ms0)

    def gap_s(self, ms: float) -> float | None:
        """Film seconds between the anchors either side of `ms`: how far a fitted time can be off."""
        xs = [p[0] for p in self.pairs]
        i = bisect.bisect_left(xs, ms)
        if not self.pairs:
            return None
        if i == 0 or i >= len(xs):
            return None
        return self.pairs[i][1] - self.pairs[i - 1][1]


def proportional_fit(log_span: tuple[float, float], film_span: tuple[float, float]) -> TimeFit:
    """Before any anchors: the logged period maps onto the film period end to end."""
    return TimeFit([(log_span[0], film_span[0]), (log_span[1], film_span[1])])


# ---------------------------------------------------------------- alignment


def _made_term(play: Play, card: Card) -> float:
    if card.made_prob is None or play.made is None:
        return 0.0
    q = card.made_prob if play.made else 1.0 - card.made_prob
    return MADE_WEIGHT * max(-2.0, min(1.0, math.log(max(q, 1e-3) / 0.5)))


def _held_out_error(fit: TimeFit) -> float:
    """How far the fit misses an anchor it was not given: the honest scale of its error. Logged live this is
    the tapping lag, a few seconds; logged while rewatching it is as large as the pauses and skips made it."""
    pairs = fit.pairs
    if len(pairs) < 3:
        return 30.0
    errs = sorted(abs(TimeFit(pairs[:k] + pairs[k + 1:])(ms) - t) for k, (ms, t) in enumerate(pairs[1:-1], 1))
    return errs[len(errs) // 2]


def match_score(play: Play, card: Card, expect_t: float | None, sigma: float, attack: dict,
                time_cap: float = TIME_CAP) -> float:
    s = FT_MATCH_REWARD if play.is_ft else MATCH_REWARD * (0.5 + 0.5 * card.strength)
    s += _made_term(play, card)
    if expect_t is not None:
        s -= min(time_cap, 0.5 * ((card.t - expect_t) / sigma) ** 2)
    want = attack.get(play.team)
    if want and card.side:
        s += SIDE_REWARD if card.side == want else -SIDE_PENALTY
    d = play.dist_ft
    if d is not None and card.dist_ft is not None:
        s -= min(2.0, abs(d - card.dist_ft) / DIST_SCALE_FT)
    if play.is_ft and card.dist_ft is not None and abs(card.dist_ft - 15.0) > 5.0:
        s -= 1.0
    return s


def gap_cost(log_gap_s: float | None, film_gap_s: float, rate: float, weight: float) -> float:
    """Consecutive matched shots should be as far apart on film as the log says, scaled by how fast the log
    ran against the film (`rate`: film seconds per logged second). Loose: tapping lag is a few seconds and a
    rewatch speeds up and pauses. This is what stops a run of shots from all sliding one card along."""
    if log_gap_s is None or weight == 0:
        return 0.0
    want = max(0.0, log_gap_s) * rate
    if film_gap_s <= 0:
        return GAP_CAP * weight
    return weight * min(GAP_CAP, abs(film_gap_s - want) / (GAP_SLACK_S + GAP_SLACK_FRACTION * want))


def align(plays: list[Play], cards: list[Card], expect: list[float | None], sigma: float,
          attack: dict, time_cap: float = TIME_CAP, rate: float = 1.0,
          gap_weight: float = GAP_WEIGHT) -> list[tuple[int, int, float]]:
    """Order-preserving alignment of logged shots to film cards. Returns (play index, card index, score).

    A chain of matches: each match earns its match_score, each strong card left out between two matches
    costs its skip, and each link pays gap_cost for spacing that disagrees with the log. A chain may also
    restart after a gap too long to link (more than LINK_PLAYS shots or LINK_CARDS cards) for RESTART_COST."""
    n, m = len(plays), len(cards)
    if n == 0 or m == 0:
        return []
    skip = [STRONG_CARD_SKIP * c.strength for c in cards]
    cum = [0.0]
    for v in skip:
        cum.append(cum[-1] + v)                       # cum[j] = skips of cards 0..j-1
    secs = [p.ms / 1000.0 if p.ms is not None else None for p in plays]
    NEG = -1e18
    sc = [[match_score(plays[i], cards[j], expect[i], sigma, attack, time_cap) for j in range(m)] for i in range(n)]
    best = [[NEG] * m for _ in range(n)]
    back: list[list[tuple[int, int] | None]] = [[None] * m for _ in range(n)]
    # pre[i][j]: the best chain ending at play <= i, card <= j, as best + cum[j + 1] (so skips telescope)
    pre = [[(NEG, None)] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if sc[i][j] <= 0:
                cand, link = NEG, None
            else:
                cand, link = -cum[j], None                         # the chain starts here
                if i > 0 and j > 0:                                 # restart after a long unlinked stretch
                    v, at = pre[i - 1][j - 1]
                    if v > NEG / 2 and v - cum[j] - RESTART_COST > cand:
                        cand, link = v - cum[j] - RESTART_COST, at
                for ii in range(max(0, i - LINK_PLAYS), i):
                    if secs[ii] is None or secs[i] is None:
                        continue
                    for jj in range(max(0, j - LINK_CARDS), j):
                        if best[ii][jj] <= NEG / 2:
                            continue
                        v = (best[ii][jj] - (cum[j] - cum[jj + 1])
                             - gap_cost(secs[i] - secs[ii], cards[j].t - cards[jj].t, rate, gap_weight))
                        if v > cand:
                            cand, link = v, (ii, jj)
                cand += sc[i][j]
            best[i][j], back[i][j] = cand, link
            here = (best[i][j] + cum[j + 1], (i, j)) if cand > NEG / 2 else (NEG, None)
            up = pre[i - 1][j] if i > 0 else (NEG, None)
            left = pre[i][j - 1] if j > 0 else (NEG, None)
            pre[i][j] = max(here, up, left, key=lambda x: x[0])
    total, end = NEG, None
    for i in range(n):
        for j in range(m):
            if best[i][j] > NEG / 2:
                v = best[i][j] - (cum[m] - cum[j + 1])
                if v > total:
                    total, end = v, (i, j)
    if end is None or total <= -cum[m]:
        return []
    out = []
    at: tuple[int, int] | None = end
    while at is not None:
        i, j = at
        out.append((i, j, sc[i][j]))
        at = back[i][j]
    return out[::-1]


def _attack_sides(pairs: list[tuple[Play, Card]]) -> dict:
    """Which basket each team attacked this period, by majority of the matched cards that know their side."""
    out = {}
    for team in ("us", "them"):
        votes = [c.side for p, c in pairs if p.team == team and c.side]
        if len(votes) >= 3:
            left = votes.count("left")
            if max(left, len(votes) - left) >= 0.7 * len(votes):
                out[team] = "left" if left * 2 > len(votes) else "right"
    if "us" in out and "them" not in out:
        out["them"] = "right" if out["us"] == "left" else "left"
    if "them" in out and "us" not in out:
        out["us"] = "right" if out["them"] == "left" else "left"
    if out.get("us") == out.get("them"):
        return {}
    return out


@dataclass
class PeriodResult:
    period: int
    film_span: tuple[float, float] | None
    shots_logged: int
    shots_matched: int
    fts_logged: int
    fts_matched: int
    cards: int
    strong_cards_unmatched: int
    attack: dict = field(default_factory=dict)
    sigma_s: float | None = None
    mode: str = ""
    side_totals: list = field(default_factory=list)   # the period's fit with us attacking left, then right


def live_offset(shots: list[Play], cards: list[Card], tol_s: float = 3.0) -> tuple[float | None, float]:
    """Logged live, film time = logged time + one offset. Find the offset most shots agree on, and the share
    of shots that have a strong card within `tol_s` of it. A rewatch log has no such offset."""
    strong = sorted(c.t for c in cards if c.strength >= 0.6)
    secs = [p.ms / 1000.0 for p in shots if p.ms is not None and not p.is_ft]
    if not strong or len(secs) < 4:
        return None, 0.0
    candidates = [t - s for s in secs for t in strong]

    def support(d: float) -> int:
        n = 0
        for s in secs:
            k = bisect.bisect_left(strong, s + d - tol_s)
            n += k < len(strong) and strong[k] <= s + d + tol_s
        return n

    best = max(candidates, key=support)
    return best, support(best) / len(secs)


def _rate(fit: TimeFit, log_span, film_span) -> float:
    """Film seconds per logged second over the period, from the fit's anchors once there are some."""
    if len(fit.pairs) >= 2 and fit.pairs[-1][0] > fit.pairs[0][0]:
        (m0, t0), (m1, t1) = fit.pairs[0], fit.pairs[-1]
        return (t1 - t0) / ((m1 - m0) / 1000.0)
    span = (log_span[1] - log_span[0]) / 1000.0
    return (film_span[1] - film_span[0]) / span if span > 0 else 1.0


def _prune(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop anchors the others disagree with: a wrong match must not bend the fit for its neighbours."""
    pairs = sorted(pairs)
    for _ in range(3):
        if len(pairs) < 4:
            return pairs
        errs = [abs(TimeFit(pairs[:k] + pairs[k + 1:])(ms) - t) for k, (ms, t) in enumerate(pairs)]
        med = sorted(errs)[len(errs) // 2]
        keep = [p for p, e in zip(pairs, errs) if e <= max(3 * med, 10.0)]
        if len(keep) == len(pairs):
            return pairs
        pairs = keep
    return pairs


def match_period(shots: list[Play], cards: list[Card], log_span: tuple[float, float],
                 film_span: tuple[float, float]) -> tuple[list[tuple[int, int, float]], TimeFit, PeriodResult]:
    offset, share = live_offset(shots, cards)
    log_len = (log_span[1] - log_span[0]) / 1000.0
    film_len = film_span[1] - film_span[0]
    # live, the logged period lasts about as long as the film's; a rewatch skips and pauses
    same_length = film_len > 0 and 0.75 <= log_len / film_len <= 1.25
    live = offset is not None and share >= LIVE_SHARE and same_length
    cap, floor = (TIME_CAP, MIN_SIGMA_S) if live else (REWATCH_TIME_CAP, REWATCH_MIN_SIGMA_S)
    if live:
        mode = "live"
        fit = TimeFit([(log_span[0], log_span[0] / 1000 + offset), (log_span[1], log_span[1] / 1000 + offset)])
        sigma = 2 * MIN_SIGMA_S
    else:
        mode = "rewatch"
        fit = proportional_fit(log_span, film_span)
        sigma = max(MIN_SIGMA_S, FIRST_SIGMA_FRACTION * (film_span[1] - film_span[0]))
    start = (fit, sigma)
    sided = sum(c.side is not None for c in cards) >= 0.5 * max(1, len(cards))
    # Which basket each team attacks is not known in advance (and a scrimmage may not switch at the half):
    # try both, and keep the one the whole period fits better. Without basket sides, one pass with none.
    options = ([{"us": "left", "them": "right"}, {"us": "right", "them": "left"}] if sided else [{}])
    best = None
    totals: list[float] = []
    for attack in options:
        fit, sigma = start
        matches: list[tuple[int, int, float]] = []
        for _ in range(PASSES):
            expect = [fit(p.ms) if p.ms is not None else None for p in shots]
            rate = 1.0 if live else _rate(fit, log_span, film_span)
            matches = align(shots, cards, expect, sigma, attack, cap, rate=rate)
            sure = [(shots[i].ms, cards[j].t) for i, j, s in matches
                    if s >= MATCH_REWARD * 0.75 and not shots[i].is_ft and shots[i].ms is not None]
            sure = _prune(sure)
            if len(sure) >= 3:
                # the period's ends hold the fit in place before its first matched shot and after its last
                edges = [(log_span[0], min(film_span[0], min(t for _, t in sure))),
                         (log_span[1], max(film_span[1], max(t for _, t in sure)))]
                fit = TimeFit(sure + edges)
                sigma = max(floor, _held_out_error(fit))
        used = {j for _, j, _ in matches}
        total = sum(s for _, _, s in matches) - sum(STRONG_CARD_SKIP * c.strength for j, c in enumerate(cards) if j not in used)
        totals.append(round(total, 1))
        if best is None or total > best[0]:
            best = (total, attack, fit, sigma, matches)
    _, attack, fit, sigma, matches = best
    matched_cards = {j for _, j, _ in matches}
    res = PeriodResult(
        period=shots[0].period if shots else 0, film_span=film_span,
        shots_logged=sum(not p.is_ft for p in shots), shots_matched=sum(not shots[i].is_ft for i, _, _ in matches),
        fts_logged=sum(p.is_ft for p in shots), fts_matched=sum(shots[i].is_ft for i, _, _ in matches),
        cards=len(cards), strong_cards_unmatched=sum(1 for j, c in enumerate(cards) if c.strength >= 1.0 and j not in matched_cards),
        attack=attack, sigma_s=round(sigma, 1), mode=mode, side_totals=totals)
    return matches, fit, res


# ---------------------------------------------------------------- the whole game


def film_period_spans(cards: list[Card], timeline=None) -> dict[int, tuple[float, float]]:
    """Each film period's span: from the scoreboard's period digit while its clock runs, else from the cards."""
    spans: dict[int, list[float]] = {}
    if timeline:
        reads = [r for r in timeline if r.period is not None]
        for prev, r in zip(reads, reads[1:]):
            running = (prev.clock_s is not None and r.clock_s is not None and r.period == prev.period
                       and 0 < prev.clock_s - r.clock_s <= (r.t - prev.t) + 1.5)
            if running:
                s = spans.setdefault(r.period, [prev.t, r.t])
                s[0], s[1] = min(s[0], prev.t), max(s[1], r.t)
    if not spans:
        for c in cards:
            if c.period is not None:
                s = spans.setdefault(c.period, [c.t, c.t])
                s[0], s[1] = min(s[0], c.t), max(s[1], c.t)
    return {p: (a, b) for p, (a, b) in sorted(spans.items())}


def pair_periods(log_periods: list[int], film_spans: dict[int, tuple[float, float]]) -> dict[int, int]:
    """Logged period -> film period. Same numbers when the film has them all; else in order."""
    film = sorted(film_spans)
    if all(p in film_spans for p in log_periods):
        return {p: p for p in log_periods}
    return {p: f for p, f in zip(sorted(log_periods), sorted(film, key=lambda f: film_spans[f][0]))}


def clip_window(placed: Placed) -> tuple[float, float]:
    """Seconds before and after the play's film time to show; wider when its time is a fit, not a match."""
    if placed.how == "shot":
        return CLIP_BEFORE_S, CLIP_AFTER_S
    slack = min(12.0, (placed.uncertainty_s or 0.0))
    if placed.play.type == "reb":
        return 4.0 + slack, 3.0 + slack
    return 6.0 + slack, 4.0 + slack


def place_game(book: StatBook, cards: list[Card], timeline=None, clock_at=None,
               film_spans: dict | None = None) -> tuple[list[Placed], list[PeriodResult]]:
    plays = book.plays
    spans = film_spans or film_period_spans(cards, timeline)
    log_periods = sorted({p.period for p in plays if p.period is not None})
    pairing = pair_periods(log_periods, spans)
    placed: dict[str, Placed] = {}
    results: list[PeriodResult] = []
    fits: dict[int, TimeFit] = {}
    for lp in log_periods:
        fp = pairing.get(lp)
        in_period = [p for p in plays if p.period == lp]
        timed = [p.ms for p in in_period if p.ms is not None]
        if fp is None or fp not in spans or not timed:
            continue
        film_span = spans[fp]
        pc = [c for c in cards if film_span[0] - 5 <= c.t <= film_span[1] + 5]
        shots = [p for p in in_period if p.is_shot]
        log_span = (min(timed), max(timed))
        matches, fit, res = match_period(shots, pc, log_span, film_span)
        res.period = lp
        results.append(res)
        fits[lp] = fit
        for i, j, s in matches:
            placed[shots[i].id] = Placed(shots[i], pc[j].t, "shot", card_id=pc[j].id, score=s, uncertainty_s=1.0,
                                         film_period=fp)
    for p in plays:
        if p.id in placed:
            continue
        parent = placed.get(p.ref) if p.ref else None
        if parent is not None and parent.film_t is not None and p.type in FOLLOW_OFFSET_S:
            # a play that follows an estimated shot is as estimated as the shot
            how = "follow" if parent.how == "shot" else "fit"
            placed[p.id] = Placed(p, parent.film_t + FOLLOW_OFFSET_S[p.type], how, card_id=parent.card_id,
                                  uncertainty_s=2.0 if how == "follow" else parent.uncertainty_s,
                                  film_period=parent.film_period)
            continue
        fit = fits.get(p.period)
        if fit is None or p.ms is None:
            placed[p.id] = Placed(p, None, "none")
            continue
        gap = fit.gap_s(p.ms)
        t = fit(p.ms)
        span = spans.get(pairing.get(p.period))
        if span is not None:                         # a play logged in a period happened in that period
            t = min(max(t, span[0]), span[1])
        placed[p.id] = Placed(p, t, "fit", uncertainty_s=(gap / 2 if gap else 15.0),
                              film_period=pairing.get(p.period))
    out = [placed[p.id] for p in plays]
    if clock_at is not None:
        for x in out:
            if x.film_t is not None:
                x.clock_s = clock_at(x.film_t)
    return out, results


def label(placed: Placed, roster: dict, them: str = "Opp") -> str:
    """What the play was, from the stat book. "~" marks a play placed by the time fit, not matched on film."""
    text = _label(placed, roster, them)
    return f"~ {text}" if placed.how == "fit" else text


def _label(placed: Placed, roster: dict, them: str) -> str:
    p = placed.play
    who = p.name or (roster.get(str(p.player)) if p.player else None) or (them if p.team == "them" else "Team")
    if p.type == "shot":
        kind = "FT" if p.pts == 1 else f"{p.pts}PT"
        return f"{who} {kind} {'made' if p.made else 'missed'}"
    names = {"reb": ("off rebound" if p.off else "def rebound"), "ast": "assist", "stl": "steal", "blk": "block",
             "to": "turnover", "foul": "foul", "timeout": "timeout", "sub": "sub", "period": p.action or "period"}
    return f"{who} {names.get(p.type, p.type)}"
