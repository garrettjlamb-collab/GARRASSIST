import json
import random

from film.interpret.statbook import (Card, Play, StatBook, TimeFit, align, label, load_statbook, place_game)


def _game(seed=1, rewatch=False, n_shots=40, period_s=900.0, noise_cards=25, missing=0.1):
    """A synthetic period: true shots in the film, the log of them, and the film's noisy cards."""
    rng = random.Random(seed)
    film_t = sorted(rng.uniform(10, period_s - 10) for _ in range(n_shots))
    t0_ms = 1_800_000_000_000.0
    plays, cards = [], []
    seq = 0
    clock_ms, prev_t = t0_ms, 0.0
    for k, t in enumerate(film_t):
        made = rng.random() < 0.45
        team = "us" if rng.random() < 0.5 else "them"
        # live: the phone clock runs with the film, plus 2-6 s of tapping lag. Rewatch: he skips dead time and
        # plays some stretches fast (0.3x to 1x of film time), and now and then pauses or rewinds (up to a minute)
        if rewatch:
            clock_ms += (t - prev_t) * 1000 * rng.uniform(0.3, 1.0) + (rng.uniform(0, 60_000) if rng.random() < 0.15 else 0)
            ms, prev_t = clock_ms + rng.uniform(1000, 4000), t
        else:
            ms = t0_ms + t * 1000 + rng.uniform(2000, 6000)
        seq += 1
        plays.append(Play(seq=seq, id=f"s{k}", ms=ms, period=1, type="shot", team=team, made=made, pts=2))
        if not made:
            seq += 1
            plays.append(Play(seq=seq, id=f"r{k}", ms=ms + 800, period=1, type="reb", team=team, ref=f"s{k}"))
        if rng.random() > missing:
            side = ("left" if team == "us" else "right")
            film_made = made if rng.random() < 0.78 else not made
            cards.append(Card(id=f"c{k}", t=t + rng.uniform(-0.5, 0.5), period=1,
                              made_prob=0.8 if film_made else 0.2, strength=1.0, side=side))
    for n in range(noise_cards):
        cards.append(Card(id=f"n{n}", t=rng.uniform(0, period_s), period=1, made_prob=None, strength=0.0,
                          side=rng.choice(["left", "right"])))
    cards.sort(key=lambda c: c.t)
    plays.insert(0, Play(seq=0, id="start", ms=plays[0].ms - 5000, period=1, type="period", action="start"))
    return StatBook(game={}, roster={}, plays=plays), cards


def _accuracy(book, cards):
    placed, results = place_game(book, cards)
    shots = [p for p in placed if p.play.type == "shot"]
    right = sum(1 for p in shots if p.card_id == "c" + p.play.id[1:])
    findable = sum(1 for p in book.plays if p.type == "shot" and any(c.id == "c" + p.id[1:] for c in cards))
    return right, findable, placed, results


def test_live_logged_game_matches_nearly_every_findable_shot():
    """Measured 2026-11-17 over 8 seeds: 269 of 281 (96%). The floor guards against a regression."""
    right = findable = 0
    for seed in range(8):
        book, cards = _game(seed)
        r, f, _, results = _accuracy(book, cards)
        right, findable = right + r, findable + f
        assert results[0].mode == "live"
    assert right >= 0.92 * findable, (right, findable)


def test_a_rewatch_with_scrambled_times_still_matches_most_shots_by_order():
    """A rewatch played at 0.3x to 1x with pauses up to a minute: the order, the made/missed call and the
    basket carry it. Measured 2026-11-17 over 8 seeds: 228 of 282 (81%)."""
    right = findable = 0
    for seed in range(8):
        book, cards = _game(seed, rewatch=True)
        r, f, _, results = _accuracy(book, cards)
        right, findable = right + r, findable + f
        assert results[0].mode == "rewatch"
    assert right >= 0.75 * findable, (right, findable)


def test_each_period_finds_which_basket_each_team_attacked():
    for seed in range(4):
        book, cards = _game(seed, rewatch=True)
        _, results = place_game(book, cards)
        assert results[0].attack == {"us": "left", "them": "right"}


def test_a_rebound_follows_its_missed_shot():
    book, cards = _game(3)
    placed, _ = place_game(book, cards)
    by = {p.play.id: p for p in placed}
    for p in placed:
        if p.play.type == "reb" and by[p.play.ref].how == "shot":
            assert p.how == "follow" and p.film_t == by[p.play.ref].film_t + 2.0


def test_no_shot_is_matched_to_a_card_out_of_order():
    book, cards = _game(4, rewatch=True)
    placed, _ = place_game(book, cards)
    times = [p.film_t for p in placed if p.how == "shot"]
    assert times == sorted(times)


def test_align_prefers_the_card_whose_made_call_agrees():
    shot = Play(seq=1, id="a", ms=0.0, period=1, type="shot", team="us", made=True, pts=2)
    cards = [Card("miss", 10.0, 1, 0.1, 1.0, None), Card("make", 11.0, 1, 0.9, 1.0, None)]
    [(i, j, _)] = align([shot], cards, [10.5], 10.0, {})
    assert cards[j].id == "make"


def test_time_fit_is_piecewise_linear_and_real_time_outside():
    fit = TimeFit([(0.0, 100.0), (10_000.0, 105.0)])
    assert fit(5_000.0) == 102.5
    assert fit(12_000.0) == 107.0          # past the last anchor: one logged second is one film second
    assert fit.gap_s(5_000.0) == 5.0


def test_loads_the_raw_game_and_the_export(tmp_path):
    raw = {"id": "g", "date": "2026-11-14", "us": "Marcos", "them": "ALA", "qlen": 8,
           "roster": [{"num": "3", "name": "Ryan"}],
           "events": [{"id": "e1", "type": "shot", "team": "us", "player": "3", "made": True, "pts": 3, "p": 1,
                       "ms": 1000, "grp": "e1", "on": ["3"], "x": 25, "y": 25}]}
    (tmp_path / "raw.json").write_text(json.dumps(raw))
    book = load_statbook(tmp_path / "raw.json")
    assert book.plays[0].name == "Ryan" and book.plays[0].score_us == 3 and round(book.plays[0].dist_ft, 2) == 19.75
    export = {"format": "titans-stat-book", "game": {"id": "g", "roster": [{"num": "3", "name": "Ryan"}]},
              "events": [{"seq": 1, "id": "e1", "time": "2026-11-14T20:55:13.633Z", "period": 1, "type": "shot",
                          "team": "us", "player": "3", "name": "Ryan", "pts": 3, "made": False, "x_ft": None}]}
    (tmp_path / "exp.json").write_text(json.dumps(export))
    book = load_statbook(tmp_path / "exp.json")
    assert book.plays[0].ms == 1794689713633.0
    placed, _ = place_game(book, [])
    assert label(placed[0], book.roster) == "Ryan 3PT missed"


def _two_periods(seed, rewatch=False, missing=0.55):
    """Two periods of `_game` on one film: the second starts 1200 s later in the film and in the log. With
    over half the cards missing, a period alone falls short of LIVE_SHARE."""
    a, ca = _game(seed, rewatch=rewatch, missing=missing)
    b, cb = _game(seed + 100, rewatch=rewatch, missing=missing)
    shift = 1200.0
    plays = list(a.plays)
    for p in b.plays:
        plays.append(Play(seq=p.seq + 1000, id="b" + p.id, ms=p.ms + shift * 1000, period=2, type=p.type,
                          team=p.team, made=p.made, pts=p.pts, ref=("b" + p.ref) if p.ref else None,
                          action=p.action))
    cards = list(ca) + [Card(id="b" + c.id, t=c.t + shift, period=2, made_prob=c.made_prob, strength=c.strength,
                             side=c.side) for c in cb]
    return StatBook(game={}, roster={}, plays=plays), sorted(cards, key=lambda c: c.t)


def test_periods_that_agree_on_one_offset_are_live_even_when_the_board_misreads_them():
    """Shadow Mountain 2026-11-20: all four quarters found the same offset (373-374 s), three fell short of
    LIVE_SHARE alone, and the board's quarter spans were wrong, so their plays were left loose.
    Measured 2026-11-20 over 8 seeds: 242 of 272 findable shots right (26 before agreement)."""
    want = lambda pid: ("b" if pid.startswith("b") else "") + "c" + pid.lstrip("b")[1:]
    right = findable = 0
    for seed in range(8):
        book, cards = _two_periods(seed)
        wrong_spans = {1: (0.0, 600.0), 2: (1500.0, 1800.0)}      # the board cut both periods short
        placed, results = place_game(book, cards, film_spans=wrong_spans)
        assert [r.mode for r in results] == ["live", "live"], (seed, [r.mode for r in results])
        right += sum(p.play.type == "shot" and p.card_id == want(p.play.id) for p in placed)
        findable += sum(1 for p in book.plays if p.type == "shot" and any(c.id == want(p.id) for c in cards))
    assert right >= 0.85 * findable, (right, findable)


def test_a_rewatch_does_not_agree_with_itself_across_periods():
    for seed in range(4):
        book, cards = _two_periods(seed, rewatch=True, missing=0.1)
        _, results = place_game(book, cards)
        assert [r.mode for r in results] == ["rewatch", "rewatch"], seed
