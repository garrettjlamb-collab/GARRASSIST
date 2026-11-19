import random

import numpy as np

from film.interpret.camstops import Stretch, camera_position, film_time, find_cuts, fit_offsets
from film.interpret.statbook import Play, StatBook, place_by_stretches


def _game(seed=3):
    """A synthetic 4-quarter game on a camera that stopped at breaks.

    Possessions alternate ends; the camera faces the end the ball is at. Logged shots carry the team, and the
    teams swap ends at halftime. The film is the game minus the stops."""
    rng = random.Random(seed)
    game_len = 4800.0
    stops = [(840.0, 60.0), (1400.0, 75.0), (2150.0, 650.0), (3400.0, 90.0), (3900.0, 100.0)]   # (game s, length)
    # which end the ball is at, every possession of 8-20 s; us attack +1 in the first half
    ends, t, side = [], 0.0, 1
    while t < game_len:
        dur = rng.uniform(8, 20)
        ends.append((t, t + dur, side))
        t += dur
        side = -side
    shots = []
    for a, b, s in ends:
        if rng.random() < 0.7:
            team_side = 1 if a < 2150 else -1          # us attack +1 before the half, -1 after
            team = 1 if s == team_side else -1
            period = 1 + int(a // 1200)
            shots.append((b - 1.5 + 3.0, team, period))   # logged ~1.5 s before the end of the trip, plus tap lag
    # film time from game time
    def to_film(g):
        removed = 0.0
        for at, length in stops:
            if g >= at + length:
                removed += length
            elif g >= at:
                return None
        return g - removed
    true_cuts = [to_film(at + length) for at, length in stops]
    film_len = to_film(game_len)
    rate = 5.0
    ts = np.arange(0, film_len, 1 / rate)
    # camera heading: towards the ball's end, with drift; sample-to-sample shift is its derivative
    film_to_game = []
    for ft in ts:
        g, acc = ft, 0.0
        for at, length in stops:
            if g + acc >= at:
                acc += length
        film_to_game.append(ft + acc)
    heading = []
    for g in film_to_game:
        s = next((s for a, b, s in ends if a <= g < b), 1)
        heading.append(40.0 * s)
    heading = np.convolve(np.array(heading), np.ones(10) / 10, "same") + 0.01 * ts
    pan = [(float(ts[i]), float(heading[i] - heading[i - 1])) for i in range(1, len(ts))]
    return shots, true_cuts, film_len, pan, rate, stops


def test_cuts_are_the_frame_jumps_far_above_their_neighbours():
    rng = random.Random(0)
    diffs = [(i / 10, rng.uniform(2, 6)) for i in range(3000)]
    for k in (500, 1700, 1705):                      # 1705 is the same restart (within MIN_STRETCH_S)
        diffs[k] = (k / 10, 55.0)
    diffs[2400] = (240.0, 9.0)                       # a fast pan: bigger, not a restart
    assert find_cuts(diffs) == [50.0, 170.0]


def test_offsets_come_back_from_the_camera_swing_and_fall_at_every_stop():
    shots, cuts, film_len, pan, rate, stops = _game()
    bounds = [0.0] + cuts + [film_len]
    cam_t, cam_pos = camera_position(pan, bounds, rate)
    got = fit_offsets(bounds, cam_t, cam_pos, shots)
    want = [0.0]
    for _, length in stops:
        want.append(want[-1] - length)
    for s, w in zip(got, want):
        assert abs(s.offset - w) <= 6, (s, w)          # clips run 10 s before and 8 after
    assert all(a.offset >= b.offset for a, b in zip(got, got[1:]))


def test_a_play_logged_while_the_camera_was_off_is_off_the_film():
    stretches = [Stretch(0, 100, 0.0, 5, 1.0), Stretch(100, 200, -60.0, 5, 1.0)]
    assert film_time(stretches, 50)[0] == 50
    assert film_time(stretches, 130) == (None, None)        # 100-160 logged: the camera was stopped
    assert film_time(stretches, 170)[0] == 110


def test_placing_by_stretches_puts_each_play_at_its_logged_time_plus_offset():
    t0 = 1_800_000_000_000.0
    plays = [Play(seq=1, id="p", ms=t0, period=1, type="period", action="start"),
             Play(seq=2, id="a", ms=t0 + 20_000, period=1, type="shot", team="us", made=True, pts=2),
             Play(seq=3, id="b", ms=t0 + 130_000, period=1, type="shot", team="them", made=False, pts=3),
             Play(seq=4, id="c", ms=t0 + 175_000, period=1, type="to", team="us")]
    book = StatBook(game={}, roster={}, plays=plays)
    placed = place_by_stretches(book, [Stretch(0, 100, 0.0, 5, 1.0), Stretch(100, 200, -60.0, 5, 1.0)])
    assert [p.how for p in placed] == ["none", "camera", "none", "camera"]
    assert placed[1].film_t == 17.0                          # tapped 3 s after the shot
    assert placed[3].film_t == 115.0
