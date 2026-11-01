import numpy as np

from film.interpret.court import Mapping, WIDTH, FT_LINE_Y, template
from film.interpret.location import Look, ball_sightings, find_flight, free_throw_setup, locate, shot_spot

# Feet -> pixels: along the court runs right at 10 px/ft, across runs down at 10 px/ft (right-handed, as fit_mapping wants).
BACK = np.array([[0.0, 10.0, 50.0], [10.0, 0.0, 50.0], [0.0, 0.0, 1.0]])
MAP = Mapping(H=np.linalg.inv(BACK), points_used=8, error_px=1.0, conf=1.0, end="near", length=94.0)


def px(x, y):
    v = BACK @ np.array([x, y, 1.0])
    return float(v[0]), float(v[1])


def person_at(x, y, h=60, w=24):
    fx, fy = px(x, y)
    return (fx - w / 2, fy - h, fx + w / 2, fy, 0.9)


def ball_box(x, y, r=6):
    return (x - r, y - r, x + r, y + r, 0.6)


def flight(start=(300.0, 200.0), rim=(120.0, 200.0), t0=10.0, dur=1.0, n=11):
    """Ball sightings from the shooter's hands to the rim, rising above the straight line by a parabola."""
    out = []
    for i in range(n):
        u = i / (n - 1)
        t = t0 + u * dur
        x = start[0] + (rim[0] - start[0]) * u
        y = start[1] + (rim[1] - start[1]) * u - 450 * dur * dur * u * (1 - u)
        out.append((t, x, y))
    return out


def looks_for(ball_path, people, t0=7.5, t1=11.3):
    out = []
    t = t0
    while t <= t1 + 1e-6:
        here = [ball_box(x, y) for tt, x, y in ball_path if abs(tt - t) < 0.051]
        out.append(Look(round(t, 2), MAP, list(people), here))
        t += 0.1
    return out


def test_a_ball_that_stays_put_in_the_gym_is_dropped():
    wall = [Look(1.0 + i * 0.1, MAP, [], [ball_box(500, 100)]) for i in range(8)]
    assert ball_sightings(wall) == []


def test_the_flight_is_traced_back_from_the_rim():
    path = flight()
    sightings = [(t, x, y, 0.6) for t, x, y in path]
    got = find_flight(sightings, [(120.0, 200.0)], end_t=11.0)
    assert len(got) == len(path) and got[0][0] == path[0][0]


def test_a_flight_must_end_at_the_rim():
    sightings = [(t, x, y, 0.6) for t, x, y in flight(rim=(800.0, 400.0))]
    assert find_flight(sightings, [(120.0, 200.0)], end_t=11.0) == []


def test_crowd_noise_does_not_join_the_flight():
    noise = [(10.2, 900.0, 40.0, 0.5), (10.4, 40.0, 700.0, 0.5), (10.6, 1000.0, 600.0, 0.5)]
    got = find_flight([(t, x, y, 0.6) for t, x, y in flight()] + noise, [(120.0, 200.0)], end_t=11.0)
    assert not any(p[1] in (900.0, 40.0, 1000.0) for p in got)


def test_the_shooter_is_the_person_holding_the_ball_at_the_release():
    fx, fy = px(15.0, 25.0)
    shooter = person_at(15.0, 25.0)
    path = flight(start=((shooter[0] + shooter[2]) / 2, shooter[1]), rim=(120.0, 150.0))
    other = person_at(40.0, 10.0)
    spot = locate(looks_for(path, [shooter, other]), [(120.0, 150.0)], arrival_t=11.0)
    assert spot.method == "flight" and abs(spot.x - 15.0) < 0.5 and abs(spot.y - 25.0) < 0.5


def test_no_flight_places_nothing_and_says_why():
    spot = locate(looks_for([], [person_at(15.0, 25.0)]), [(120.0, 150.0)], arrival_t=11.0)
    assert spot.x is None and spot.flags == ["no-flight"]


def test_a_person_standing_still_at_the_line_is_a_free_throw():
    assert free_throw_setup(looks_for([], [person_at(WIDTH / 2, FT_LINE_Y)]), 11.0) is True
    assert free_throw_setup(looks_for([], [person_at(10.0, 30.0)]), 11.0) is False


def test_someone_running_through_the_line_is_not_a_free_throw():
    looks = []
    for i, t in enumerate(np.arange(8.4, 10.2, 0.1)):
        looks.append(Look(float(t), MAP, [person_at(WIDTH / 2 - 3.2 + i * 0.4, FT_LINE_Y)], []))
    assert free_throw_setup(looks, 11.0) is False


def test_plus_one_on_the_board_is_a_free_throw_at_the_line():
    spot = shot_spot(looks_for([], []), [], None, 11.0, 1)
    assert (spot["x"], spot["y"], spot["method"], spot["shot_type"]) == (WIDTH / 2, FT_LINE_Y, "free-throw", "free throw")


def test_a_shot_from_the_far_end_lands_on_the_half_court():
    shooter = person_at(20.0, 94.0 - 25.0)
    rim_x, rim_y = px(25.0, 94.0 - 5.25)
    path = flight(start=((shooter[0] + shooter[2]) / 2, shooter[1]), rim=(rim_x, rim_y - 60))
    spot = shot_spot(looks_for(path, [shooter]), [(rim_x, rim_y - 60)], 11.0, 11.0, None)
    assert spot["method"] == "flight" and abs(spot["x"] - 30.0) < 0.6 and abs(spot["y"] - 25.0) < 0.6 and 0 < spot["conf"] <= 1


def test_lengths_and_widths_scale_the_rule():
    path = [(t, x * 2 / 3, y * 2 / 3) for t, x, y in flight()]
    got = find_flight([(t, x, y, 0.6) for t, x, y in path], [(80.0, 400 / 3)], end_t=11.0, scale=2 / 3)
    assert len(got) == len(path)


def test_a_ball_that_only_falls_loses_to_one_that_rose_first():
    up = [(t, x, y, 0.6) for t, x, y in flight(t0=9.2, dur=1.0)]
    falling = [(10.6 + i * 0.1, 120.0, 210.0 + 40 * i, 0.6) for i in range(9)]        # through the net, seen for longer
    got = find_flight(up + falling, [(120.0, 200.0)], end_t=10.6)
    assert got[0][0] == up[0][0]


def test_raised_arms_are_within_reach():
    from film.interpret.location import _reach

    box = (100.0, 200.0, 140.0, 300.0, 0.9)
    assert _reach(box, 120.0, 165.0) == 0.0
    assert _reach(box, 120.0, 100.0) > 0.0


def test_a_person_set_aside_is_gone_from_every_frame_they_can_be_followed_through():
    from film.interpret.location import set_aside

    walking = [(100.0 + i, 50.0, 124.0 + i, 110.0, 0.9) for i in range(6)]
    other = (400.0, 50.0, 424.0, 110.0, 0.9)
    looks = [Look(float(i), MAP, [walking[i], other], []) for i in range(6)]
    out = set_aside(looks, 3.0, walking[3])
    assert all(k.people == [other] for k in out)
    assert all(len(k.people) == 2 for k in looks)      # the caller's looks are left alone


def test_a_hidden_frame_does_not_lose_the_person():
    from film.interpret.location import set_aside

    box = (100.0, 50.0, 124.0, 110.0, 0.9)
    looks = [Look(0.0, MAP, [box], []), Look(0.1, MAP, [], []), Look(0.2, MAP, [box], [])]
    assert all(k.people == [] for k in set_aside(looks, 0.0, box))
