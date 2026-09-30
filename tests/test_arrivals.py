import numpy as np
import pandas as pd

from film.interpret.arrivals import Arrival, find_ball_arrivals, find_motion_arrivals, merge_arrivals
from film.perceive.tracks import rows_to_frame

RIM = (900.0, 500.0, 950.0, 530.0)
FPS = 10.0


def rim_rows(frames, rim=RIM, track_id=1):
    return [{"frame": f, "t": f / FPS, "kind": "rim", "track_id": track_id, "ref_id": None, "x1": rim[0], "y1": rim[1],
             "x2": rim[2], "y2": rim[3], "conf": 0.9, "on_court": None, "court_x": None, "court_y": None,
             "team": None, "model_version": "x"} for f in frames]


def ball_rows(points, rim_id=1):
    """points: iterable of (frame, cx, cy) ball centres, 12 px boxes."""
    return [{"frame": f, "t": f / FPS, "kind": "ball", "track_id": None, "ref_id": rim_id, "x1": cx - 6, "y1": cy - 6,
             "x2": cx + 6, "y2": cy + 6, "conf": 0.7, "on_court": None, "court_x": None, "court_y": None,
             "team": None, "model_version": "x"} for f, cx, cy in points]


def arc(start_frame, cx=925.0, top=300.0, bottom=560.0, n=12):
    ys = np.linspace(top, bottom, n)
    return [(start_frame + i, cx, float(y)) for i, y in enumerate(ys)]


def test_ball_crossing_the_rim_line_is_one_arrival():
    rims = rows_to_frame(rim_rows(range(0, 60)))
    balls = rows_to_frame(ball_rows(arc(0)))
    found = find_ball_arrivals(balls, rims, FPS)
    assert len(found) == 1
    a = found[0]
    assert a.source == "ball" and a.rim_id == 1 and a.rim_box == RIM
    assert 0.8 <= a.t <= 1.1                       # crosses y=500 between frames 8 and 9


def test_pass_under_the_rim_and_ball_rising_are_not_arrivals():
    rims = rows_to_frame(rim_rows(range(0, 60)))
    under = rows_to_frame(ball_rows([(f, 700.0 + 20 * f, 560.0) for f in range(12)]))
    assert find_ball_arrivals(under, rims, FPS) == []
    rising = rows_to_frame(ball_rows([(f, 925.0, 560.0 - 25 * f) for f in range(12)]))
    assert find_ball_arrivals(rising, rims, FPS) == []
    wide = rows_to_frame(ball_rows([(f, 700.0, y) for f, _, y in arc(0)]))   # far to the left of the rim
    assert find_ball_arrivals(wide, rims, FPS) == []


def test_min_gap_and_repeat_arrivals():
    rims = rows_to_frame(rim_rows(range(0, 100)))
    close = rows_to_frame(ball_rows(arc(0) + arc(8)))          # second crossing within 1.5 s
    assert len(find_ball_arrivals(close, rims, FPS)) == 1
    apart = rows_to_frame(ball_rows(arc(0) + arc(40)))         # 4 s later
    assert len(find_ball_arrivals(apart, rims, FPS)) == 2


def test_detection_gap_breaks_the_crossing():
    rims = rows_to_frame(rim_rows(range(0, 60)))
    pts = arc(0)
    pts = [p for p in pts if p[0] not in (6, 7, 8, 9)]         # ball lost for four frames around the rim line
    assert find_ball_arrivals(rows_to_frame(ball_rows(pts)), rims, FPS) == []


def test_motion_peak_becomes_a_low_confidence_arrival():
    rims = rows_to_frame(rim_rows(range(0, 200)))
    energy = np.full(200, 2.0)
    energy[120] = 40.0
    motion = pd.DataFrame({"frame": range(200), "t": np.arange(200) / FPS, "rim_id": 1, "energy": energy, "energy_global": 1.0})
    found = find_motion_arrivals(motion, rims, FPS)
    assert len(found) == 1 and found[0].source == "motion" and found[0].t == 12.0 and found[0].conf < 0.5


def test_merge_keeps_ball_arrivals_and_adds_distant_motion_ones():
    ball = [Arrival(t=10.0, frame=100, rim_id=1, rim_box=RIM, source="ball", conf=0.8)]
    motion = [Arrival(t=11.0, frame=110, rim_id=1, rim_box=RIM, source="motion", conf=0.4),
              Arrival(t=30.0, frame=300, rim_id=1, rim_box=RIM, source="motion", conf=0.4),
              Arrival(t=11.0, frame=110, rim_id=2, rim_box=RIM, source="motion", conf=0.4)]
    merged = merge_arrivals(ball, motion)
    assert [(a.t, a.rim_id, a.source) for a in merged] == [(10.0, 1, "ball"), (11.0, 2, "motion"), (30.0, 1, "motion")]
