import numpy as np

from film.perceive.detectors import Box, FakeDetector
from film.perceive.rims import RimTracker, detect_rims, smooth_rims
from film.perceive.tracks import rows_to_frame


def rim(x, conf=0.9, cls="rim"):
    return Box(x, 100, x + 50, 130, conf, cls)


def test_tracker_keeps_id_while_panning_and_starts_new_far_away():
    tr = RimTracker()
    assert tr.update([rim(100)], 0) == [(1, rim(100))]
    assert tr.update([rim(130)], 1)[0][0] == 1          # 30 px pan, same rim
    assert tr.update([rim(160), rim(900)], 2) == [(1, rim(160)), (2, rim(900))]
    assert [tid for tid, _ in tr.update([rim(900)], 3)] == [2]


def test_tracker_expires_after_gap():
    tr = RimTracker(max_gap_frames=5)
    tr.update([rim(100)], 0)
    assert tr.update([rim(100)], 10)[0][0] == 2


def test_detect_rims_filters_and_caps():
    boxes = [rim(10, 0.5), rim(200, 0.9), rim(400, 0.7), Box(0, 0, 5, 5, 0.99, "backboard")]
    det = FakeDetector(per_call=[boxes])
    rows = detect_rims(np.zeros((4, 4, 3), np.uint8), det, RimTracker(), frame=3, t=0.3, model_version="rim@v1")
    assert [r["x1"] for r in rows] == [200, 400]
    assert all(r["kind"] == "rim" and r["model_version"] == "rim@v1" and r["frame"] == 3 for r in rows)
    assert rows[0]["track_id"] == 1 and rows[1]["track_id"] == 2


def test_smooth_rims_removes_a_spike():
    rows = []
    for f in range(9):
        b = rim(100 if f != 4 else 300)
        rows.append({"frame": f, "t": f / 10, "kind": "rim", "track_id": 1, "ref_id": None, "x1": b.x1, "y1": b.y1,
                     "x2": b.x2, "y2": b.y2, "conf": 0.9, "on_court": None, "court_x": None, "court_y": None,
                     "team": None, "model_version": "x"})
    df = smooth_rims(rows_to_frame(rows))
    assert df.loc[df["frame"] == 4, "x1"].item() == 100
    assert smooth_rims(rows_to_frame([])).empty


def test_template_boxes_keep_their_physical_id_across_gaps():
    left = Box(100, 100, 150, 130, 0.9, "rim", side="left")
    right = Box(1700, 100, 1750, 130, 0.8, "rim", side="right")
    det = FakeDetector(per_call=[[left], [right, left], [], [], [left.shifted(900, 0)]])
    tr = RimTracker(max_gap_frames=5, first_id=3)
    img = np.zeros((4, 4, 3), np.uint8)
    frames = [0, 1, 2, 3, 200]                                  # a long gap, then the left rim far across the frame
    ids = [[r["track_id"] for r in detect_rims(img, det, tr, frame=f, t=f / 10, model_version="rim@template")] for f in frames]
    assert ids == [[1], [1, 2], [], [], [1]]


def test_boxes_without_a_side_still_go_through_the_tracker_from_first_id():
    det = FakeDetector(per_call=[[rim(100)], [rim(100)]])
    tr = RimTracker(max_gap_frames=5, first_id=3)
    img = np.zeros((4, 4, 3), np.uint8)
    assert detect_rims(img, det, tr, frame=0, t=0.0, model_version="rim@v1")[0]["track_id"] == 3
    assert detect_rims(img, det, tr, frame=50, t=5.0, model_version="rim@v1")[0]["track_id"] == 4



def test_smooth_rims_treats_each_visit_separately():
    def row(f, x):
        b = rim(x)
        return {"frame": f, "t": f / 10, "kind": "rim", "track_id": 1, "ref_id": None, "x1": b.x1, "y1": b.y1,
                "x2": b.x2, "y2": b.y2, "conf": 0.9, "on_court": None, "court_x": None, "court_y": None,
                "team": None, "model_version": "x"}

    rows = [row(f, 100) for f in range(10)] + [row(110, 900), row(111, 900)]   # the camera panned back, 100 frames later
    df = smooth_rims(rows_to_frame(rows))
    assert df.loc[df["frame"] == 110, "x1"].item() == 900
    assert df.loc[df["frame"] == 9, "x1"].item() == 100
