
import numpy as np

from film.interpret.court import Mapping
from film.perceive import official, shotspots
from film.perceive.official import fit_linear, held_out_scores, predict_linear

# Feet -> pixels at 10 px/ft, as in tests/test_location.py.
BACK = np.array([[0.0, 10.0, 50.0], [10.0, 0.0, 50.0], [0.0, 0.0, 1.0]])
MAP = Mapping(H=np.linalg.inv(BACK), points_used=8, error_px=1.0, conf=1.0, end="near", length=94.0)


def _features(n, seed=3):
    rng = np.random.default_rng(seed)
    y = np.array([i % 5 == 0 for i in range(n)], dtype=np.float32)
    X = rng.normal(size=(n, 16))
    X[:, 0] += 4.0 * y
    return X, y


def test_the_linear_model_separates_what_is_separable():
    X, y = _features(100)
    p = predict_linear(fit_linear(X, y), X)
    assert (p[y > 0] > 0.5).mean() > 0.9 and (p[y == 0] < 0.5).mean() > 0.9


def test_held_out_scores_come_from_a_model_that_never_saw_the_game(monkeypatch):
    X, y = _features(60)
    games = np.array(["a", "b", "c"] * 20)
    seen = []
    real = official.fit_linear

    def spy(Xs, ys):
        seen.append(len(Xs))
        return real(Xs, ys)

    monkeypatch.setattr(official, "fit_linear", spy)
    assert len(held_out_scores(X, y, games)) == 60
    assert seen == [40, 40, 40]


def _flight_row():
    # Ball rises from (300, 200) to the rim at (120, 200) over a second; a referee stands at the start, a player right behind.
    frames, ref, player = [], (285.0, 195.0, 315.0, 275.0, 0.9), (283.0, 170.0, 317.0, 250.0, 0.9)
    for i in range(38):
        t = round(7.5 + i * 0.1, 2)
        u = (t - 10.0) / 1.0
        ball = []
        if 0 <= u <= 1:
            x, y = 300 - 180 * u, 200 - 450 * u * (1 - u)
            ball = [[x - 6, y - 6, x + 6, y + 6, 0.6]]
        frames.append({"t": t, "people": [list(ref), list(player)], "ball": ball})
    return {"id": "s", "t": 11.0, "rim_id": 1, "frames": frames, "rims": [[11.0, 1, 110.0, 195.0, 130.0, 205.0]]}, ref, player


def _looks(row):
    from film.interpret.location import Look

    return [Look(f["t"], MAP, [tuple(b) for b in f["people"]], [tuple(b) for b in f["ball"]]) for f in row["frames"]]


def _verdict(row, score_of_box):
    """Scores already kept on the shot, the way a run with the film leaves them."""
    row["officials"] = {shotspots._person_key(f["t"], b): score_of_box(b) for f in row["frames"] for b in f["people"]}


def test_a_chosen_non_player_is_set_aside_and_the_shooter_chosen_again():
    row, ref, player = _flight_row()
    _verdict(row, lambda b: 0.9 if b[1] == ref[1] else 0.1)
    spot = shotspots.spot_of_a_player(_looks(row), row, 11.0, 11.0, None, 1.0, None, None)
    assert spot["method"] == "flight" and spot["set_aside"] == 1 and "official-set-aside" in spot["flags"]
    assert spot["box"][1] == player[1]


def test_with_no_model_and_nothing_kept_the_first_choice_stands():
    row, ref, _ = _flight_row()
    spot = shotspots.spot_of_a_player(_looks(row), row, 11.0, 11.0, None, 1.0, None, None)
    assert spot["set_aside"] == 0 and spot["box"][1] == ref[1]


def test_when_everyone_looks_like_a_non_player_there_is_no_dot():
    row, _, _ = _flight_row()
    _verdict(row, lambda b: 0.9)
    spot = shotspots.spot_of_a_player(_looks(row), row, 11.0, 11.0, None, 1.0, None, None)
    assert spot["x"] is None and spot["box"] is None


def _posed(row, arms_of):
    """Poses for _flight_row's two people: COCO keypoints with arms as `arms_of(box)` says."""
    out = []
    for f in row["frames"]:
        people = []
        for b in f["people"]:
            x, top, bottom = (b[0] + b[2]) / 2, b[1], b[3]
            kp = [[x, top + 5, 0.9] for _ in range(17)]
            kp[5], kp[6] = [x - 8, top + 15, 0.9], [x + 8, top + 15, 0.9]
            kp[11], kp[12] = [x - 6, top + 40, 0.9], [x + 6, top + 40, 0.9]
            wy = top - 5 if arms_of(b) else top + 45
            kp[9], kp[10] = [x - 5, wy, 0.9], [x + 5, wy, 0.9]
            kp[15], kp[16] = [x - 5, bottom, 0.9], [x + 5, bottom, 0.9]
            people.append({"box": list(b[:4]), "conf": 0.9, "kp": kp})
        out.append({"t": f["t"], "people": people})
    return out


def _maps():
    return [(i * 1.0, MAP) for i in range(30)]


def test_the_player_with_arms_up_takes_the_dot_from_the_rules():
    row, ref, player = _flight_row()
    poses = _posed(row, lambda b: b[1] == player[1])
    spot = shotspots.spot_of_a_player(_looks(row), row, 11.0, 11.0, None, 1.0, None, None, size=(1920, 1080), poses=poses, maps=_maps())
    assert spot["method"] == "pose" and spot["box"][1] == player[1] and spot["pose_score"] >= shotspots.POSE_MIN_SCORE


def test_no_arms_up_leaves_the_rules_choice():
    row, ref, _ = _flight_row()
    poses = _posed(row, lambda b: False)
    spot = shotspots.spot_of_a_player(_looks(row), row, 11.0, 11.0, None, 1.0, None, None, size=(1920, 1080), poses=poses, maps=_maps())
    assert spot["method"] != "pose"


def test_a_non_player_with_arms_up_is_passed_over():
    row, ref, player = _flight_row()
    moved = [player[0] - 45.0, player[1], player[2] - 45.0, player[3], player[4]]     # beside the referee, not on top of him
    for f in row["frames"]:
        f["people"][1] = moved
    poses = _posed(row, lambda b: True)
    row["officials"] = {}
    for f in poses:
        for p in f["people"]:
            row["officials"][shotspots._person_key(f["t"], (*p["box"], 1.0))] = 0.9 if p["box"][1] == ref[1] else 0.1
    spot = shotspots.spot_of_a_player(_looks(row), row, 11.0, 11.0, None, 1.0, None, None, size=(1920, 1080), poses=poses, maps=_maps())
    assert spot["method"] == "pose" and spot["box"][1] == player[1]
