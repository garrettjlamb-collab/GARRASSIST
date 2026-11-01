import numpy as np

from film.perceive.shooter_trust import FEATURES, fit_model, good_enough, judge, label_of


def _cand(on=False, x=10.0, y=30.0):
    f = [0.0] * len(FEATURES)
    if on:
        f[FEATURES.index("arms_ball")] = 1.0
    return {"f": f, "ab": f[0], "floor": (x, y)}


def test_the_paint_has_more_room_than_outside_it():
    assert good_enough((25.0, 8.0), (25.0, 13.0))               # 5 ft, in the paint
    assert not good_enough((25.0, 8.0), (25.0, 15.0))           # 7 ft, in the paint
    assert not good_enough((40.0, 20.0), (40.0, 25.0))          # 5 ft outside it
    assert good_enough((40.0, 30.0), (41.0, 31.0))
    assert not good_enough((25.0, 24.6), (25.0, 25.4))          # close, but the other side of the arc
    assert not good_enough((25.0, 8.0), None)


def test_the_label_is_the_candidate_on_his_dot_within_the_tolerance():
    cands = [_cand(x=25.0, y=8.0), _cand(x=40.0, y=30.0)]
    assert label_of(cands, (25.0, 12.0)) == 0                   # paint: 4 ft is within 6
    assert label_of(cands, (40.0, 35.0)) is None                # outside: 5 ft is over 4
    assert label_of([], (1.0, 1.0)) is None


def test_the_model_learns_the_feature_that_marks_the_shooter():
    shots = [{"c": [_cand(True), _cand(), _cand()], "label": 0} for _ in range(20)]
    w = fit_model(shots)
    assert w[FEATURES.index("arms_ball")] > 0.5


def test_a_dot_is_trusted_only_if_the_model_is_sure_and_lands_on_it():
    w = np.zeros(len(FEATURES))
    w[FEATURES.index("arms_ball")] = 5.0
    cands = [_cand(True, x=20.0, y=10.0), _cand(x=40.0, y=30.0)]
    assert judge(w, cands, (20.5, 10.5))["trusted"] is True
    assert judge(w, cands, (40.0, 30.0))["trusted"] is False    # sure, but of someone else
    assert judge(w * 0.0, cands, (20.5, 10.5))["trusted"] is False    # on it, but unsure
    assert judge(w, cands, None)["trusted"] is False
    assert judge(w, [], (1.0, 1.0)) is None
