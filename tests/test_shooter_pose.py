import numpy as np

from film.interpret.court import Mapping
from film.interpret.shooter_pose import arms_and_ball, feet_on_floor, foot_pixel, picks

# Feet -> pixels at 10 px/ft: along the court runs right, across runs down (as in tests/test_location.py).
BACK = np.array([[0.0, 10.0, 50.0], [10.0, 0.0, 50.0], [0.0, 0.0, 1.0]])
MAP = Mapping(H=np.linalg.inv(BACK), points_used=8, error_px=1.0, conf=1.0, end="near", length=94.0)
at = lambda t: MAP


def person(x=300.0, foot=300.0, arms="up", lift=0.0):
    """Someone 100 px tall with their feet at (x, foot - lift), keypoints as COCO, all seen."""
    y = foot - lift
    kp = [[x, y - 90, 0.9] for _ in range(17)]
    kp[5], kp[6] = [x - 10, y - 75, 0.9], [x + 10, y - 75, 0.9]
    kp[11], kp[12] = [x - 8, y - 45, 0.9], [x + 8, y - 45, 0.9]
    wrist_y = y - 110 if arms == "up" else y - 50
    kp[9], kp[10] = [x - 6, wrist_y, 0.9], [x + 6, wrist_y, 0.9]
    kp[15], kp[16] = [x - 6, y, 0.9], [x + 6, y, 0.9]
    return {"box": [x - 15, y - 100, x + 15, y], "conf": 0.9, "kp": kp}


def test_arms_up_with_the_ball_at_the_hands_scores_high_and_nothing_else_does():
    ball = (300.0, 180.0)
    up = arms_and_ball(person(), ball)
    assert up > 0.7
    assert arms_and_ball(person(arms="down"), ball) < 0.1              # arms down
    assert arms_and_ball(person(), (900.0, 40.0)) < 0.01               # arms up, the ball elsewhere: a defender away from it


def test_unseen_shoulders_or_wrists_score_nothing():
    p = person()
    p["kp"][5][2] = 0.1
    assert arms_and_ball(p, (300.0, 180.0)) == 0.0
    q = person()
    q["kp"][9][2] = q["kp"][10][2] = 0.1
    assert arms_and_ball(q, (300.0, 180.0)) == 0.0


def test_the_foot_is_the_lower_ankle_or_else_the_bottom_of_the_box():
    p = person()
    p["kp"][15][1] += 4
    assert foot_pixel(p) == (300.0, 304.0)
    p["kp"][15][2] = p["kp"][16][2] = 0.1
    assert foot_pixel(p) == (300.0, 300.0)


def test_a_jump_shooters_feet_come_from_the_floor_not_the_air():
    # The camera is below the picture (image y grows toward it), so rising in the air reads as moving away from it.
    frames = [{"t": 9.8, "people": [person()]}, {"t": 9.9, "people": [person(lift=8.0)]}, {"t": 10.0, "people": [person(lift=15.0)]}]
    camera = MAP.to_court(300.0, 2000.0)
    assert feet_on_floor(frames, 2, frames[2]["people"][0], at, None) == MAP.to_court(300.0, 285.0)
    got = feet_on_floor(frames, 2, frames[2]["people"][0], at, camera)
    want = MAP.to_court(300.0, 300.0)
    assert abs(got[0] - want[0]) < 0.01 and abs(got[1] - want[1]) < 0.01


def _frames(*people):
    return [{"t": round(9.0 + i * 0.1, 1), "people": list(people)} for i in range(14)]


def test_the_picks_put_the_shooter_ahead_of_someone_under_the_flight():
    shooter, bystander = person(x=300.0), person(x=700.0, arms="down")
    # A flight that starts at the shooter's hands and passes over the bystander.
    flight = [(10.0 + i * 0.1, 300.0 + 80.0 * i, 190.0 - 40.0 * i + 4.5 * i * i, 0.6) for i in range(8)]
    got = picks(_frames(shooter, bystander), flight, MAP, at, 0.2)
    assert len(got) == 1 and got[0].box == tuple(shooter["box"][:4])
    assert got[0].score > 0.5


def test_no_one_with_their_arms_up_means_no_pick():
    flight = [(10.0 + i * 0.1, 300.0 + 80.0 * i, 190.0 - 40.0 * i + 4.5 * i * i, 0.6) for i in range(8)]
    assert picks(_frames(person(arms="down")), flight, MAP, at, 0.2) == []


def test_two_frames_of_one_person_are_one_pick():
    flight = [(10.0 + i * 0.1, 300.0 + 80.0 * i, 190.0 - 40.0 * i + 4.5 * i * i, 0.6) for i in range(8)]
    assert len(picks(_frames(person()), flight, MAP, at, 0.2)) == 1
