from pathlib import Path

import cv2

from film.perceive.detectors import load_detector
from film.perceive.rim_templates import TemplateRimDetector, load_rim_profile

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "frames"


def load(name):
    return cv2.imread(str(FIXTURES / name))


def test_profile_without_templates_gives_none():
    assert load_rim_profile("test") is None
    assert TemplateRimDetector.for_profile("test") is None


def test_left_basket_is_found_where_it_is():
    det = TemplateRimDetector.for_profile("hudl-tactical")
    assert det.name == "rim@template-hudl-tactical"
    rims = det.detect(load("hudl_tactical_t1500.jpg"))
    assert len(rims) == 1 and rims[0].cls == "rim" and rims[0].conf >= 0.8 and rims[0].side == "left"
    assert abs(rims[0].x1 - 378) <= 8 and abs(rims[0].y1 - 177) <= 8
    rims = det.detect(load("hudl_tactical_t4800.jpg"))
    assert len(rims) == 1 and rims[0].conf >= 0.95 and rims[0].side == "left"
    assert abs(rims[0].x1 - 455) <= 4 and abs(rims[0].y1 - 198) <= 4


def test_right_basket_is_found_where_it_is():
    det = TemplateRimDetector.for_profile("hudl-tactical")
    rims = det.detect(load("hudl_tactical_t3600.jpg"))
    assert len(rims) == 1 and rims[0].conf >= 0.9 and rims[0].side == "right"
    assert abs(rims[0].x1 - 1470) <= 4 and abs(rims[0].y1 - 186) <= 4


def test_frames_without_a_basket_give_nothing():
    det = TemplateRimDetector.for_profile("hudl-tactical")
    for name in ("hudl_tactical_t2400.jpg", "hudl_tactical_t600.jpg", "hudl_tactical_t60.jpg"):
        assert det.detect(load(name)) == [], name


def test_load_detector_prefers_templates_for_the_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path / "models"))
    det = load_detector("rim", "cpu", profile="hudl-tactical")
    assert isinstance(det, TemplateRimDetector)
