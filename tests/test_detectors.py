import types

import numpy as np
import pytest

from film.perceive import detectors as D
from film.perceive.detectors import Box, FakeDetector, YoloDetector, boxes_from_result, load_detector


def _result(rows, names):
    xyxy = [r[:4] for r in rows]
    return types.SimpleNamespace(
        names=names,
        boxes=types.SimpleNamespace(
            xyxy=types.SimpleNamespace(tolist=lambda: xyxy),
            conf=types.SimpleNamespace(tolist=lambda: [r[4] for r in rows]),
            cls=types.SimpleNamespace(tolist=lambda: [r[5] for r in rows]),
        ),
    )


def test_box_geometry():
    b = Box(10, 20, 30, 60, 0.9, "rim")
    assert (b.w, b.h, b.cx, b.cy) == (20, 40, 20, 40)
    s = b.shifted(5, -5)
    assert (s.x1, s.y1, s.x2, s.y2, s.cls) == (15, 15, 35, 55, "rim")


def test_fake_detector_per_call_and_fn():
    a, b = Box(0, 0, 1, 1, 1, "rim"), Box(1, 1, 2, 2, 1, "rim")
    d = FakeDetector(per_call=[[a], [b]])
    img = np.zeros((4, 4, 3), np.uint8)
    assert d.detect(img) == [a] and d.detect(img) == [b] and d.detect(img) == [b]
    f = FakeDetector(fn=lambda image, i: [a] if i % 2 == 0 else [])
    assert f.detect(img) == [a] and f.detect(img) == [] and f.calls == 2


def test_boxes_from_result_maps_class_names():
    res = _result([[1, 2, 3, 4, 0.8, 0], [5, 6, 7, 8, 0.4, 1]], names={0: "basketball hoop", 1: "backboard"})
    boxes = boxes_from_result(res, cls_map={"basketball hoop": "rim"})
    assert boxes == [Box(1, 2, 3, 4, 0.8, "rim"), Box(5, 6, 7, 8, 0.4, "backboard")]


def test_yolo_detector_uses_model_predict():
    seen = {}

    def predict(image, **kw):
        seen.update(kw)
        return [_result([[0, 0, 10, 10, 0.9, 0]], names={0: "ball"})]

    det = YoloDetector(types.SimpleNamespace(predict=predict), device="cpu", imgsz=320, conf=0.3, name="ball@v1")
    assert det.detect(np.zeros((8, 8, 3), np.uint8)) == [Box(0, 0, 10, 10, 0.9, "ball")]
    assert seen["imgsz"] == 320 and seen["conf"] == 0.3 and seen["device"] == "cpu" and seen["verbose"] is False


def test_yolo_detector_detects_a_batch_in_one_pass():
    calls = []

    def predict(images, **kw):
        calls.append(len(images))
        return [_result([[0, 0, 10, 10, 0.9, 0]], names={0: "ball"}), _result([], names={0: "ball"})]

    det = YoloDetector(types.SimpleNamespace(predict=predict), device="cpu", imgsz=320, name="ball@v1")
    blank = np.zeros((8, 8, 3), np.uint8)
    assert det.detect_many([blank, blank]) == [[Box(0, 0, 10, 10, 0.9, "ball")], []]
    assert calls == [2] and det.detect_many([]) == [] and calls == [2]


def test_load_detector_prefers_trained_then_falls_back(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(D.registry, "latest", lambda name: ("v3", tmp_path) if name == "rim" else None)
    monkeypatch.setattr(YoloDetector, "from_weights", classmethod(lambda cls, w, device, imgsz, conf=0.25, name="": calls.append(("weights", str(w), imgsz, name)) or "trained"))
    monkeypatch.setattr(YoloDetector, "open_vocab", classmethod(lambda cls, classes, device, imgsz, conf=0.15, name="", weights="": calls.append(("world", classes, imgsz, name)) or "world"))
    assert load_detector("rim", "cpu") == "trained"
    assert load_detector("ball", "cpu") == "world"
    assert calls[0] == ("weights", str(tmp_path / "best.pt"), 640, "rim@v3")
    assert calls[1] == ("world", D.BALL_WORLD_CLASSES, 640, "ball@world-fallback")     # the 320 px crop, enlarged 2x


def test_load_detector_falls_back_to_open_vocab_without_templates(monkeypatch):
    calls = []
    monkeypatch.setattr(D.registry, "latest", lambda name: None)
    monkeypatch.setattr(YoloDetector, "open_vocab", classmethod(lambda cls, classes, device, imgsz, conf=0.15, name="", weights="": calls.append(name) or "world"))
    assert load_detector("rim", "cpu") == "world"
    assert load_detector("rim", "cpu", profile="test") == "world"
    assert calls == ["rim@world-fallback", "rim@world-fallback"]


def test_open_vocab_loads_world_weights_from_models_base(monkeypatch, tmp_path):
    import ultralytics

    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.chdir(tmp_path)              # never adopt the repo's own stray weights into a temp dir
    seen = {}

    class StubWorld:
        def __init__(self, weights):
            seen["weights"] = weights

        def set_classes(self, classes):
            seen["classes"] = classes

    monkeypatch.setattr(ultralytics, "YOLOWorld", StubWorld)
    det = YoloDetector.open_vocab(D.BALL_WORLD_CLASSES, "cpu", 320)
    assert seen["weights"] == str(tmp_path / "models" / "base" / D.WORLD_WEIGHTS)
    assert det.cls_map == D.BALL_WORLD_CLASSES


def test_film_rim_detector_env_picks_templates_or_trained(monkeypatch, tmp_path):
    from film.perceive.rim_templates import TemplateRimDetector

    monkeypatch.setattr(D.registry, "latest", lambda name: ("v2", tmp_path) if name == "rim" else None)
    monkeypatch.setattr(YoloDetector, "from_weights", classmethod(lambda cls, w, device, imgsz, conf=0.25, name="": "trained"))
    monkeypatch.setattr(YoloDetector, "open_vocab", classmethod(lambda cls, classes, device, imgsz, conf=0.15, name="", weights=None: "world"))
    monkeypatch.delenv("FILM_RIM_DETECTOR", raising=False)
    # A camera with templates keeps them: they were measured on that camera, and a trained rim model is
    # trained on other cameras (it boxes backboards, not the template's strip).
    assert isinstance(load_detector("rim", "cpu", profile="hudl-tactical"), TemplateRimDetector)
    assert load_detector("rim", "cpu", profile="nfhs-auto") == "trained"
    monkeypatch.setenv("FILM_RIM_DETECTOR", "template")
    assert isinstance(load_detector("rim", "cpu", profile="hudl-tactical"), TemplateRimDetector)
    assert load_detector("rim", "cpu", profile="test") == "trained"              # no templates for this profile: the ladder
    assert load_detector("ball", "cpu") == "world"                                 # the ball is unaffected
    monkeypatch.setenv("FILM_RIM_DETECTOR", "trained")
    assert load_detector("rim", "cpu", profile="hudl-tactical") == "trained"
    monkeypatch.setenv("FILM_RIM_DETECTOR", "templates")
    with pytest.raises(ValueError, match="FILM_RIM_DETECTOR"):
        load_detector("rim", "cpu", profile="hudl-tactical")


def test_a_trained_model_runs_at_the_size_it_was_trained_at(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path / "models"))
    from film.common import registry
    for version, size in (("v1", 320), ("v2", None)):
        folder = tmp_path / "models" / "ball" / version
        folder.mkdir(parents=True)
        (folder / "best.pt").write_bytes(b"w")
        registry.write_manifest(folder, name="ball", version=version, **({"imgsz": size} if size else {}))
    seen = []
    monkeypatch.setattr(YoloDetector, "from_weights", classmethod(lambda cls, w, device, imgsz, conf=0.25, name="": seen.append((name, imgsz)) or "trained"))
    load_detector("ball", "cpu")
    assert seen == [("ball@v2", D.IMGSZ["ball"])]              # no size recorded: the current default
    (tmp_path / "models" / "ball" / "v2" / "best.pt").unlink()
    load_detector("ball", "cpu")
    assert seen[-1] == ("ball@v1", 320)                        # an older model keeps its own size
