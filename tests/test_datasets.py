import numpy as np

from film.perceive.detectors import Box
from film.train import datasets as ds


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FILM_LABELS_DIR", str(tmp_path / "labels"))


def test_pending_roundtrip(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    img = np.zeros((100, 200, 3), np.uint8)
    ds.write_pending("rim", "g_00010", img, [Box(10, 20, 30, 40, 0.5, "rim")], extra={"t": 10.0})
    assert ds.list_pending("rim") == ["g_00010"]
    data = ds.read_pending("rim", "g_00010")
    assert data["width"] == 200 and data["height"] == 100 and data["t"] == 10.0
    assert data["boxes"] == [{"cls": "rim", "x1": 10, "y1": 20, "x2": 30, "y2": 40, "conf": 0.5}]
    ds.remove_pending("rim", "g_00010")
    assert ds.list_pending("rim") == [] and ds.list_pending("nothing") == []


def test_yolo_line_and_split():
    assert ds.yolo_line(1, Box(10, 20, 30, 40, 0.5, "backboard"), 200, 100) == "1 0.100000 0.300000 0.100000 0.200000"
    assert [ds.split_for(i) for i in range(6)] == ["val", "train", "train", "train", "train", "val"]


def test_accept_boxes_writes_yolo_dataset(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    img = np.zeros((100, 200, 3), np.uint8)
    for i in range(3):
        ds.write_pending("rim", f"g_{i}", img, [])
    split0 = ds.accept_boxes("rim", "g_0", [Box(10, 20, 30, 40, 0.5, "rim"), Box(0, 0, 5, 5, 0.9, "person")])
    assert split0 == "val"
    root = tmp_path / "datasets" / "rim"
    assert (root / "images" / "val" / "g_0.jpg").exists()
    assert (root / "labels" / "val" / "g_0.txt").read_text() == "0 0.100000 0.300000 0.100000 0.200000\n"
    assert ds.accept_boxes("rim", "g_1", []) == "train"
    assert (root / "labels" / "train" / "g_1.txt").read_text() == ""
    assert ds.count_accepted("rim") == 2 and ds.list_pending("rim") == ["g_2"]
    yaml = ds.data_yaml("rim").read_text()
    assert "train: images/train" in yaml and "0: rim" in yaml and "1: backboard" in yaml


def test_class_dataset_guarantees_val_has_each_class(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    img = np.zeros((288, 288, 3), np.uint8)
    p1 = ds.store_class_image("rimclf", "a", img, "made")
    assert p1.parts[-3:] == ("val", "made", "a.jpg")            # first of its class lands in val
    assert ds.store_class_image("rimclf", "b", img, "made").parts[-3] == "train"
    ds.write_pending("rimclf", "c", img, [], extra={"suggested": "missed"})
    assert ds.accept_class("rimclf", "c", "missed") == "val"
    assert ds.list_pending("rimclf") == []
