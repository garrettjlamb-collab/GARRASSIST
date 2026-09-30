from film.common import registry


def test_latest_needs_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path))
    assert registry.latest("rim") is None
    v1, d1 = registry.new_version_dir("rim")
    assert v1 == "v1" and d1.is_dir()
    assert registry.latest("rim") is None            # no best.pt yet
    (d1 / "best.pt").write_bytes(b"x")
    assert registry.latest("rim") == ("v1", d1)
    v2, d2 = registry.new_version_dir("rim")
    assert v2 == "v2"
    assert registry.latest("rim") == ("v1", d1)      # v2 has no weights
    (d2 / "best.pt").write_bytes(b"y")
    assert registry.latest("rim") == ("v2", d2)


def test_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path))
    _, d = registry.new_version_dir("ball")
    registry.write_manifest(d, name="ball", map50=0.91)
    info = registry.read_manifest(d)
    assert info["name"] == "ball" and info["map50"] == 0.91 and "written_at" in info


def test_base_weights_path_lives_under_models_base(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.chdir(tmp_path)
    path = registry.base_weights_path("yolo11n.pt")
    assert path == (tmp_path / "models" / "base" / "yolo11n.pt").resolve()
    assert path.is_absolute() and path.parent.is_dir() and not path.exists()


def test_base_weights_path_adopts_a_stray_download_in_the_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "yolov8s-worldv2.pt").write_bytes(b"w")
    path = registry.base_weights_path("yolov8s-worldv2.pt")
    assert path.read_bytes() == b"w" and not (tmp_path / "yolov8s-worldv2.pt").exists()
