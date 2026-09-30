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
