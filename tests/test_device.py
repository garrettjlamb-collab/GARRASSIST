import sys
import types

from film.common.device import pick_device


def _fake_torch(mps: bool, cuda: bool):
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
    )


def test_explicit_preference_wins(monkeypatch):
    monkeypatch.setenv("FILM_DEVICE", "cpu")
    assert pick_device("mps") == "mps"


def test_env_variable_used(monkeypatch):
    monkeypatch.setenv("FILM_DEVICE", "cuda")
    assert pick_device() == "cuda"


def test_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv("FILM_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=False, cuda=False))
    assert pick_device() == "cpu"


def test_prefers_mps(monkeypatch):
    monkeypatch.delenv("FILM_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True, cuda=False))
    assert pick_device() == "mps"
