import pytest

from film.review.decisions import apply_decisions, load_decisions, save_decision


def test_save_and_apply(tmp_path):
    assert load_decisions(tmp_path) == {}
    entry = save_decision(tmp_path, "shot-1", {"status": "confirmed", "made": False})
    assert entry["status"] == "confirmed" and entry["made"] is False and "decided_at" in entry
    save_decision(tmp_path, "shot-2", {"status": "rejected"})
    decisions = load_decisions(tmp_path)
    assert set(decisions) == {"shot-1", "shot-2"}
    events = [{"id": "shot-1", "made": True, "team": "home"}, {"id": "shot-2", "made": None, "team": None}, {"id": "shot-3", "made": True, "team": None}]
    merged = apply_decisions(events, decisions)
    assert merged[0]["made"] is False and merged[0]["team"] == "home" and merged[0]["decision"]["status"] == "confirmed"
    assert merged[1]["decision"]["status"] == "rejected"
    assert merged[2]["decision"] is None and merged[2]["made"] is True


def test_bad_status_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_decision(tmp_path, "shot-1", {"status": "maybe"})


def test_save_is_atomic_and_threadsafe(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(20) as pool:
        list(pool.map(lambda i: save_decision(tmp_path, f"shot-{i}", {"status": "unsure"}), range(20)))
    assert set(load_decisions(tmp_path)) == {f"shot-{i}" for i in range(20)}
    assert not list(tmp_path.glob("*.tmp"))
    import json
    json.loads((tmp_path / "review.json").read_text())
