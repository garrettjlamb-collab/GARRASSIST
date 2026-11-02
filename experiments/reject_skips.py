"""He said of the 60 shots he skipped on MDN vs Cactus Shadows that there was no shot in the clip: rejected, with what they were kept in a backup."""
from collections import Counter
from pathlib import Path

from film.export.shots import reviewed_shots
from film.review.decisions import load_decisions, save_decision
from film.review.dotreview import read_answers

g = "2026-10-08-new-marcos-de-niza-vs-cactus-shadows-angle-1"
gd = Path("games") / g
ids = [a["event_id"] for a in read_answers() if a["game_id"] == g and a.get("skipped")]
assert len(ids) == len(set(ids)) == 60, len(ids)
before = load_decisions(gd)
assert all(before[i]["status"] == "confirmed" and before[i].get("x") is None for i in ids)
rows = reviewed_shots(g)
print("before", len(rows), "made", sum(bool(e["made"]) for e in rows))
for i in ids:
    d = before[i]
    save_decision(gd, i, {"status": "rejected", **{k: d[k] for k in ("made", "team", "foul") if k in d}})
after = load_decisions(gd)
print(Counter(after[i]["status"] for i in ids), "others changed", sum(after[k] != before[k] for k in before if k not in ids))
rows = reviewed_shots(g)
made = sum(bool(e["made"]) for e in rows)
print("after", len(rows), "made", made, "of", len(rows), f"{made / len(rows):.0%}", "with a dot", sum(e.get("x") is not None for e in rows), "kept unseen", sum((e["decision"] or {}).get("dot_source") == "bulk" for e in rows))
