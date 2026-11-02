"""Shots confirmed by the "Confirm all above 0.9" button rather than one at a time: runs of ten or more decisions in one minute. How many are on each game, and how many he skipped in review as unplaceable."""
from collections import Counter
from pathlib import Path

from film.interpret.events import load_events
from film.review import dotcheck, dotreview
from film.review.decisions import load_decisions

skips = Counter()
skip_ids = {(a["game_id"], a["event_id"]) for a in dotreview.read_answers() if a.get("skipped")}
tot = Counter()
for gd in sorted(p for p in Path("games").iterdir() if (p / "shot_looks.json.gz").exists()):
    dec = load_decisions(gd)
    ev = {e.id: e for e in load_events(gd)}
    conf = {i: d for i, d in dec.items() if d.get("status") == "confirmed" and i in ev and ev[i].signals.get("phase") != "warmup"}
    by_min = Counter(d.get("decided_at", "")[:16] for d in conf.values())
    bulk = {i for i, d in conf.items() if by_min[d.get("decided_at", "")[:16]] >= 10 and not d.get("dot_source") and d.get("x") is None}
    ever_bulk = {i for i, d in conf.items() if by_min[d.get("decided_at", "")[:16]] >= 10}
    sk = {i for i in conf if (gd.name, i) in skip_ids}
    misses = sum(not (conf[i].get("made", ev[i].made)) for i in sk)
    print(f"{gd.name.replace('2026-10-09-new-marcos-de-niza-high-school-vs-', '')[:34]:34s} confirmed {len(conf):3d} confirmed in a run {len(ever_bulk):3d} | skipped {len(sk):2d} of them in a run {len(sk & ever_bulk):2d} misses {misses:2d} | no dot now {sum(ev[i].x is None and conf[i].get('x') is None for i in conf):2d}")
    tot.update(c=len(conf), run=len(ever_bulk), sk=len(sk))
print(dict(tot))
