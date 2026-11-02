"""MDN vs CS: he skipped 60 of 231 shots, most with no shot in the clip. Who are they: bulk confirmed, what signals, and did the machine see a flight? A strip of some of them."""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from film.export.clips import clip_path
from film.interpret.events import load_events
from film.review.decisions import load_decisions
from film.review.dotreview import read_answers

g = "2026-10-08-new-marcos-de-niza-vs-cactus-shadows-angle-1"
gd = Path("games") / g
dec = load_decisions(gd)
ev = {e.id: e for e in load_events(gd)}
skips = [a["event_id"] for a in read_answers() if a["game_id"] == g and a.get("skipped")]
live = [e for e in ev.values() if (dec.get(e.id) or {}).get("status") == "confirmed" and e.signals.get("phase") != "warmup"]
others = [e for e in live if e.id not in skips]
when = Counter(dec[i].get("decided_at", "")[:16] for i in skips)
print("skipped", len(skips), "decided at, top minutes", when.most_common(4), "| others top", Counter(dec[e.id].get("decided_at", "")[:16] for e in others).most_common(3))
def desc(name, es):
    print(name, len(es), "method", Counter((e.signals.get("location") or {}).get("method") for e in es).most_common(), "source", Counter(e.signals.get("source") for e in es).most_common(), "conf median", round(float(np.median([e.confidence for e in es])), 2), "agreement", Counter(e.signals.get("agreement") for e in es).most_common(3), "moved_from", sum("moved_from" in dec[e.id] for e in es), "made", sum(bool(e.made) for e in es), "flags", Counter(f for e in es for f in e.flags).most_common(4))
desc("SKIPPED", [ev[i] for i in skips])
desc("OTHERS ", others)
print("no-flight among skipped", sum("no-flight" in ev[i].flags for i in skips), "no-flight among others", sum("no-flight" in e.flags for e in others))
print("clip vs t, skipped:", Counter(round(ev[i].t - ev[i].clip["start"], 0) for i in skips).most_common(3), "others", Counter(round(e.t - e.clip["start"], 0) for e in others).most_common(3))
print("t range minutes", round(min(ev[i].t for i in skips) / 60), round(max(ev[i].t for i in skips) / 60))
tiles = []
rng = np.random.default_rng(3)
for i in rng.choice(skips, min(5, len(skips)), replace=False):
    e = ev[str(i)]
    cap = cv2.VideoCapture(str(clip_path(gd, e.id)))
    for off in (-4, -2, -1, 0, 1, 2):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, e.t + off - e.clip["start"]) * 1000)
        ok, f = cap.read()
        f = cv2.resize(f, (320, 180)) if ok else np.zeros((180, 320, 3), np.uint8)
        cv2.putText(f, f"{e.id[5:]} {off:+d}s", (4, 16), 0, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(f)
    cap.release()
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
