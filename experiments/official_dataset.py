"""Put the by-eye labelled shooter crops (experiments/shooter_sheets.py) into datasets/official as film train official reads them."""
import json
import shutil
import sys
from pathlib import Path

from film.common.paths import datasets_dir

OFFICIAL = {15, 36, 71, 74, 143, 216, 235, 242, 243, 251, 258, 265, 274, 283, 285, 287, 289, 296, 386, 392, 395, 401, 404, 416, 417, 419, 427, 450, 485, 524}
UNSURE = {83, 118, 403, 489, 496}
d = Path(sys.argv[1])
root = datasets_dir() / "official"
for r in json.loads((d / "rows.json").read_text()):
    if r["n"] in UNSURE:
        continue
    to = root / ("official" if r["n"] in OFFICIAL else "player")
    to.mkdir(parents=True, exist_ok=True)
    shutil.copy(d / f"{r['n']:04d}.jpg", to / f"{r['game']}--{r['id']}.jpg")
print(len(list((root / "official").glob("*.jpg"))), "non-players", len(list((root / "player").glob("*.jpg"))), "players")
