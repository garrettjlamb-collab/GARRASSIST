import sys
from pathlib import Path
from film.common.paths import game_dir
from film.interpret.events import load_events
from film.interpret.zones import is_three
for e in load_events(game_dir(sys.argv[1])):
    if e.x is not None and e.made and e.signals.get("points") in (2, 3) and e.signals["location"]["method"] == "flight" and is_three(e.x, e.y) != (e.signals["points"] == 3):
        l = e.signals["location"]
        print(e.id, "pts", e.signals["points"], "dot", e.x, e.y, "rel", round(l["release_t"] - e.t, 1), "flight", l["flight_points"], "reach", l["reach_px"])
