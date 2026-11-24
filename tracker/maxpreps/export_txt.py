"""Write a game's MaxPreps rows as MaxPreps' import file: pipe-delimited .txt, upload with Import Stats.

    python3 tracker/maxpreps/export_txt.py tracker/maxpreps/2026-11-23-alhambra.json

Format (maxpreps.com/utility/stat_import/field_specs.aspx, Boys Basketball): line 1 would be a stat
supplier ID (we have none: an unaccredited import leaves it out), then a header whose first field is
Jersey, then one line per player. Field names must match MaxPreps' exactly. An import never touches a
field the header leaves out, so Deflections and Charges (the stat book does not keep them) are not
declared and whatever is typed there by hand stays.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# rows JSON column -> MaxPreps field; order "Pts 2FG 2FGA 3FG 3FGA FT FTA Off Def Reb Ast Blk Stl Defl TO Chrg PF"
FIELDS = ["Points", "TwoPointsMade", "TwoPointAttempts", "ThreePointsMade", "ThreePointAttempts",
          "FreeThrowsMade", "FreeThrowAttempts", "OffensiveRebounds", "DefensiveRebounds", "Rebounds",
          "Assists", "BlockedShots", "Steals", "Deflections", "Turnovers", "Charges", "PersonalFouls"]
SKIP = {"Deflections", "Charges"}


def to_txt(doc: dict) -> str:
    keep = [i for i, f in enumerate(FIELDS) if f not in SKIP]
    tf = doc.get("technicals", {})           # jersey -> technical fouls, only when there were any
    head = ["Jersey"] + [FIELDS[i] for i in keep] + (["TechnicalFouls"] if tf else [])
    lines = ["|".join(head)]
    for jersey, row in doc["rows"].items():
        if len(row) != len(FIELDS):
            raise SystemExit(f"#{jersey}: {len(row)} values, expected {len(FIELDS)}")
        vals = [str(row[i]) for i in keep] + ([str(tf.get(jersey, 0))] if tf else [])
        lines.append("|".join([jersey] + vals))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    src = Path((argv or sys.argv[1:])[0])
    out = src.with_suffix(".maxpreps.txt")
    out.write_text(to_txt(json.loads(src.read_text())))
    print(out)
    print(out.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
