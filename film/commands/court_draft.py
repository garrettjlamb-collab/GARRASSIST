# film/commands/court_draft.py
from __future__ import annotations

import json


def register(sub) -> None:
    parser = sub.add_parser("court-draft", help="draw the court on queued pictures for the coach to approve in Label court")
    what = parser.add_subparsers(dest="action", required=True)
    what.add_parser("list", help="the queued court pictures and who has drawn each")
    view = what.add_parser("view", help="write a picture, or a piece of it, enlarged with a numbered pixel grid")
    view.add_argument("image_id")
    view.add_argument("--region", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None)
    view.add_argument("--scale", type=float, default=1.0)
    put = what.add_parser("set", help="save marks as the picture's draft and write a picture of the court they give")
    put.add_argument("image_id")
    put.add_argument("--points", required=True, help='JSON: {"near-lane-left": [800, 425], ...}')
    put.add_argument("--length", type=float, default=None, help="the court's length in feet (74, 84, 94)")
    put.add_argument("--by", default="claude")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.perceive import court_draft

    if args.action == "list":
        for p in court_draft.pending_pictures():
            drawn = f"{p['points']} marks by {p['by']}, {p['length']:g} ft" if p["by"] else "not drawn"
            print(f"{p['image_id']}  {drawn}")
    elif args.action == "view":
        print(court_draft.view(args.image_id, region=args.region, scale=args.scale))
    else:
        try:
            got = court_draft.set_draft(args.image_id, json.loads(args.points), length=args.length, by=args.by)
        except ValueError as err:
            print(f"not saved: {err}")
            return 1
        print(f"saved {len(got['kept'])} marks at {got['length']:g} ft, {got['error_px']} px apart on average")
        if got["dropped"]:
            print(f"left out, look again: {', '.join(got['dropped'])}")
        print(got["picture"])
    return 0
