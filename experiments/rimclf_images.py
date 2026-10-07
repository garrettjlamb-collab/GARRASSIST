"""Experiment: can a classifier that looks at the pictures beat the six ball-path features?

    uv run python experiments/rimclf_images.py cut                 # one sheet of rim pictures per labeled clip
    uv run python experiments/rimclf_images.py score frames        # each game held out in turn
    uv run python experiments/rimclf_images.py score frames --hold rkda,marcos --epochs 8

Sheets go to datasets/rimclf-images (git-ignored). Every labeled item of datasets/rimclf with a stored ball
path is used; labels, games and basket sides come from there, so the comparison with the path classifier is
on the very same clips. Nothing here is used by the pipeline: it only prints scores.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from film.common.manifest import Manifest
from film.common.paths import datasets_dir, game_dir
from film.common.video import iter_frames
from film.interpret.ballpath import hoop_offset
from film.perceive.tracks import read_tracks
from film.train.montage import rim_follower
from film.train.rimclf import CLASSES
from film.train.rimclf_dataset import rim_box_at

FPS = 15.0
LEAD_S, N_FRAMES = 2 / 15, 16            # from two frames before the arrival to 0.93 s after it
TILE, GRID = 112, 4
VIEW_WIDTHS = 1.6                        # the crop is this many rim-box widths across, centred under the hoop
DROP_WIDTHS = 0.35                       # ... and this far below the rim box's centre


def labeled_items() -> list[dict]:
    root = datasets_dir() / "rimclf"
    paths = {p.stem: json.loads(p.read_text()) for p in sorted((root / "paths").glob("*.json"))}
    by_basket = defaultdict(list)
    for p in paths.values():
        by_basket[(p["game"], p.get("side") or p["rim"])].append(p)
    offsets = {k: hoop_offset(v) for k, v in by_basket.items()}
    items = []
    for split in ("train", "val"):
        for k, cls in enumerate(CLASSES):
            for image in sorted((root / split / cls).glob("*.jpg")):
                p = paths.get(image.stem)
                if p is not None:
                    items.append({"id": image.stem, "y": k, "game": p["game"], "rim": p["rim"], "t": p["t"],
                                  "hoop_dx": offsets[(p["game"], p.get("side") or p["rim"])]})
    return items


def sheet_path(item_id: str) -> Path:
    return datasets_dir() / "rimclf-images" / f"{item_id}.jpg"


def cut(limit: int | None = None) -> None:
    items = labeled_items()
    todo = [it for it in items if not sheet_path(it["id"]).exists()][:limit]
    sheet_path("x").parent.mkdir(parents=True, exist_ok=True)
    games: dict[str, tuple] = {}
    started = time.time()
    for n, it in enumerate(todo, 1):
        if it["game"] not in games:
            gd = game_dir(it["game"])
            games[it["game"]] = (Manifest.load(gd), read_tracks(gd, "rim"))
        m, rims = games[it["game"]]
        rim = rim_box_at(rims, int(round(it["t"] * m.fps)), it["rim"])
        if rim is None:
            continue
        follow = rim_follower(rims, it["rim"], m.fps)
        tiles = []
        start = max(0.0, it["t"] - LEAD_S)
        for frame_t, frame in iter_frames(Path(m.source), fps=FPS, start=start, end=start + (N_FRAMES + 1) / FPS):
            rim = follow(frame_t) or rim
            w = max(rim[2] - rim[0], 8.0)
            side = int(round(max(VIEW_WIDTHS * w, 64)))
            cx = (rim[0] + rim[2]) / 2 + it["hoop_dx"] * w
            cy = (rim[1] + rim[3]) / 2 + DROP_WIDTHS * w
            x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
            canvas = cv2.copyMakeBorder(frame, side, side, side, side, cv2.BORDER_REPLICATE)
            tile = canvas[y1 + side:y1 + 2 * side, x1 + side:x1 + 2 * side]
            tiles.append(cv2.resize(tile, (TILE, TILE), interpolation=cv2.INTER_AREA))
            if len(tiles) == N_FRAMES:
                break
        if not tiles:
            continue
        while len(tiles) < N_FRAMES:
            tiles.append(tiles[-1].copy())
        sheet = np.vstack([np.hstack(tiles[r * GRID:(r + 1) * GRID]) for r in range(GRID)])
        cv2.imwrite(str(sheet_path(it["id"])), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if n % 100 == 0:
            print(f"  {n}/{len(todo)} sheets, {time.time() - started:.0f}s", flush=True)
    print(f"cut {len(todo)} sheets in {time.time() - started:.0f}s; {sum(sheet_path(i['id']).exists() for i in items)} of {len(items)} items have one")


# ---- models ----

def build(variant: str):
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.DEFAULT)
    if variant == "montage":            # the whole sheet as one picture
        net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
        return net

    class Frames(nn.Module):            # every frame through the same network, then pooled over time
        def __init__(self):
            super().__init__()
            net.fc = nn.Identity()
            self.net = net
            self.head = nn.Linear(512 * 2, len(CLASSES))

        def forward(self, x):           # x: batch x 3 x (GRID*TILE) x (GRID*TILE)
            b = x.shape[0]
            tiles = x.unfold(2, TILE, TILE).unfold(3, TILE, TILE)                    # b,3,G,G,T,T
            tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(b * N_FRAMES, 3, TILE, TILE)
            f = self.net(tiles).reshape(b, N_FRAMES, 512)
            return self.head(torch.cat([f.mean(1), f.max(1).values], dim=1))

    return Frames()


def _trainer(items, variant: str, epochs: int, lr: float, seed: int):
    """A function (train indices, test indices) -> class probabilities for the test items."""
    import torch
    from torch import nn

    y = np.array([it["y"] for it in items])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def batch(idx, train: bool):
        sheets = np.stack([cv2.cvtColor(cv2.imread(str(sheet_path(items[i]["id"]))), cv2.COLOR_BGR2RGB) for i in idx])
        x = torch.from_numpy(sheets).permute(0, 3, 1, 2).float() / 255.0
        if train:
            flip = torch.rand(len(idx)) < 0.5            # mirror every frame: a left basket becomes a right one
            if flip.any():
                t = x[flip].unfold(2, TILE, TILE).unfold(3, TILE, TILE).flip(-1)
                x[flip] = t.permute(0, 1, 2, 4, 3, 5).reshape(-1, 3, GRID * TILE, GRID * TILE)
            x = x * (1 + 0.3 * (torch.rand(len(idx), 1, 1, 1) - 0.5)) + 0.15 * (torch.rand(len(idx), 1, 1, 1) - 0.5)
        return (x - mean) / std

    def run(tr, te):
        torch.manual_seed(seed)
        rs = np.random.RandomState(seed)
        model = build(variant).to("mps")
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(tr) + 15) // 16))
        w = torch.tensor([len(tr) / (len(CLASSES) * max((y[tr] == c).sum(), 1)) for c in range(len(CLASSES))], dtype=torch.float32).to("mps")
        for _ in range(epochs):
            model.train()
            perm = rs.permutation(tr)
            for b in range(0, len(perm), 16):
                idx = perm[b:b + 16]
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(batch(idx, True).to("mps")), torch.from_numpy(y[idx]).to("mps"), weight=w)
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        out = np.zeros((len(te), len(CLASSES)), np.float32)
        with torch.no_grad():
            for b in range(0, len(te), 32):
                out[b:b + 32] = torch.softmax(model(batch(te[b:b + 32], False).to("mps")), 1).cpu().numpy()
        del model
        return out

    return run, y


def score(variant: str, hold: list[str] | None, epochs: int, limit: int | None, lr: float, seed: int) -> None:
    items = [it for it in labeled_items() if sheet_path(it["id"]).exists()]
    if limit:
        items = items[::max(1, len(items) // limit)]
    run, y = _trainer(items, variant, epochs, lr, seed)
    g = np.array([it["game"] for it in items])
    names = sorted(set(g))
    held = [n for n in names if not hold or any(h in n for h in hold)]
    out = datasets_dir() / "rimclf-images" / f"prob-{variant}.json"
    saved = json.loads(out.read_text()) if out.exists() else {}        # earlier held-out games are kept
    started = time.time()
    for name in held:
        tr, te = np.where(g != name)[0], np.where(g == name)[0]
        prob = run(tr, te)
        report(name[11:30], prob, y[te])
        print(f"    [{time.time() - started:.0f}s so far]", flush=True)
        saved.update({items[i]["id"]: prob[k].tolist() for k, i in enumerate(te)})
        out.write_text(json.dumps(saved))
    have = np.array([it["id"] in saved for it in items])
    report(f"all {len(set(g[have]))} games held out", np.array([saved[it["id"]] for it in items if it["id"] in saved]), y[have])


def curve(variant: str, test: str, order: list[str], epochs: int, lr: float, seed: int) -> None:
    """Does it get better with more games? Hold one game out and train on the first 1, 2, 3, ... of the rest."""
    items = [it for it in labeled_items() if sheet_path(it["id"]).exists()]
    run, y = _trainer(items, variant, epochs, lr, seed)
    g = np.array([it["game"] for it in items])
    names = sorted(set(g))
    pick = lambda part: next(n for n in names if part in n)
    test_name, train_names = pick(test), [pick(o) for o in order]
    te = np.where(g == test_name)[0]
    started = time.time()
    for k in range(1, len(train_names) + 1):
        tr = np.where(np.isin(g, train_names[:k]))[0]
        if len(set(y[tr])) < len(CLASSES):
            continue
        report(f"{test}: {k} game(s), {len(tr)} clips", run(tr, te), y[te])
        print(f"    + {order[k - 1]}  [{time.time() - started:.0f}s so far]", flush=True)


def report(tag: str, prob: np.ndarray, y: np.ndarray) -> None:
    p = prob.argmax(1)
    shots = y < 2
    mm = ((prob[:, 0] > prob[:, 1]) == (y == 0))[shots].mean() if shots.any() else float("nan")
    fakes = (y == 2)
    print(f"{tag:32s} made-vs-missed {mm:.3f} (n={shots.sum()})  3-class {(p == y).mean():.3f}  "
          f"fakes caught {((p == 2) & fakes).sum()}/{fakes.sum()}  real kept {((p < 2) & shots).sum()}/{shots.sum()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut")
    c.add_argument("--limit", type=int)
    s = sub.add_parser("score")
    s.add_argument("variant", choices=["montage", "frames"])
    s.add_argument("--hold", help="comma-separated parts of game ids to hold out (default: every game in turn)")
    s.add_argument("--epochs", type=int, default=10)
    s.add_argument("--limit", type=int)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("curve")
    k.add_argument("variant", choices=["montage", "frames"])
    k.add_argument("test", help="part of the id of the game to hold out")
    k.add_argument("order", help="comma-separated parts of the other games' ids, in the order they are added")
    k.add_argument("--epochs", type=int, default=8)
    k.add_argument("--lr", type=float, default=2e-4)
    k.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "cut":
        cut(a.limit)
    elif a.cmd == "curve":
        curve(a.variant, a.test, a.order.split(","), a.epochs, a.lr, a.seed)
    else:
        score(a.variant, a.hold.split(",") if a.hold else None, a.epochs, a.limit, a.lr, a.seed)
