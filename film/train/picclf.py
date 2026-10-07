"""Made, missed or none from the pictures: every frame of a rim view through one network, pooled over time.

The second opinion beside the ball-path classifier (film.train.rimclf). Measured on six reviewed games, each
held out in turn: 0.81 on made versus missed against 0.74 for the path classifier, and unlike it this one
keeps improving as reviewed games are added."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from film.common import registry
from film.interpret.rimview import GRID, N_FRAMES, TILE
from film.train.rimclf import CLASSES

KIND = "rim-view-frames"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BATCH = 16


class FrameNet(nn.Module):
    """One backbone applied to each of the sixteen frames; mean and max over time; a linear head."""

    def __init__(self, arch: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if arch == "tiny":                     # for tests: no download, trains in a second
            self.backbone, width = nn.Sequential(nn.Conv2d(3, 8, 7, stride=8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()), 8
        else:
            from torchvision.models import ResNet18_Weights, resnet18

            net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            net.fc = nn.Identity()
            self.backbone, width = net, 512
        self.head = nn.Linear(2 * width, len(CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # batch x 3 x sheet x sheet
        b = x.shape[0]
        tiles = x.unfold(2, TILE, TILE).unfold(3, TILE, TILE).permute(0, 2, 3, 1, 4, 5)
        f = self.backbone(tiles.reshape(b * N_FRAMES, 3, TILE, TILE)).reshape(b, N_FRAMES, -1)
        return self.head(torch.cat([f.mean(dim=1), f.max(dim=1).values], dim=1))


def _load(view) -> np.ndarray:
    sheet = cv2.imread(str(view)) if isinstance(view, (str, Path)) else np.asarray(view)
    if sheet is None or sheet.shape != (GRID * TILE, GRID * TILE, 3):
        raise ValueError(f"a rim view must be {GRID * TILE} px square (see film.interpret.rimview), got "
                         f"{None if sheet is None else sheet.shape} for {view if isinstance(view, (str, Path)) else 'an array'}")
    return sheet


def _tensor(views, train: bool = False, gen: torch.Generator | None = None) -> torch.Tensor:
    x = torch.from_numpy(np.stack([cv2.cvtColor(_load(v), cv2.COLOR_BGR2RGB) for v in views])).permute(0, 3, 1, 2).float() / 255.0
    if train:
        n = x.shape[0]
        flip = torch.rand(n, generator=gen) < 0.5            # mirror every frame: a left basket becomes a right one
        if flip.any():
            t = x[flip].unfold(2, TILE, TILE).unfold(3, TILE, TILE).flip(-1)
            x[flip] = t.permute(0, 1, 2, 4, 3, 5).reshape(-1, 3, GRID * TILE, GRID * TILE)
        x = x * (1 + 0.3 * (torch.rand(n, 1, 1, 1, generator=gen) - 0.5)) + 0.15 * (torch.rand(n, 1, 1, 1, generator=gen) - 0.5)
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


class PictureClassifier:
    def __init__(self, model: FrameNet, device: str, arch: str, version: str = "picclf@test"):
        self.model = model.to(device).eval()
        self.device = device
        self.arch = arch
        self.version = version

    @torch.no_grad()
    def scores(self, views) -> np.ndarray:
        """Class probabilities, one row per view (a sheet array or a file)."""
        views = list(views)
        out = np.zeros((len(views), len(CLASSES)), np.float32)
        for b in range(0, len(views), 2 * BATCH):
            logits = self.model(_tensor(views[b:b + 2 * BATCH]).to(self.device))
            out[b:b + 2 * BATCH] = torch.softmax(logits, dim=1).cpu().numpy()
        return out

    def predict(self, view) -> tuple[str, float, dict[str, float]]:
        scores = dict(zip(CLASSES, (float(v) for v in self.scores([view])[0])))
        label = max(scores, key=scores.get)
        return label, scores[label], scores

    def save(self, path: Path) -> None:
        torch.save({"kind": KIND, "arch": self.arch, "classes": CLASSES, "frames": N_FRAMES, "tile": TILE,
                    "state": {k: v.cpu() for k, v in self.model.state_dict().items()}}, path)

    @classmethod
    def load(cls, device: str) -> "PictureClassifier | None":
        found = registry.latest("picclf")
        if found is None:
            return None
        version, folder = found
        saved = torch.load(folder / "best.pt", map_location="cpu")
        if saved.get("kind") != KIND or saved.get("frames") != N_FRAMES or saved.get("tile") != TILE:
            return None
        model = FrameNet(saved["arch"], pretrained=False)
        model.load_state_dict(saved["state"])
        return cls(model, device, saved["arch"], version=f"picclf@{version}")


def fit_picclf(views, y, device: str, arch: str = "resnet18", pretrained: bool = True, epochs: int = 8,
               lr: float = 2e-4, seed: int = 0) -> PictureClassifier:
    """Fine-tune on rim views (arrays or files) with each class weighted by its rarity. Seeded."""
    views, y = list(views), np.asarray(y, np.int64)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    order = np.random.RandomState(seed)
    model = FrameNet(arch, pretrained).to(device)
    steps = epochs * ((len(views) + BATCH - 1) // BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1))
    counts = np.bincount(y, minlength=len(CLASSES))
    weight = torch.tensor([len(y) / (len(CLASSES) * c) if c else 0.0 for c in counts], dtype=torch.float32).to(device)
    for _ in range(epochs):
        model.train()
        perm = order.permutation(len(views))
        for b in range(0, len(perm), BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            logits = model(_tensor([views[i] for i in idx], train=True, gen=gen).to(device))
            nn.functional.cross_entropy(logits, torch.from_numpy(y[idx]).to(device), weight=weight).backward()
            opt.step()
            sched.step()
    return PictureClassifier(model, device, arch)
