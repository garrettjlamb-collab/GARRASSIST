"""Experiment: tell a non-player from a player on the chosen-shooter crops. ResNet18 features, a linear model, scored on a
game the model did not train on. Labels are Claude's by eye, from experiments/shooter_sheets.py."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

OFFICIAL = {15, 36, 71, 74, 143, 216, 235, 242, 243, 251, 258, 265, 274, 283, 285, 287, 289, 296, 386, 392, 395, 401, 404, 416, 417, 419, 427, 450, 485, 524}
UNSURE = {83, 118, 403, 489, 496}
d = Path(sys.argv[1])
rows = json.loads((d / "rows.json").read_text())
model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
model.fc = torch.nn.Identity()


class Linear:
    """Logistic regression with balanced classes and an L2 penalty, on standardised features."""

    def fit(self, X, y, l2=1e-2, steps=400):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        x = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        t = torch.tensor(y, dtype=torch.float32)
        w = torch.where(t > 0, 0.5 / t.sum(), 0.5 / (1 - t).sum())
        self.w = torch.zeros(x.shape[1], requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([self.w, self.b], max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(x @ self.w + self.b, t, reduction="none") * w).sum() + l2 * (self.w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def proba(self, X):
        with torch.no_grad():
            return torch.sigmoid(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32) @ self.w + self.b).numpy()


mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def tensor(p: Path) -> np.ndarray:
    im = cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (112, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) / 255.0
    return ((im - mean) / std).transpose(2, 0, 1)


X = []
with torch.no_grad():
    for i in range(0, len(rows), 32):
        b = np.stack([tensor(d / f"{r['n']:04d}.jpg") for r in rows[i:i + 32]])
        X.append(model(torch.tensor(b, dtype=torch.float32)).numpy())
X = np.vstack(X)
np.save(d / "features.npy", X)
y = np.array([r["n"] in OFFICIAL for r in rows])
keep = np.array([r["n"] not in UNSURE for r in rows])
games = np.array([r["game"] for r in rows])
scores = np.full(len(rows), np.nan)
for g in sorted(set(games)):
    tr = keep & (games != g)
    scores[games == g] = Linear().fit(X[tr], y[tr]).proba(X[games == g])
for g in sorted(set(games)):
    k = keep & (games == g)
    print(g.replace("2026-10-09-new-marcos-de-niza-high-school-vs-", "")[:26].ljust(26), "shots", k.sum(), "non-players", y[k].sum(), "caught at 0.5", int((scores[k] >= .5)[y[k]].sum()), "players wrongly flagged", int((scores[k] >= .5)[~y[k]].sum()))
for th in (0.3, 0.5, 0.7, 0.9):
    print(f"at {th}: caught {int(((scores >= th) & y & keep).sum())} of {int((y & keep).sum())}; players wrongly flagged {int(((scores >= th) & ~y & keep).sum())} of {int((~y & keep).sum())}")
print("missed at .5", [r["n"] for r, s in zip(rows, scores) if r["n"] in OFFICIAL and s < .5])
print("wrongly flagged at .5", [(r["n"], round(float(s), 2)) for r, s in zip(rows, scores) if r["n"] not in OFFICIAL | UNSURE and s >= .5])
np.save(d / "scores.npy", scores)
