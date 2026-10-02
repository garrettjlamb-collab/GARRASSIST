from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from film.common import registry
from film.interpret.ballpath import FEATURES

CLASSES = ["made", "missed", "none"]
KIND = "ball-path"
L2 = 0.02


class RimClassifier:
    """Made, missed or none from the ball's path after it reaches the rim (see film.interpret.ballpath).

    A linear model over six interpretable features: two reviewed games are far too few clips for an image
    classifier (tried: no better than guessing), and enough for this."""

    def __init__(self, mean, std, weight, bias, version: str = "rimclf@test"):
        self.mean = np.asarray(mean, np.float64)
        self.std = np.asarray(std, np.float64)
        self.weight = np.asarray(weight, np.float64)      # classes x features
        self.bias = np.asarray(bias, np.float64)
        self.version = version

    def scores(self, features) -> np.ndarray:
        """Class probabilities for one feature row or a matrix of them."""
        z = ((np.atleast_2d(np.asarray(features, np.float64)) - self.mean) / self.std) @ self.weight.T + self.bias
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, features) -> tuple[str, float, dict[str, float]]:
        scores = dict(zip(CLASSES, (float(v) for v in self.scores(features)[0])))
        label = max(scores, key=scores.get)
        return label, scores[label], scores

    def save(self, path: Path) -> None:
        torch.save({"kind": KIND, "features": FEATURES, "classes": CLASSES, "mean": self.mean.tolist(),
                    "std": self.std.tolist(), "weight": self.weight.tolist(), "bias": self.bias.tolist()}, path)

    @classmethod
    def load(cls, device: str | None = None) -> "RimClassifier | None":
        found = registry.latest("rimclf")
        if found is None:
            return None
        version, folder = found
        state = torch.load(folder / "best.pt", map_location="cpu")
        if state.get("kind") != KIND or state.get("features") != FEATURES:
            return None
        return cls(state["mean"], state["std"], state["weight"], state["bias"], version=f"rimclf@{version}")


class FakeClassifier:
    """Scripted classifier for tests: returns the given (label, prob) per call, the last one repeating."""

    def __init__(self, labels: list[tuple[str, float]] | None = None, version: str = "rimclf@fake"):
        self.labels = labels or [("made", 0.9)]
        self.calls = 0
        self.version = version

    def predict(self, features) -> tuple[str, float, dict[str, float]]:
        label, prob = self.labels[min(self.calls, len(self.labels) - 1)]
        self.calls += 1
        scores = {c: (prob if c == label else (1 - prob) / 2) for c in CLASSES}
        return label, prob, scores


def fit_rimclf(X, y, l2: float = L2) -> RimClassifier:
    """Softmax regression on standardised features, each class weighted by its rarity. Deterministic."""
    X, y = np.asarray(X, np.float64), np.asarray(y, np.int64)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
    xt, yt = torch.tensor((X - mean) / std, dtype=torch.float64), torch.tensor(y)
    counts = np.bincount(y, minlength=len(CLASSES))
    class_w = torch.tensor([len(y) / (len(CLASSES) * c) if c else 0.0 for c in counts], dtype=torch.float64)
    weight = torch.zeros(len(CLASSES), X.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(len(CLASSES), dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.LBFGS([weight, bias], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad()
        loss = torch.nn.functional.cross_entropy(xt @ weight.T + bias, yt, weight=class_w) + l2 * (weight ** 2).sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    return RimClassifier(mean, std, weight.detach().numpy(), bias.detach().numpy())
