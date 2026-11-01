"""Is this person a player? Referees, coaches, staff and people on the sideline stand where a ball's flight can pass in the
picture, so the shooter rules (film/interpret/location.py) must be able to set them aside. ResNet18 features and a linear
model, on the person's crop. Trained from datasets/official/{official,player}/*.jpg with `film train official`."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from film.common import registry
from film.common.paths import datasets_dir

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
CROP_W, CROP_H = 112, 224
KIND = "official-linear"
BATCH, L2, STEPS = 32, 1e-2, 400
THRESHOLD = 0.5


def _backbone():
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    net.fc = torch.nn.Identity()
    return net


def _tensor(crop_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(crop_bgr, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def embed(crops: list[np.ndarray], net=None) -> np.ndarray:
    net = net if net is not None else _backbone()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), BATCH):
            batch = np.stack([_tensor(c) for c in crops[i:i + BATCH]])
            out.append(net(torch.tensor(batch, dtype=torch.float32)).numpy())
    return np.vstack(out) if out else np.zeros((0, 512))


def fit_linear(X: np.ndarray, y: np.ndarray) -> dict:
    """Logistic regression on standardised features with the two classes weighted equally: non-players are about 1 in 20."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    x, t = torch.tensor((X - mu) / sd, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    weight = torch.where(t > 0, 0.5 / t.sum(), 0.5 / (1 - t).sum())
    w, b = torch.zeros(x.shape[1], requires_grad=True), torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=STEPS, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, t, reduction="none") * weight).sum() + L2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"kind": KIND, "mu": mu, "sd": sd, "w": w.detach().numpy(), "b": float(b.detach())}


def predict_linear(model: dict, X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(((X - model["mu"]) / model["sd"]) @ model["w"] + model["b"])))


def held_out_scores(X: np.ndarray, y: np.ndarray, games: np.ndarray) -> np.ndarray:
    """Each game scored by a model that never saw it."""
    scores = np.zeros(len(y))
    for g in sorted(set(games)):
        scores[games == g] = predict_linear(fit_linear(X[games != g], y[games != g]), X[games == g])
    return scores


class OfficialClassifier:
    def __init__(self, model: dict, version: str = "official@test"):
        self.model, self.version = model, version
        self.net = _backbone()

    def scores(self, crops: list[np.ndarray]) -> np.ndarray:
        return predict_linear(self.model, embed(crops, self.net)) if crops else np.zeros(0)

    @classmethod
    def load(cls) -> "OfficialClassifier | None":
        found = registry.latest("official")
        if found is None:
            return None
        version, folder = found
        saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        return cls(saved, version=f"official@{version}") if saved.get("kind") == KIND else None


def dataset_rows() -> list[tuple[Path, bool, str]]:
    """(crop, is it a non-player, its game). A crop is named <game_id>--<anything>.jpg."""
    root = datasets_dir() / "official"
    return [(p, label == "official", p.stem.split("--")[0]) for label in ("official", "player") for p in sorted((root / label).glob("*.jpg"))]


def train_official(log=print) -> dict:
    """Train on every labelled crop and register it, with its score on games it did not see in the manifest."""
    rows = dataset_rows()
    if not any(r[1] for r in rows):
        raise RuntimeError("no labelled non-players in datasets/official/official")
    X = embed([cv2.imread(str(r[0])) for r in rows])
    y = np.array([r[1] for r in rows], dtype=np.float32)
    games = np.array([r[2] for r in rows])
    held = held_out_scores(X, y, games) >= THRESHOLD
    result = {"crops": len(rows), "non_players": int(y.sum()), "games": len(set(games)),
              "held_out_caught": int((held & (y > 0)).sum()), "held_out_players_flagged": int((held & (y == 0)).sum())}
    version, folder = registry.new_version_dir("official")
    torch.save(fit_linear(X, y), folder / "best.pt")
    registry.write_manifest(folder, **result)
    log(f"official@{version}: {result['crops']} crops; on games it did not see, caught {result['held_out_caught']} of {result['non_players']} "
        f"non-players and wrongly flagged {result['held_out_players_flagged']} of {result['crops'] - result['non_players']} players")
    return result
