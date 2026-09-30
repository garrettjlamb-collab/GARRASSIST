from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from film.common import registry

CLASSES = ["made", "missed", "none"]
INPUT = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(arch: str = "resnet18", pretrained: bool = True, num_classes: int = len(CLASSES)) -> nn.Module:
    if arch == "tiny":
        return nn.Sequential(nn.Conv2d(3, 8, 5, stride=4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, num_classes))
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def preprocess(montage_bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(montage_bgr, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)


class RimClassifier:
    def __init__(self, model: nn.Module, device: str, version: str = "rimclf@test"):
        self.model = model.to(device).eval()
        self.device = device
        self.version = version

    @classmethod
    def load(cls, device: str) -> "RimClassifier | None":
        found = registry.latest("rimclf")
        if found is None:
            return None
        version, folder = found
        info = registry.read_manifest(folder) if (folder / "manifest.json").exists() else {}
        model = build_model(arch=info.get("arch", "resnet18"), pretrained=False)
        model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu"))
        return cls(model, device, version=f"rimclf@{version}")

    @torch.no_grad()
    def predict(self, montage_bgr: np.ndarray) -> tuple[str, float, dict[str, float]]:
        logits = self.model(preprocess(montage_bgr).to(self.device))
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        scores = dict(zip(CLASSES, probs))
        label = max(scores, key=scores.get)
        return label, scores[label], scores


class FakeClassifier:
    """Scripted classifier for tests: returns the given (label, prob) per call, the last one repeating."""

    def __init__(self, labels: list[tuple[str, float]] | None = None, version: str = "rimclf@fake"):
        self.labels = labels or [("made", 0.9)]
        self.calls = 0
        self.version = version

    def predict(self, montage_bgr) -> tuple[str, float, dict[str, float]]:
        label, prob = self.labels[min(self.calls, len(self.labels) - 1)]
        self.calls += 1
        scores = {c: (prob if c == label else (1 - prob) / 2) for c in CLASSES}
        return label, prob, scores


def train_rimclf(dataset_dir: Path, out_dir: Path, device: str, epochs: int = 8, batch: int = 32,
                 lr: float = 1e-3, arch: str = "resnet18", pretrained: bool = True) -> dict:
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.Resize((INPUT, INPUT)), transforms.ToTensor(),
                             transforms.Normalize(MEAN.tolist(), STD.tolist())])
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    for split in ("train", "val"):
        found = sorted(p.name for p in (dataset_dir / split).iterdir() if p.is_dir())
        if found != CLASSES:
            raise ValueError(f"{dataset_dir / split} must hold exactly the folders {CLASSES}, found {found}")
    train_ds = datasets.ImageFolder(str(dataset_dir / "train"), transform=tf)
    val_ds = datasets.ImageFolder(str(dataset_dir / "val"), transform=tf)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch)
    model = build_model(arch, pretrained).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, history = -1.0, []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(y)
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += int((pred == y).sum())
                n += len(y)
        acc = correct / max(n, 1)
        history.append({"epoch": epoch + 1, "loss": round(total / max(len(train_ds), 1), 4), "val_acc": round(acc, 4)})
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_dir / "best.pt")
    return {"arch": arch, "epochs": epochs, "best_val_acc": best_acc, "train_images": len(train_ds),
            "val_images": len(val_ds), "history": history}
