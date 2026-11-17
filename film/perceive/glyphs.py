"""Read a scoreboard's digits by matching them against labeled examples of that skin's own font.

The seven-segment zones (sevenseg.py) assume upright digits whose strokes fill fixed parts of the cell. The
2026-27 Marcos de Niza skin draws slanted LED-dot digits that sit a few pixels off last season's cells, so
its 7 reads as two segments and its 4 as something else. A few dozen digits read by eye from one game, at
known film times, teach the reader the font: each cell is compared with every example of every digit,
a few pixels either way, and the best match wins when it is clear.

`film glyphs <game_id> 256=7:58 1160=0:08 ...` builds the examples from a game's film and writes them next
to the profile; a profile with "glyphs" uses them for the fields it names.
"""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import cv2
import numpy as np

from film.perceive.sevenseg import HUE_RANGES

PAD = 4                 # the template is the cell plus this margin; the search slides this far again
MIN_LIT = 0.06          # below this share of lit pixels a cell is blank
MIN_SCORE = 0.55        # the best match must correlate at least this well ...
MIN_MARGIN = 0.04       # ... and beat the best other digit by this much


def lit_image(frame_bgr: np.ndarray, color: str) -> np.ndarray:
    """A float image of how strongly each pixel is a lit LED of this colour: value where the hue and
    saturation fit, else 0. Softer than led_mask, which helps a dotted font match."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    in_hue = np.zeros(hue.shape, bool)
    for lo, hi in HUE_RANGES[color]:
        in_hue |= (hue >= lo) & (hue <= hi)
    return np.where(in_hue & (sat > 70), val / 255.0, 0.0).astype(np.float32)


def _box(cell, pad: int) -> tuple[int, int, int, int]:
    x, y, w, h = cell
    return x - pad, y - pad, x + w + pad, y + h + pad


def crop(img: np.ndarray, cell, pad: int) -> np.ndarray:
    x1, y1, x2, y2 = _box(cell, pad)
    h, w = img.shape[:2]
    out = np.zeros((y2 - y1, x2 - x1), np.float32)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = img[sy1:sy2, sx1:sx2]
    return out


class Glyphs:
    """Examples of each digit for each named cell ("clock0".."clock3", "period0", ...)."""

    def __init__(self, examples: dict[str, list[tuple[int, np.ndarray]]]):
        self.examples = examples

    @classmethod
    def load(cls, path: Path) -> "Glyphs":
        data = np.load(path)
        examples: dict[str, list[tuple[int, np.ndarray]]] = {}
        for key in data.files:                              # "<cell>|<digit>|<n>"
            cell, digit, _ = key.split("|")
            examples.setdefault(cell, []).append((int(digit), data[key]))
        return cls(examples)

    def save(self, path: Path) -> None:
        arrays = {f"{cell}|{d}|{i}": img for cell, ex in self.examples.items() for i, (d, img) in enumerate(ex)}
        np.savez_compressed(path, **arrays)

    def pool(self, cell_key: str, size: tuple[int, int]) -> list[tuple[int, np.ndarray]]:
        """This cell's examples, and every other cell's resized to fit, for digits this cell has not seen."""
        own = self.examples.get(cell_key, [])
        seen = {d for d, _ in own}
        borrowed = [(d, cv2.resize(img, size)) for k, ex in self.examples.items() if k != cell_key
                    for d, img in ex if d not in seen]
        return own + borrowed

    def read(self, lit: np.ndarray, cell, cell_key: str) -> int | None:
        """The digit in the cell: None when blank, -1 when no digit matches clearly."""
        ranked = self.rank(lit, cell, cell_key)
        if ranked is None:
            return None
        if not ranked:
            return -1
        top, score = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else -1.0
        if score < MIN_SCORE or score - runner < MIN_MARGIN:
            return -1
        return top

    def rank(self, lit: np.ndarray, cell, cell_key: str) -> list[tuple[int, float]] | None:
        """Every digit with its best match score, best first; None when the cell is blank."""
        tmpl_h, tmpl_w = cell[3] + 2 * PAD, cell[2] + 2 * PAD
        region = crop(lit, cell, 2 * PAD)
        core = crop(lit, cell, 0)
        if float((core > 0.3).mean()) < MIN_LIT:
            return None
        best: dict[int, float] = {}
        for digit, tmpl in self.pool(cell_key, (tmpl_w, tmpl_h)):
            if tmpl.shape != (tmpl_h, tmpl_w):
                tmpl = cv2.resize(tmpl, (tmpl_w, tmpl_h))
            if float(tmpl.std()) == 0:
                continue
            score = float(cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED).max())
            best[digit] = max(best.get(digit, -1.0), score)
        return sorted(best.items(), key=lambda kv: -kv[1])


def glyphs_path(name: str):
    return files("film.perceive.profiles") / name


def clock_digits(value: str) -> list[int | None]:
    """"7:58" -> [None, 7, 5, 8]; "0:08" -> [None, 0, 0, 8]; "12:00" -> [1, 2, 0, 0] (cells M M S S)."""
    minutes, seconds = value.split(":")
    m = [None, int(minutes)] if len(minutes) == 1 else [int(minutes[0]), int(minutes[1])]
    return m + [int(seconds[0]), int(seconds[1])]


def harvest(frames: list[tuple[np.ndarray, str]], profile: dict) -> Glyphs:
    """Examples from frames whose clock is known: [(frame, "7:58"), ...]. Only the clock's cells are labeled;
    other fields that use glyphs borrow them (same font, resized)."""
    field = profile["fields"]["clock"]
    examples: dict[str, list[tuple[int, np.ndarray]]] = {}
    for frame, value in frames:
        lit = lit_image(frame, field["color"])
        for i, (cell, digit) in enumerate(zip(field["cells"], clock_digits(value))):
            if digit is None:
                continue
            examples.setdefault(f"clock{i}", []).append((digit, crop(lit, cell, PAD)))
    return Glyphs(examples)
