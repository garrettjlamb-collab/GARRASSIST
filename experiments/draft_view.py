"""One sheet per game: its queued court pictures, gridded in full-picture pixels, to draw drafts from."""
import sys

import cv2
import numpy as np

import film.train.datasets as ds

game, out = sys.argv[1], sys.argv[2]
tiles = []
for i in ds.list_pending("court"):
    if not i.startswith(game + "_"):
        continue
    d = ds.read_pending("court", i)
    img = cv2.imread(d["image"])
    h, w = img.shape[:2]
    step = 100 if w > 1300 else 50
    for x in range(step, w, step):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255) if x % (2 * step) == 0 else (0, 110, 110), 1)
        if x % (2 * step) == 0:
            cv2.putText(img, str(x), (x + 2, 16), 0, 0.5 * w / 1920 + 0.2, (0, 255, 255), 1, cv2.LINE_AA)
    for y in range(step, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 255) if y % (2 * step) == 0 else (0, 110, 110), 1)
        if y % (2 * step) == 0:
            cv2.putText(img, str(y), (2, y - 3), 0, 0.5 * w / 1920 + 0.2, (0, 255, 255), 1, cv2.LINE_AA)
    t = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 520), (240, 540), (0, 0, 0), -1)
    cv2.putText(t, f"{i[-6:]} {w}x{h}", (4, 535), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(t)
while len(tiles) % 2:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[r:r + 2]) for r in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(out.split("/")[-1], len(tiles))
