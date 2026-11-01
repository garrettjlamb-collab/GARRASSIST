"""The ball's real path. A shot is a parabola under gravity that ends at a rim 10 ft up, seen from one camera: given when it
reaches the rim, each sighting of the ball is a ray that the path must cross, and the path's velocity follows by least squares.
Run backward to the height of a shooter's hands, it says where the shot was taken from, with no person involved."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from film.interpret.camera import Camera
from film.interpret.court import BASKET, WIDTH

G = 32.17                                  # ft/s^2
RIM_Z = 10.0
RELEASE_Z = 8.0                            # about where a shot leaves the hands
TOL_PX = 12.0                              # a sighting this close to the path is on it, at REFERENCE_WIDTH
TA_STEP_S, TA_FINE_S = 0.1, 0.02
MIN_FLIGHT_S, MAX_FLIGHT_S = 0.25, 2.2
MAX_VXY, MIN_DESCENT, MAX_DESCENT = 40.0, 2.0, 45.0     # ft/s
MIN_POINTS = 4
MIN_SPAN_S = 0.3
MAX_PAIRS = 600
COURT_MARGIN_FT = 4.0


@dataclass(frozen=True)
class Path3D:
    rim: tuple[float, float, float]
    ta: float                              # when it reaches the rim
    v: np.ndarray                          # velocity at the rim, ft/s
    release_t: float
    spot: tuple[float, float]              # full-court feet, under the ball at the height of the hands
    points: int
    span_s: float
    error_px: float

    def at(self, t: float) -> np.ndarray:
        tau = t - self.ta
        return np.array([self.rim[0] + self.v[0] * tau, self.rim[1] + self.v[1] * tau, self.rim[2] + self.v[2] * tau - 0.5 * G * tau * tau])


def rims_of(length: float) -> list[tuple[float, float, float]]:
    return [(BASKET[0], BASKET[1], RIM_Z), (WIDTH - BASKET[0], length - BASKET[1], RIM_Z)]


def _skew_batch(d: np.ndarray) -> np.ndarray:
    z = np.zeros(len(d))
    return np.stack([np.stack([z, -d[:, 2], d[:, 1]], 1), np.stack([d[:, 2], z, -d[:, 0]], 1), np.stack([-d[:, 1], d[:, 0], z], 1)], 1)


def _release_tau(vz: np.ndarray, z: float = RELEASE_Z) -> np.ndarray:
    """How long before the rim the ball was at height z on its way up (negative)."""
    return (vz - np.sqrt(vz * vz + 2.0 * G * (RIM_Z - z))) / G


def _project(cam: Camera, xyz: np.ndarray) -> np.ndarray:
    """xyz: ... x 3 feet -> ... x 2 pixels (nan behind the camera)."""
    v = xyz @ cam.P[:, :3].T + cam.P[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v[..., 2:] > 1e-9, v[..., :2] / v[..., 2:], np.nan)


def _fit_v(cam: Camera, rim, ta, t, rays, skew, keep) -> np.ndarray | None:
    """Velocity at the rim, least squares over the sightings kept."""
    tau = t[keep] - ta
    if len(tau) < 2:
        return None
    q = np.array(rim) - cam.centre + np.outer(-0.5 * G * tau * tau, [0.0, 0.0, 1.0])
    a = (tau[:, None, None] * skew[keep]).reshape(-1, 3)
    b = np.cross(q, rays[keep]).reshape(-1)
    try:
        return np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None


def _residuals(cam: Camera, rim, ta: float, V: np.ndarray, t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Pixels between each sighting and each candidate path; V: h x 3, t: n -> h x n."""
    tau = t - ta
    pts = np.array(rim)[None, None, :] + V[:, None, :] * tau[None, :, None]
    pts = pts + np.array([0.0, 0.0, 1.0])[None, None, :] * (-0.5 * G * tau * tau)[None, :, None]
    p = _project(cam, pts)
    return np.hypot(p[..., 0] - xy[:, 0], p[..., 1] - xy[:, 1])


def fit_path(cam: Camera, sightings: list[tuple[float, float, float, float]], rims: list[tuple[float, float, float]], t_from: float, t_to: float,
             length: float, scale: float = 1.0, release_z: float = RELEASE_Z) -> Path3D | None:
    """The path, ending at one of `rims` between t_from and t_to, that the most sightings lie on. The sightings are pixels of the
    camera's own frame. None when no path a shot could take explains at least MIN_POINTS of them."""
    if len(sightings) < MIN_POINTS:
        return None
    s = np.array(sorted(sightings))
    t, xy = s[:, 0], s[:, 1:3]
    d = np.linalg.solve(cam.P[:, :3], np.column_stack([xy, np.ones(len(xy))]).T).T
    if np.dot(d[0], d[0]) <= 0:
        return None
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    # The rays point from the camera into the scene: a sighting ahead of the camera has positive depth along them.
    skew = _skew_batch(d)
    tol = TOL_PX * scale
    best = None
    i_all, j_all = np.triu_indices(len(t), 1)
    for rim in rims:
        for ta in np.arange(t_from, t_to + 1e-6, TA_STEP_S):
            before = t <= ta + 0.05
            pairs = before[i_all] & before[j_all] & (np.abs(t[i_all] - t[j_all]) >= 0.15) & (t[i_all] < ta - 0.05) & (t[j_all] < ta + 0.05)
            pi, pj = i_all[pairs], j_all[pairs]
            if len(pi) == 0:
                continue
            if len(pi) > MAX_PAIRS:
                keep = np.linspace(0, len(pi) - 1, MAX_PAIRS).astype(int)
                pi, pj = pi[keep], pj[keep]
            V = _solve_pairs(cam, rim, ta, t, d, skew, pi, pj)
            cand = _physical(V, rim, length)
            if not cand.any():
                continue
            V = V[cand]
            res = _residuals(cam, rim, ta, V, t, xy)
            tau_r = _release_tau(V[:, 2], release_z)
            within = (t[None, :] - ta >= tau_r[:, None] - 0.05) & (t[None, :] <= ta + 0.05)
            inl = (res <= tol) & within
            counts = _distinct_times(inl, t)
            k = int(np.argmax(counts))
            if counts[k] >= MIN_POINTS and (best is None or counts[k] > best[0]):
                best = (int(counts[k]), rim, float(ta), V[k])
    if best is None:
        return None
    return _refine(cam, best, t, xy, d, skew, tol, length, release_z)


def _solve_pairs(cam, rim, ta, t, d, skew, pi, pj) -> np.ndarray:
    def rows(idx):
        tau = t[idx] - ta
        q = np.array(rim) - cam.centre + np.outer(-0.5 * G * tau * tau, [0.0, 0.0, 1.0])
        return tau[:, None, None] * skew[idx], np.cross(q, d[idx])[..., None]

    ai, bi = rows(pi)
    aj, bj = rows(pj)
    a, b = np.concatenate([ai, aj], 1), np.concatenate([bi, bj], 1)
    ata = np.transpose(a, (0, 2, 1)) @ a + 1e-9 * np.eye(3)
    atb = np.transpose(a, (0, 2, 1)) @ b
    return np.linalg.solve(ata, atb)[..., 0]


def _physical(V: np.ndarray, rim, length: float) -> np.ndarray:
    """Could a shot have this velocity at the rim: coming down, not too fast, from a spot on the court in the time a shot takes?"""
    ok = (V[:, 2] <= -MIN_DESCENT) & (V[:, 2] >= -MAX_DESCENT) & (np.hypot(V[:, 0], V[:, 1]) <= MAX_VXY)
    tau_r = _release_tau(np.minimum(V[:, 2], -1e-3))
    ok &= (-tau_r >= MIN_FLIGHT_S) & (-tau_r <= MAX_FLIGHT_S)
    x, y = rim[0] + V[:, 0] * tau_r, rim[1] + V[:, 1] * tau_r
    return ok & (x >= -COURT_MARGIN_FT) & (x <= WIDTH + COURT_MARGIN_FT) & (y >= -COURT_MARGIN_FT) & (y <= length + COURT_MARGIN_FT)


def _distinct_times(inl: np.ndarray, t: np.ndarray) -> np.ndarray:
    """How many different moments each path is seen at: two boxes in one frame are one sighting."""
    uniq, inv = np.unique(t, return_inverse=True)
    out = np.zeros((inl.shape[0], len(uniq)), dtype=bool)
    np.logical_or.at(out, (np.arange(inl.shape[0])[:, None].repeat(inl.shape[1], 1), np.tile(inv, (inl.shape[0], 1))), inl)
    return out.sum(1)


def _refine(cam, best, t, xy, d, skew, tol, length, release_z) -> Path3D | None:
    _, rim, ta0, V0 = best
    top = None
    for ta in np.arange(ta0 - TA_STEP_S, ta0 + TA_STEP_S + 1e-6, TA_FINE_S):
        V = V0
        keep = None
        for _ in range(3):                 # refit on who agrees, then see who agrees now
            res = _residuals(cam, rim, ta, V[None, :], t, xy)[0]
            tau_r = float(_release_tau(np.array([min(V[2], -1e-3)]), release_z)[0])
            now = (res <= tol) & (t - ta >= tau_r - 0.05) & (t <= ta + 0.05)
            now = _one_per_time(now, res, t)
            if now.sum() < 2:
                break
            keep = now
            fit = _fit_v(cam, rim, ta, t, d, skew, keep)
            if fit is None:
                break
            V = fit
        if keep is None or not _physical(V[None, :], rim, length)[0]:
            continue
        res = _residuals(cam, rim, ta, V[None, :], t, xy)[0]
        tau_r = float(_release_tau(np.array([V[2]]), release_z)[0])
        now = _one_per_time((res <= tol) & (t - ta >= tau_r - 0.05) & (t <= ta + 0.05), res, t)
        if now.sum() < MIN_POINTS:
            continue
        rank = (int(now.sum()), -float(res[now].mean()))
        if top is None or rank > top[0]:
            top = (rank, ta, V.copy(), now.copy(), float(res[now].mean()), tau_r)
    if top is None:
        return None
    _, ta, V, now, err, tau_r = top
    span = float(t[now].max() - t[now].min())
    if span < MIN_SPAN_S:
        return None
    return Path3D(rim=rim, ta=float(ta), v=V, release_t=float(ta + tau_r), spot=(float(rim[0] + V[0] * tau_r), float(rim[1] + V[1] * tau_r)),
                  points=int(now.sum()), span_s=span, error_px=round(err, 1))


def _one_per_time(mask: np.ndarray, res: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    for tt in np.unique(t[mask]):
        idx = np.flatnonzero(mask & (t == tt))
        out[idx[np.argmin(res[idx])]] = True
    return out
