"""Sim(3) between two reconstructions of the same keyframes, and the Sim(3) pose graph over all windows."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def keep(conf, percentile):
    """Pixels above the given confidence percentile of their own frame."""
    return conf > np.percentile(conf.reshape(len(conf), -1), percentile, axis=1)[:, None, None]


def procrustes(a, b, w, s):
    """R, t with b ~ s R a + t for a given scale s (weighted)."""
    w = w / w.sum(); ma, mb = w @ a, w @ b; U, _, Vt = np.linalg.svd(((b - mb) * w[:, None]).T @ (a - ma)); sg = np.eye(3); sg[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ sg @ Vt
    return R, mb - s * R @ ma


def fit_sim3(a, b, conf, da, db, iters=8):
    """b ~ T a. The scale is the confidence weighted median of the depth ratio of the same pixels (the same whichever side is the reference);
    R, t follow by IRLS on depth-relative residuals so that depth edges and far points do not pull the fit."""
    lr = np.log(da / db); o = np.argsort(lr); cw = np.cumsum(conf[o]); s = float(np.exp(lr[o][np.searchsorted(cw, cw[-1] / 2)]))
    w0 = conf / da ** 2; w = w0
    for _ in range(iters):
        R, t = procrustes(a, b, w, s); r = np.linalg.norm(s * a @ R.T + t - b, axis=1) / da; w = w0 / np.maximum(r, np.median(r) + 1e-9)
    T = np.eye(4); T[:3, :3], T[:3, 3] = s * R, t
    return T, r


def tie(A, rows_a, B, rows_b, percentile, rng, n=80000):
    """Sim(3) taking window B coordinates to window A coordinates from keyframes they share (rows_a of A are rows_b of B)."""
    ok = keep(A.conf[rows_a], percentile) & keep(B.conf[rows_b], percentile); k = np.flatnonzero(ok.reshape(-1)); k = rng.choice(k, min(len(k), n), replace=False)
    pa, pb = A.pts[rows_a].reshape(-1, 3)[k].astype(np.float64), B.pts[rows_b].reshape(-1, 3)[k].astype(np.float64)
    cf, da, db = np.sqrt(A.conf[rows_a] * B.conf[rows_b]).reshape(-1)[k], A.depth[rows_a].reshape(-1)[k], B.depth[rows_b].reshape(-1)[k]
    return fit_sim3(pb, pa, cf, da, db)


def sim3_params(T):
    s = np.cbrt(np.linalg.det(T[:3, :3]))
    return np.r_[Rotation.from_matrix(T[:3, :3] / s).as_rotvec(), T[:3, 3], np.log(s)]


def sim3_matrices(x):
    x = x.reshape(-1, 7); T = np.tile(np.eye(4), (len(x), 1, 1)); T[:, :3, :3] = Rotation.from_rotvec(x[:, :3]).as_matrix() * np.exp(x[:, 6])[:, None, None]; T[:, :3, 3] = x[:, 3:6]
    return T


def solve_graph(poses, edges, depth_scale):
    """Window poses (window -> world Sim(3), window 0 fixed) that best agree with every edge (i, j, measured j -> i Sim(3)), all edges with the
    same weight as in VGGT-SLAM 2.0. The residual is the Sim(3) log of measured^-1 * estimated with the translation in units of the median
    depth of window j. Returns the poses and the residual norm of every edge."""
    def residual(x):
        T, out = sim3_matrices(np.r_[np.zeros(7), x]), []
        for i, j, M in edges:
            E = np.linalg.inv(M) @ np.linalg.inv(T[i]) @ T[j]; s = np.cbrt(np.linalg.det(E[:3, :3]))
            out += [*Rotation.from_matrix(E[:3, :3] / s).as_rotvec(), *(E[:3, 3] / depth_scale[j]), np.log(s)]
        return np.array(out)

    x0 = np.concatenate([sim3_params(T) for T in poses[1:]])
    sol = least_squares(residual, x0, method="lm")
    return list(sim3_matrices(np.r_[np.zeros(7), sol.x])), np.linalg.norm(sol.fun.reshape(-1, 7), axis=1)
