"""Video in, 3D map out.

    from video2map import VideoToMap
    m = VideoToMap({"reconstruction": {"checkpoint": "vggt_omega_1b_512.pt"}})("walk.mp4")
    m.save("out")                      # map.ply, poses.txt, loops.json

Steps: keyframes from summed optical flow -> VGGT-Omega on overlapping windows of keyframes -> Sim(3) between neighbouring windows on all
shared pixels -> loop closures as in VGGT-SLAM 2.0 (SALAD + VGGT-1B match ratio), each one a two-keyframe VGGT-Omega reconstruction tied to
both windows -> one Sim(3) graph over the windows -> every keyframe pixel placed in the world (confidence weighted over the windows holding it).
"""
import copy
import os
import tempfile
import time

import cv2
import numpy as np
import yaml

from .align import keep, solve_graph, tie
from .keyframes import read_keyframes, window_starts
from .map3d import Map3D
from .reconstruct import OmegaReconstructor

DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.yaml")


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class VideoToMap:
    def __init__(self, config=None, verbose=True):
        over = yaml.safe_load(open(config, encoding="utf-8")) if isinstance(config, str) else config
        self.cfg, self.verbose = merge(yaml.safe_load(open(DEFAULT, encoding="utf-8")), over), verbose
        if not self.cfg["reconstruction"]["checkpoint"]:
            raise ValueError("reconstruction.checkpoint: path of the VGGT-Omega weights is needed")
        self._omega = self._loops = None

    def log(self, *a):
        if self.verbose:
            print("[video2map %6.0f s]" % (time.time() - self.t0), *a, flush=True)

    def __call__(self, video, work_dir=None):
        c, rc, rng, self.t0 = self.cfg, self.cfg["reconstruction"], np.random.default_rng(0), time.time()
        fid, imgs = read_keyframes(video, c["keyframes"]["min_disparity"], c["keyframes"]["frame_stride"]); N = len(fid)
        if N < 2:
            raise ValueError("fewer than two keyframes in %s" % video)
        tmp = tempfile.TemporaryDirectory(dir=work_dir); paths = [os.path.join(tmp.name, "%06d.png" % i) for i in fid]
        for p, im in zip(paths, imgs):
            cv2.imwrite(p, im)
        self.log("keyframes %d of video frames up to %d" % (N, fid[-1]))

        self._omega = self._omega or OmegaReconstructor(rc["checkpoint"], rc["resolution"], c["device"]); om = self._omega
        IM = om.load(paths); om.set_grid(imgs[0].shape[:2], IM.shape[-2:], c["output"]["width"])
        win = [list(range(a, min(a + rc["window"], N))) for a in window_starts(N, rc["window"], rc["overlap"])]
        W = [om.unproject(om(IM[w], w)) for w in win]; poses, edges = [np.eye(4)], []
        for q in range(1, len(W)):   # neighbouring windows: every shared keyframe
            sh = sorted(set(win[q - 1]) & set(win[q])); T, _ = tie(W[q - 1], [W[q - 1].row(k) for k in sh], W[q], [W[q].row(k) for k in sh], rc["confidence_percentile"], rng)
            poses.append(poses[-1] @ T); edges.append((q - 1, q, T))
        self.log("windows %d reconstructed and tied" % len(W))
        home = np.array([max(range(len(W)), key=lambda q: min(k - win[q][0], win[q][-1] - k) if k in win[q] else -1) for k in range(N)])   # window where k is most central

        loops = []
        if c["loops"]["enabled"] and len(W) > 2:
            from .loops import LoopDetector
            self._loops = self._loops or LoopDetector(c["loops"]["salad_checkpoint"], c["device"]); ld = self._loops
            I518 = ld.load(paths); found = ld.find(ld.descriptors(I518), I518, win, c["loops"]["retrieval_threshold"], c["loops"]["match_ratio"])
            for lp in [l for l in found if l["accepted"]]:   # the pair reconstructed alone, tied to the query window and to the match window
                w, q, m = lp["window"], lp["query"], lp["match"]; j = int(home[m]); L = om.unproject(om(IM[[m, q]], [m, q]))
                Twl, rw = tie(W[w], [W[w].row(q)], L, [1], rc["confidence_percentile"], rng); Tjl, rj = tie(W[j], [W[j].row(m)], L, [0], rc["confidence_percentile"], rng)
                edges.append((w, j, Twl @ np.linalg.inv(Tjl))); loops.append({"query_frame": int(fid[q]), "match_frame": int(fid[m]), "window": w, "match_window": j,
                                                                          "salad_distance": lp["salad_distance"], "match_ratio": lp["match_ratio"]})
            if loops:
                poses, res = solve_graph(poses, edges, [float(np.median(x.depth)) for x in W])
                for lp, r in zip(loops, res[len(W) - 1:]):
                    lp["graph_residual"] = round(float(r), 4)
            self.log("loop closures %d of %d candidates" % (len(loops), len(found)))

        h, w2 = W[0].depth.shape[1:]; P, Cl, M, Pose = np.zeros((N, h, w2, 3), np.float32), np.zeros((N, h, w2, 3), np.uint8), np.zeros((N, h, w2), bool), np.zeros((N, 4, 4))
        su, sv = om.src_uv; acc, cs = np.zeros((h, w2, 3)), np.zeros((h, w2))
        for k in range(N):
            acc[:], cs[:] = 0, 0
            for q, x in enumerate(W):
                if k in x.idx:
                    r = x.row(k); cf = x.conf[r]; acc += cf[..., None] * (x.pts[r].astype(np.float64) @ poses[q][:3, :3].T + poses[q][:3, 3]); cs += cf
            q = home[k]; r = W[q].row(k); s = np.cbrt(np.linalg.det(poses[q][:3, :3]))
            P[k], M[k], Cl[k] = acc / cs[..., None], keep(cs[None], rc["confidence_percentile"])[0], imgs[k][sv][:, su][..., ::-1]
            Pose[k] = np.eye(4); Pose[k][:3, :3] = poses[q][:3, :3] / s @ W[q].c2w[r, :3, :3]; Pose[k][:3, 3] = poses[q][:3, :3] @ W[q].c2w[r, :3, 3] + poses[q][:3, 3]
        tmp.cleanup(); self.log("map done")
        return Map3D(P, Cl, M, Pose, fid, loops)
