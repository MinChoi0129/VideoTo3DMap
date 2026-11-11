"""The map a video turns into: world points and colour of every keyframe pixel, keyframe camera poses, loop closures."""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

PLY_HEADER = "ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"


class Map3D:
    """points (K, h, w, 3) float32 and colors (K, h, w, 3) uint8 per keyframe pixel, mask (K, h, w) of the confident ones,
    poses (K, 4, 4) camera-to-world, frame_ids (K,) video frame of every keyframe. The map has one global scale of its own (not metres)."""

    def __init__(self, points, colors, mask, poses, frame_ids, loops=()):
        self.points, self.colors, self.mask, self.poses, self.frame_ids, self.loops = points, colors, mask, poses, frame_ids, list(loops)

    def point_cloud(self, voxel=0.0):
        """(N, 3) points and (N, 3) colours of the confident pixels; voxel > 0 keeps one averaged point per voxel (map units)."""
        xyz, rgb = self.points[self.mask], self.colors[self.mask]
        if voxel > 0:
            key = np.unique(np.floor(xyz / voxel).astype(np.int64), axis=0, return_inverse=True)[1].ravel(); n = key.max() + 1; cnt = np.bincount(key, minlength=n)[:, None]
            xyz = np.stack([np.bincount(key, xyz[:, i], n) for i in range(3)], 1) / cnt; rgb = (np.stack([np.bincount(key, rgb[:, i], n) for i in range(3)], 1) / cnt).round().astype(np.uint8)
        return xyz.astype(np.float32), rgb

    def save_ply(self, path, voxel=0.0):
        xyz, rgb = self.point_cloud(voxel)
        v = np.empty(len(xyz), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        v["x"], v["y"], v["z"], v["r"], v["g"], v["b"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], rgb[:, 0], rgb[:, 1], rgb[:, 2]
        with open(path, "wb") as f:
            f.write((PLY_HEADER % len(v)).encode()); v.tofile(f)
        return len(v)

    def save_poses(self, path):
        """One line per keyframe: video frame index, camera position x y z, rotation quaternion qx qy qz qw (camera-to-world, OpenCV axes)."""
        with open(path, "w") as f:
            for i, T in zip(self.frame_ids, self.poses):
                f.write(" ".join("%.8f" % v for v in [i, *T[:3, 3], *Rotation.from_matrix(T[:3, :3]).as_quat()]) + "\n")

    def save(self, out_dir, voxel=0.0):
        """map.ply, poses.txt and loops.json in out_dir."""
        os.makedirs(out_dir, exist_ok=True); n = self.save_ply(os.path.join(out_dir, "map.ply"), voxel); self.save_poses(os.path.join(out_dir, "poses.txt"))
        json.dump(self.loops, open(os.path.join(out_dir, "loops.json"), "w"), indent=1)
        return n
