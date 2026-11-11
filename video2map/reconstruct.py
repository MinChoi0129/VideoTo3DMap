"""VGGT-Omega reconstruction of a group of keyframes, resampled on the output pixel grid."""
import numpy as np
import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


class Window:
    """One reconstruction in its own coordinates: per keyframe depth along z, confidence, camera-to-window pose, intrinsics and points."""

    def __init__(self, idx, depth, conf, c2w, K):
        self.idx, self.depth, self.conf, self.c2w, self.K, self.pts = list(idx), depth, conf, c2w, K, None

    def row(self, k):
        return self.idx.index(k)


class OmegaReconstructor:
    def __init__(self, checkpoint, resolution=512, device="cuda"):
        self.resolution, self.device = resolution, device
        self.model = VGGTOmega(autocast=device.startswith("cuda")).to(device).eval()
        sd = torch.load(checkpoint, map_location="cpu"); sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
        self.model.load_state_dict(sd, strict=False)

    def load(self, paths):
        """Network input of image files (CPU tensor, S x 3 x h x w)."""
        return load_and_preprocess_images(paths, image_resolution=self.resolution)

    def set_grid(self, src_hw, model_hw, out_width):
        """Output grid: out_width wide with the aspect of the video (0 = the model grid); each output pixel reads its nearest model pixel."""
        (H, W), (h, w) = src_hw, model_hw
        if out_width:
            w2 = out_width; h2 = round(H * w2 / W / 14) * 14
            u, v = (np.arange(w2) + 0.5) * W / w2, (np.arange(h2) + 0.5) * H / h2   # source pixel centres
        else:
            u, v = (np.arange(w) + 0.5) * W / w, (np.arange(h) + 0.5) * H / h
        self.UM, self.VM = np.meshgrid(np.clip(np.floor(u * w / W).astype(int), 0, w - 1), np.clip(np.floor(v * h / H).astype(int), 0, h - 1))
        self.src_uv = (np.clip(u.astype(int), 0, W - 1), np.clip(v.astype(int), 0, H - 1))   # for colours

    @torch.inference_mode()
    def __call__(self, images, idx):
        p = self.model(images.to(self.device))
        ex, K = encoding_to_camera(p["pose_enc"], images.shape[-2:])
        d, cf = p["depth"], p["depth_conf"]; d = d[0] if d.dim() == 5 else d; d = d[..., 0] if d.shape[-1] == 1 else d; cf = cf[0] if cf.dim() == 4 else cf
        ex = (ex[0] if ex.dim() == 4 else ex).float().cpu().numpy()[:, :3, :4]; K = (K[0] if K.dim() == 4 else K).float().cpu().numpy()
        d, cf = d.float().cpu().numpy()[:, self.VM, self.UM], cf.float().cpu().numpy()[:, self.VM, self.UM]
        c2w = np.tile(np.eye(4), (len(idx), 1, 1)); c2w[:, :3, :3] = np.transpose(ex[:, :, :3], (0, 2, 1)); c2w[:, :3, 3] = -np.einsum("nji,nj->ni", ex[:, :, :3], ex[:, :, 3])
        return Window(idx, d, cf, c2w, K)

    def unproject(self, win, K=None):
        """Window points through intrinsics K (S x 3 x 3, default: the predicted ones)."""
        K = win.K if K is None else K
        cam = np.stack([(self.UM[None] + 0.5 - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None], (self.VM[None] + 0.5 - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None],
                        np.ones_like(win.depth)], -1) * win.depth[..., None]
        win.pts = (np.einsum("nij,nhwj->nhwi", win.c2w[:, :3, :3], cam) + win.c2w[:, None, None, :3, 3]).astype(np.float32)
        return win
