"""Loop closure detection of VGGT-SLAM 2.0: SALAD image retrieval, then VGGT-1B attention match ratio on the two images."""
import os

import torch
import torchvision.transforms as T
from salad.eval import load_model
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


class LoopDetector:
    def __init__(self, salad_checkpoint=None, device="cuda"):
        ckpt = salad_checkpoint or os.path.join(torch.hub.get_dir(), "checkpoints", "dino_salad.ckpt")   # where VGGT-SLAM 2.0 keeps it
        self.device, self.salad = device, load_model(ckpt).to(device).eval()
        self.salad_in = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.vggt = VGGT(); self.vggt.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_URL)); self.vggt = self.vggt.eval().to(torch.bfloat16).to(device)

    def load(self, paths):
        """VGGT input of image files (518 wide), used for both retrieval and verification as in VGGT-SLAM 2.0."""
        return torch.cat([load_and_preprocess_images(paths[i:i + 64]) for i in range(0, len(paths), 64)])

    @torch.no_grad()
    def descriptors(self, images):
        return torch.cat([self.salad(self.salad_in(images[i:i + 64]).to(self.device)) for i in range(0, len(images), 64)]).float().cpu()

    @torch.no_grad()
    def match_ratio(self, img_a, img_b):
        return float(self.vggt(torch.stack([img_a, img_b]).to(self.device), compute_similarity=True)["image_match_ratio"].mean())

    def find(self, desc, images, windows, retrieval_threshold, match_ratio):
        """For every window, its new keyframes are queried against the keyframes before the previous window; the closest pair (SALAD L2) is
        a candidate below retrieval_threshold and is kept when the VGGT-1B match ratio reaches match_ratio (one loop per window)."""
        out = []
        for w in range(1, len(windows)):
            first = windows[w - 1][0]; new = [k for k in windows[w] if k not in windows[w - 1]]
            if first == 0 or not new:
                continue
            dist = torch.cdist(desc[new], desc[:first]); q, m = divmod(int(dist.argmin()), dist.shape[1]); q, best = new[q], float(dist.min())
            if best >= retrieval_threshold:
                continue
            ratio = self.match_ratio(images[q], images[m])
            out.append({"window": w, "query": q, "match": m, "salad_distance": round(best, 4), "match_ratio": round(ratio, 4), "accepted": ratio >= match_ratio})
        return out
