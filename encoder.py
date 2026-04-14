"""
Visual Encoder: Frozen R3M backbone only. NO projection head.

Why projection was dropped:
  R3M was pretrained on millions of manipulation videos with a temporal
  contrastive loss. Its 2048-d output already has the structure we need:
  temporally close frames are close, semantically different states are far.
  The projection head trained only on 200 random-walk episodes was
  compressing this structure away and creating a distribution mismatch
  between demo latents and replay latents.

Output: raw R3M features, L2-normalised onto the unit hypersphere.
  - All distances live in [0, 2].
  - Demo frames and replay frames live in the SAME manifold.
  - No training needed. No warmup pretraining needed.
  - Success threshold calibration is stable and meaningful.

Expected step-to-step distances (L2-normalised R3M):
  mean ~0.05–0.20 per step for Franka Kitchen random walk.
  First-to-last distance over 280 steps: ~0.3–1.5.
  These are verified by the diagnose_distances() method.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import EncoderConfig


class VisualEncoder(nn.Module):
    """
    Frozen R3M backbone. No projection head. L2-normalised output.

    Usage:
        encoder = VisualEncoder(config, device='cuda')
        z = encoder.encode_numpy(img_uint8)  # (B, 2048) float32, L2-normed
    """

    def __init__(self, config: EncoderConfig, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device

        if config.name == "r3m":
            self.backbone = self._load_r3m()
        elif config.name == "dinov2":
            self.backbone = self._load_dinov2()
        else:
            raise ValueError(f"Unknown encoder: {config.name}")

        # Freeze — never updated during training
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.to(device)

    def _load_r3m(self) -> nn.Module:
        try:
            from r3m import load_r3m
            model = load_r3m("resnet50")
            model.eval()
            print("  [Encoder] Loaded R3M ResNet-50 → 2048-d, L2-normalised")
            return model
        except ImportError:
            print("WARNING: R3M not installed. Using pretrained ResNet-50.")
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]
            return nn.Sequential(*modules, nn.Flatten())

    def _load_dinov2(self) -> nn.Module:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model.eval()
        print("  [Encoder] Loaded DINOv2 ViT-S/14 → 384-d, L2-normalised")
        return model

    def _encode_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, H, W) float in [0, 1]
        Returns: (B, raw_dim) L2-normalised — distances in [0, 2]
        """
        with torch.no_grad():
            if self.config.name == "r3m":
                x = F.interpolate(
                    images,
                    size=(self.config.img_size, self.config.img_size),
                    mode='bilinear', align_corners=False,
                )
                x = x * 255.0   # R3M expects [0, 255]
                features = self.backbone(x)
                if isinstance(features, dict):
                    features = features['embedding']
                if isinstance(features, tuple):
                    features = features[0]
            else:
                # DINOv2: ImageNet normalisation
                mean = torch.tensor([0.485, 0.456, 0.406],
                                    device=images.device).view(1, 3, 1, 1)
                std  = torch.tensor([0.229, 0.224, 0.225],
                                    device=images.device).view(1, 3, 1, 1)
                features = self.backbone((images - mean) / std)

        # L2 normalise → unit hypersphere, distances in [0, 2]
        features = F.normalize(features, dim=-1)
        return features.detach()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B,3,H,W) float[0,1] → (B, raw_dim) L2-norm"""
        return self._encode_tensor(images)

    def encode_numpy(self, images_np: np.ndarray) -> np.ndarray:
        """
        (H,W,3) uint8 or (B,H,W,3) uint8 → (B, raw_dim) float32 L2-norm
        All values in [-1,1], all distances in [0, 2].
        """
        if images_np.ndim == 3:
            images_np = images_np[np.newaxis]
        t = torch.from_numpy(images_np).float() / 255.0
        t = t.permute(0, 3, 1, 2).to(self.device)
        return self._encode_tensor(t).cpu().numpy()

    def get_trainable_params(self):
        """No trainable parameters."""
        return iter([])

    def diagnose_distances(self, env, n_steps: int = 200) -> dict:
        """
        Run n_steps random actions in env and report L2 step distances.
        Expected for L2-normalised R3M: mean ~0.05–0.20.
        Call once after warmup to verify latent space is sensible.
        """
        dists = []
        obs_img = env.reset()
        z_prev  = self.encode_numpy(obs_img).squeeze()
        for _ in range(n_steps):
            action = env.action_space.sample()
            next_img, _, done, _ = env.step(action)
            z_next = self.encode_numpy(next_img).squeeze()
            dists.append(float(np.linalg.norm(z_next - z_prev)))
            z_prev = z_next
            if done:
                obs_img = env.reset()
                z_prev  = self.encode_numpy(obs_img).squeeze()
        d = np.array(dists)
        result = {
            'mean': float(d.mean()), 'std': float(d.std()),
            'p10':  float(np.percentile(d, 10)),
            'p50':  float(np.percentile(d, 50)),
            'p90':  float(np.percentile(d, 90)),
            'max':  float(d.max()),
        }
        print(f"  [Encoder Diagnosis] Step distances over {n_steps} random steps:")
        print(f"    mean={result['mean']:.4f}  std={result['std']:.4f}  "
              f"p10={result['p10']:.4f}  p50={result['p50']:.4f}  "
              f"p90={result['p90']:.4f}  max={result['max']:.4f}")
        if result['mean'] < 0.01:
            print("  WARNING: distances too small — encoder may be collapsing.")
        elif result['mean'] > 0.5:
            print("  WARNING: distances very large — check normalisation.")
        else:
            print("  OK: distances in expected range for L2-normalised R3M.")
        return result
