"""
encoder.py — Frozen visual backbone for SMGW.

The encoder produces z_t ∈ R^{raw_dim}, used ONLY as observation context
for the manager and worker. It is NEVER used to define subgoals or to
measure option success. That job belongs to task_spec.task_error().

Supports R3M (ResNet-50, 2048-d) and DINOv2 (ViT-S/14, 384-d).
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

from config import EncoderConfig


class VisualEncoder(nn.Module):
    def __init__(self, config: EncoderConfig, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.preprocess = T.Compose([
            T.Resize((config.img_size, config.img_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if config.name == "r3m":
            self.backbone = self._load_r3m()
        elif config.name == "dinov2":
            self.backbone = self._load_dinov2()
        else:
            raise ValueError(f"Unknown encoder: {config.name}")

        if config.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.to(device)

    # -----------------------------------------------------------------

    def _load_r3m(self) -> nn.Module:
        try:
            from r3m import load_r3m
            model = load_r3m("resnet50")
            model.eval()
            return model
        except ImportError:
            print("  [Encoder] WARNING: R3M not installed; using ImageNet "
                  "ResNet-50 as placeholder (features will be less "
                  "manipulation-specific).")
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            return nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())

    def _load_dinov2(self) -> nn.Module:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model.eval()
        return model

    # -----------------------------------------------------------------

    def encode_raw(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.config.name == "r3m":
                x = F.interpolate(images,
                                  size=(self.config.img_size, self.config.img_size),
                                  mode='bilinear', align_corners=False)
                x = x * 255.0
                feats = self.backbone(x)
                if isinstance(feats, dict):
                    feats = feats['embedding']
                if isinstance(feats, tuple):
                    feats = feats[0]
            else:
                feats = self.backbone(self.preprocess(images))
        return feats.detach()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_raw(images)

    def encode_numpy(self, images_np: np.ndarray) -> np.ndarray:
        """images_np: (H,W,3) or (B,H,W,3) uint8 -> (B, raw_dim) float32."""
        if images_np.ndim == 3:
            images_np = images_np[np.newaxis]
        images_t = torch.from_numpy(images_np).float() / 255.0
        images_t = images_t.permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            z = self.forward(images_t)
        return z.cpu().numpy().astype(np.float32)

    def get_trainable_params(self):
        return []
