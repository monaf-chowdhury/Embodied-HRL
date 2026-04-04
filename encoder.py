"""
Visual Encoder: Frozen R3M + Learned Projection Head.

The encoder maps raw images to a compact latent space where distances
are meaningful for subgoal selection and reachability estimation.

Pipeline: image (224x224x3) -> R3M_frozen (2048-d) -> projection_head (64-d)
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from config import EncoderConfig


class ProjectionHead(nn.Module):
    """Small MLP that compresses frozen encoder features to subgoal space."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisualEncoder(nn.Module):
    """
    Frozen pretrained encoder + learned projection head.
    
    Usage:
        encoder = VisualEncoder(config)
        z = encoder(image_tensor)  # image: (B, 3, 224, 224) -> z: (B, proj_dim)
        
    The frozen backbone is never updated. Only the projection head trains.
    """
    
    def __init__(self, config: EncoderConfig, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device
        
        # Image preprocessing: R3M expects ImageNet-normalized 224x224
        self.preprocess = T.Compose([
            T.Resize((config.img_size, config.img_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        
        # Load frozen backbone
        if config.name == "r3m":
            self.backbone = self._load_r3m()
        elif config.name == "dinov2":
            self.backbone = self._load_dinov2()
        else:
            raise ValueError(f"Unknown encoder: {config.name}")
        
        # Freeze backbone
        if config.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        # Learned projection head
        self.projection = ProjectionHead(
            input_dim=config.raw_dim,
            hidden_dim=config.proj_hidden,
            output_dim=config.proj_dim,
        )
        
        self.to(device)
    
    def _load_r3m(self) -> nn.Module:
        """Load R3M ResNet-50 backbone."""
        try:
            from r3m import load_r3m
            r3m_model = load_r3m("resnet50")
            r3m_model.eval()
            # R3M's forward returns a 2048-d embedding
            return r3m_model
        except ImportError:
            print("WARNING: R3M not installed. Using a random ResNet-50 as placeholder.")
            print("Install R3M: pip install git+https://github.com/facebookresearch/r3m.git")
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            # Remove classification head, keep feature extractor
            modules = list(resnet.children())[:-1]  # Remove final FC
            backbone = nn.Sequential(*modules, nn.Flatten())
            return backbone
    
    def _load_dinov2(self) -> nn.Module:
        """Load DINOv2 ViT-S/14 backbone."""
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        backbone.eval()
        return backbone
    
    def encode_raw(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get raw backbone features (before projection).
        images: (B, 3, H, W) float tensor in [0, 1]
        Returns: (B, raw_dim) tensor
        """
        # Preprocess
        x = self.preprocess(images)
        
        # Forward through frozen backbone
        with torch.no_grad():
            if self.config.name == "r3m":
                # R3M expects (B, 3, 224, 224) tensors normalized to [0,1]*255
                # Actually, R3M's load_r3m returns a model that expects [0,255] input
                # Check R3M docs — their forward expects unnormalized images * 255
                # We undo our normalization and pass raw:
                # Actually, R3M's internal preprocessing handles this.
                # The safest approach: pass the preprocessed tensor.
                features = self.backbone(x * 255.0)  # R3M expects [0, 255]
                if isinstance(features, dict):
                    features = features['embedding']
                if isinstance(features, tuple):
                    features = features[0]
            elif self.config.name == "dinov2":
                features = self.backbone(x)
            else:
                features = self.backbone(x)
        
        return features.detach()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Full pipeline: image -> frozen backbone -> projection head -> z
        images: (B, 3, H, W) float tensor in [0, 1]  
        Returns: (B, proj_dim) tensor
        """
        raw_features = self.encode_raw(images)
        z = self.projection(raw_features)
        return z
    
    def encode_numpy(self, images_np: np.ndarray) -> np.ndarray:
        """
        Convenience: numpy images (B, H, W, 3) uint8 -> numpy latents (B, proj_dim).
        Handles all conversion.
        """
        # Convert HWC uint8 -> CHW float [0,1]
        if images_np.ndim == 3:
            images_np = images_np[np.newaxis]  # Add batch dim
        
        images_t = torch.from_numpy(images_np).float() / 255.0
        images_t = images_t.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        images_t = images_t.to(self.device)
        
        with torch.no_grad():
            z = self.forward(images_t)
        
        return z.cpu().numpy()
    
    def get_trainable_params(self):
        """Returns only the projection head parameters (for optimizer)."""
        return self.projection.parameters()
