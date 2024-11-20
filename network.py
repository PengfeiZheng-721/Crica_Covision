import torch
from torch import nn
import torch.nn.functional as F
import math
import os

# Assuming a ViT model implementation named vit_base
from backbone.vision_transformer import vit_base

class GeM(nn.Module):
    """Generalized Mean Pooling (GeM) layer."""
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), 
                            (x.size(-2), x.size(-1))).pow(1./self.p)

class Flatten(nn.Module):
    """Flattens a multi-dimensional tensor into a 2D tensor."""
    def forward(self, x):
        return x.view(x.size(0), -1)

class CricaVPRNet(nn.Module):
    """
    CricaVPRNet model that outputs feature vectors for each image.
    """
    def __init__(self, foundation_model_path):
        super().__init__()
        self.backbone = get_backbone(foundation_model_path)
        self.aggregation = nn.Sequential(
            GeM(),
            Flatten()
        )

        # Transformer encoder layer setup
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=16,
            dim_feedforward=2048,
            activation="gelu",
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Additional linear layer to map 14*768 back to 768
        self.feature_projection = nn.Linear(14 * 768, 768)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Extracted feature vector with shape (B, 768).
        """
        x = self.backbone(x)

        B, P, D = x["x_prenorm"].shape
        W = H = int(math.sqrt(P - 1))
        x0 = x["x_norm_clstoken"]  # Shape: (B, 1, D)
        x_p = x["x_norm_patchtokens"].view(B, W, H, D).permute(0, 3, 1, 2)  # Shape: (B, D, W, H)

        # Apply aggregation layer to different regions
        x10 = self.aggregation(x_p[:, :, 0:8, 0:8])  # Shape: (B, D)
        x11 = self.aggregation(x_p[:, :, 0:8, 8:])   # Shape: (B, D)
        x12 = self.aggregation(x_p[:, :, 8:, 0:8])   # Shape: (B, D)
        x13 = self.aggregation(x_p[:, :, 8:, 8:])    # Shape: (B, D)
        x20 = self.aggregation(x_p[:, :, 0:5, 0:5])  # Shape: (B, D)
        x21 = self.aggregation(x_p[:, :, 0:5, 5:11]) # Shape: (B, D)
        x22 = self.aggregation(x_p[:, :, 0:5, 11:])  # Shape: (B, D)
        x23 = self.aggregation(x_p[:, :, 5:11, 0:5]) # Shape: (B, D)
        x24 = self.aggregation(x_p[:, :, 5:11, 5:11])# Shape: (B, D)
        x25 = self.aggregation(x_p[:, :, 5:11, 11:]) # Shape: (B, D)
        x26 = self.aggregation(x_p[:, :, 11:, 0:5])  # Shape: (B, D)
        x27 = self.aggregation(x_p[:, :, 11:, 5:11]) # Shape: (B, D)
        x28 = self.aggregation(x_p[:, :, 11:, 11:])  # Shape: (B, D)

        # Concatenate all features including x0 (CLS token)
        features = torch.cat([
            x0.squeeze(1),  # Shape: (B, D)
            x10, x11, x12, x13,
            x20, x21, x22, x23, x24, x25, x26, x27, x28
        ], dim=1)  # Shape: (B, 14*D)

        # Project features back to 768 dimensions
        projected_features = self.feature_projection(features)  # Shape: (B, 768)

        # Transformer encoding
        encoded_features = self.encoder(projected_features.unsqueeze(1)).squeeze(1)  # Shape: (B, 768)

        return encoded_features  # Feature vector for each image

def get_backbone(foundation_model_path):
    """
    Initializes the backbone network and loads pre-trained weights if provided.

    Args:
        foundation_model_path (str): Path to pre-trained weights.

    Returns:
        backbone (nn.Module): Initialized backbone network.
    """
    backbone = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0)

    # Load pre-trained weights
    if foundation_model_path is not None and os.path.exists(foundation_model_path):
        state_dict = torch.load(foundation_model_path, map_location="cpu")
        # Filter out unnecessary keys
        backbone_state_dict = backbone.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in backbone_state_dict}
        missing_keys, unexpected_keys = backbone.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys when loading backbone state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading backbone state_dict: {unexpected_keys}")
        print("Pre-trained weights for the backbone loaded successfully.")
    else:
        print("No pre-trained weights provided or file does not exist.")

    return backbone
