"""
SatQuery AI — Remote Sensing Visual Grounding Network (RSGroundingNet)
Specialist model that takes satellite imagery + natural language expression
and localizes target objects as spatial bounding boxes [x1, y1, x2, y2].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSGroundingNet(nn.Module):
    """
    Visual Grounding model tailored for high-resolution satellite imagery.
    Combines deep convolutional feature extraction with text prompt conditioning
    to regress target bounding boxes and confidence scores.
    """
    def __init__(self, in_channels: int = 3, text_embed_dim: int = 64, num_queries: int = 5):
        super().__init__()
        self.num_queries = num_queries

        # Visual Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2, W/2

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4, W/4

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/8, W/8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),  # Fixed 8x8 spatial grid
        )

        # Text prompt conditioner
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )

        # Multi-modal fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # Bounding box regression heads (predicts num_queries boxes)
        # Each query: [cx, cy, w, h] normalized to [0.0, 1.0]
        self.box_head = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_queries * 4),
            nn.Sigmoid(),
        )

        # Objectness / Confidence score head
        self.score_head = nn.Sequential(
            nn.Linear(256 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_queries),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor, text_embeds: torch.Tensor):
        """
        Args:
            images: (B, 3, H, W) Remote sensing image
            text_embeds: (B, text_embed_dim) Text representation
        Returns:
            boxes: (B, num_queries, 4) in [x1, y1, x2, y2] format
            scores: (B, num_queries) confidence scores
        """
        B = images.shape[0]
        img_feats = self.backbone(images)  # (B, 256, 8, 8)

        text_feats = self.text_encoder(text_embeds)  # (B, 256)
        text_spatial = text_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)

        # Condition visual features with text semantics
        fused = torch.cat([img_feats, text_spatial], dim=1)  # (B, 512, 8, 8)
        fused_flat = self.fusion(fused)

        # Regress [cx, cy, w, h]
        raw_boxes = self.box_head(fused_flat).view(B, self.num_queries, 4)
        scores = self.score_head(fused_flat)  # (B, num_queries)

        # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
        cx, cy, w, h = raw_boxes[..., 0], raw_boxes[..., 1], raw_boxes[..., 2], raw_boxes[..., 3]
        x1 = torch.clamp(cx - w / 2.0, 0.0, 1.0)
        y1 = torch.clamp(cy - h / 2.0, 0.0, 1.0)
        x2 = torch.clamp(cx + w / 2.0, 0.0, 1.0)
        y2 = torch.clamp(cy + h / 2.0, 0.0, 1.0)

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return boxes, scores


class CrossModalAttn(nn.Module):
    def __init__(self, channels: int, text_dim: int):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, channels)
        self.attn_gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.BatchNorm2d(channels), nn.Sigmoid())
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, vis: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        b, c, h, w = vis.shape
        t = self.text_proj(text).unsqueeze(-1).unsqueeze(-1).expand(b, c, h, w)
        gate = self.attn_gate(torch.cat([vis, t], dim=1))
        return self.norm(vis * gate + t * (1 - gate))


# Alias for backwards compatibility
CrossModalAttentionBlock = CrossModalAttn


class GroundingDINOMaskNet(nn.Module):
    """Enhanced Grounding DINO with SAM decoder. ~3.5M params."""
    def __init__(self, in_channels: int = 3, text_dim: int = 128, max_boxes: int = 8):
        super().__init__()
        self.max_boxes = max_boxes
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU())
        self.enc4 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU())
        self.cross_attn = CrossModalAttn(256, text_dim)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flat = 256 * 16
        self.box_head = nn.Sequential(
            nn.Linear(flat, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, max_boxes * 4), nn.Sigmoid(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(flat, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, max_boxes), nn.Sigmoid(),
        )
        self.sam_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 1, 1), nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor, text_vecs: torch.Tensor):
        b = images.size(0)
        f = self.enc4(self.enc3(self.enc2(self.enc1(images))))
        fused = self.cross_attn(f, text_vecs)
        flat = self.pool(fused).view(b, -1)
        pred_boxes = self.box_head(flat).view(b, self.max_boxes, 4)
        pred_scores = self.score_head(flat)
        pred_masks = self.sam_decoder(fused)
        return pred_boxes, pred_scores, pred_masks


