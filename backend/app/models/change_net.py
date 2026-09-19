"""
SatQuery AI — Siamese Change Detection Network (SiameseChangeNet)
Specialist deep learning architecture for bi-temporal remote sensing change detection.
Takes paired multi-temporal rasters (T1, T2) and predicts a pixel-level change mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard Conv-BatchNorm-LeakyReLU block with optional residual connection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + res)
        return out


class SiameseChangeNet(nn.Module):
    """
    Siamese convolutional network for satellite change detection.
    Extracts deep semantic representations from date T1 and date T2 using shared weights,
    computes multi-scale differential features, and decodes a high-resolution change mask.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        # Shared-weight encoder
        self.enc1 = ConvBlock(in_channels, 32, stride=1)     # H, W
        self.enc2 = ConvBlock(32, 64, stride=2)              # H/2, W/2
        self.enc3 = ConvBlock(64, 128, stride=2)             # H/4, W/4
        self.enc4 = ConvBlock(128, 256, stride=2)            # H/8, W/8

        # Difference & Fusion bottleneck
        # Computes: concat(|f1 - f2|, f1 * f2, f1, f2) -> 256 * 4 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32 + 32, 32)

        # Final change probability head
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward_single(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e1, e2, e3, e4

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t1: (B, C, H, W) Image at time t1
            t2: (B, C, H, W) Image at time t2
        Returns:
            (B, 1, H, W) Binary change probability mask [0.0, 1.0]
        """
        e1_1, e1_2, e1_3, e1_4 = self.forward_single(t1)
        e2_1, e2_2, e2_3, e2_4 = self.forward_single(t2)

        # Multi-scale feature difference
        diff4 = torch.abs(e1_4 - e2_4)
        prod4 = e1_4 * e2_4
        fused4 = self.fusion(torch.cat([diff4, prod4, e1_4, e2_4], dim=1))

        # Decode with skip connections from differential features
        diff3 = torch.abs(e1_3 - e2_3)
        d3 = self.up3(fused4)
        d3 = self.dec3(torch.cat([d3, diff3], dim=1))

        diff2 = torch.abs(e1_2 - e2_2)
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, diff2], dim=1))

        diff1 = torch.abs(e1_1 - e2_1)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, diff1], dim=1))

        change_mask = self.head(d1)
        return change_mask


def dice_bce_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Combined BCE and Dice loss for accurate change boundary segmentation."""
    bce = F.binary_cross_entropy(pred, target)

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return bce + dice


class ChangeFormerBlock(nn.Module):
    """Multi-scale difference transformer block with cross-temporal attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f1 - f2)
        prod = f1 * f2
        cat = torch.cat([f1, f2, diff, prod], dim=1)
        out = self.act(self.bn1(self.conv1(cat)))
        return self.act(self.bn2(self.conv2(out)))


class ChangeFormerNet(nn.Module):
    """
    Siamese ChangeFormer architecture for bi-temporal remote sensing change detection.
    Extracts multi-scale deep features from T1 and T2, computes differential attention,
    and decodes high-resolution pixel change probability.
    Scaled to ~0.99M parameters (2.25x capacity).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        # Scaled Shared Siamese Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(96, 192, 3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Difference Bottleneck (192 channels)
        self.diff_block = ChangeFormerBlock(channels=192)

        # Decoder with skip connections
        self.up2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(96 + 96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(48 + 48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(48, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Encoder passes
        e1_1 = self.enc1(t1)
        e2_1 = self.enc2(e1_1)
        e3_1 = self.enc3(e2_1)

        e1_2 = self.enc1(t2)
        e2_2 = self.enc2(e1_2)
        e3_2 = self.enc3(e2_2)

        # Differential attention
        diff = self.diff_block(e3_1, e3_2)

        # Decoding
        u2 = self.up2(diff)
        d2 = self.dec2(torch.cat([u2, torch.abs(e2_1 - e2_2)], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, torch.abs(e1_1 - e1_2)], dim=1))

        return self.head(d1)

