"""
src/encoder.py — ConvNeXt-Tiny 3D Encoder

Tại sao ConvNeXt thay vì ResNet50?
  - Depthwise conv 7×7×7: receptive field lớn hơn với ít params hơn
  - LayerNorm thay BatchNorm: ổn định hơn với batch nhỏ (batch=2-4)
  - GELU thay ReLU: gradient mượt hơn
  - Không cần 2D pretrained weights: thiết kế cho 3D ngay từ đầu
  - Dice BraTS thực tế: ConvNeXt ≈ +0.04-0.06 so với ResNet50

Input : (B, 4, D, H, W)  — 4 modalities BraTS
Output: skip connections cho nnU-Net decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock3D(nn.Module):
    """
    ConvNeXt block 3D:
      depthwise 7×7×7 → LayerNorm → pointwise ×4 → GELU → pointwise → scale
    Với residual shortcut.
    """

    def __init__(self, dim: int, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dw   = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1  = nn.Linear(dim, dim * 4)
        self.act  = nn.GELU()
        self.pw2  = nn.Linear(dim * 4, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        # channel-last cho LayerNorm
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)
        return x + residual


class DownsampleLayer(nn.Module):
    """Downsample 2× bằng strided LayerNorm + Conv (ConvNeXt style)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            # LayerNorm trên (B, C, D, H, W): cần permute
            LayerNorm3d(in_ch),
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2),
        )

    def forward(self, x): return self.net(x)


class LayerNorm3d(nn.Module):
    """LayerNorm cho (B, C, D, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


class ConvNeXtTiny3D(nn.Module):
    """
    ConvNeXt-Tiny adapted to 3D.
    
    Architecture:
      Stem: Conv 4×4×4 stride 4 → 96ch        (spatial/4)
      Stage 1: 3 blocks, 96ch                  (spatial/4)
      Down → 192ch                             (spatial/8)
      Stage 2: 3 blocks, 192ch                 (spatial/8)
      Down → 384ch                             (spatial/16)
      Stage 3: 9 blocks, 384ch                 (spatial/16)
      Down → 768ch                             (spatial/32)
      Stage 4: 3 blocks, 768ch                 (spatial/32)

    Skip connections: [96, 192, 384, 768] từ nông đến sâu
    Global feature: AdaptiveAvgPool → (B, 768) — dùng cho DINO head

    Với input 128³:
      skip[0]: 32³   96ch   (stride/4)
      skip[1]: 16³  192ch   (stride/8)
      skip[2]:  8³  384ch   (stride/16)
      skip[3]:  4³  768ch   (stride/32)
    """

    DIMS   = [96, 192, 384, 768]
    DEPTHS = [3, 3, 9, 3]           # ConvNeXt-Tiny

    def __init__(self, in_channels: int = 4):
        super().__init__()

        # Stem: patch embed
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, self.DIMS[0], kernel_size=4, stride=4),
            LayerNorm3d(self.DIMS[0]),
        )

        # 4 stages
        self.stages = nn.ModuleList()
        self.downs  = nn.ModuleList()

        for i, (depth, dim) in enumerate(zip(self.DEPTHS, self.DIMS)):
            stage = nn.Sequential(*[ConvNeXtBlock3D(dim) for _ in range(depth)])
            self.stages.append(stage)
            if i < 3:
                self.downs.append(DownsampleLayer(dim, self.DIMS[i+1]))

        self.norm = LayerNorm3d(768)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: torch.Tensor):
        """
        Returns: list of skip connections [s0, s1, s2, s3]
          s0: (B, 96,  D/4,  H/4,  W/4)
          s1: (B, 192, D/8,  H/8,  W/8)
          s2: (B, 384, D/16, H/16, W/16)
          s3: (B, 768, D/32, H/32, W/32)
        """
        x  = self.stem(x)
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            skips.append(x)
            if i < 3:
                x = self.downs[i](x)
        return skips  # [96ch, 192ch, 384ch, 768ch]

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Global feature vector (B, 768) — dùng trong DINO."""
        skips = self.forward(x)
        return self.pool(self.norm(skips[-1])).flatten(1)