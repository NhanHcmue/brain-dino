"""
src/models.py — ConvNeXtNNUNet: ConvNeXt-Tiny 3D Encoder + nnU-Net Decoder

LỖI ĐÃ SỬA: File cũ chứa nội dung của dino.py (copy nhầm).
             File này phải chứa ConvNeXtNNUNet để train_seg.py import được.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class nnUNetConvBlock(nn.Module):
    """Double Conv + InstanceNorm + LeakyReLU với residual shortcut."""

    def __init__(self, in_ch: int, out_ch: int, p_dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(p_dropout) if p_dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) \
                    if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class DecoderBlock(nn.Module):
    """Upsample 2× → concat skip → nnUNetConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nnUNetConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='trilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ConvNeXtNNUNet(nn.Module):
    """
    ConvNeXt-Tiny 3D Encoder + nnU-Net Style Decoder.

    Encoder skips từ ConvNeXtTiny3D: [96@D/4, 192@D/8, 384@D/16, 768@D/32]

    Decoder:
      Bottleneck: 768ch → 768ch
      dec3: up + cat(384) → 384ch
      dec2: up + cat(192) → 192ch
      dec1: up + cat(96)  → 96ch
      up×4 (stem stride=4) → input size
      head 1×1 → num_classes

    Deep supervision (training only):
      ds3, ds2, ds1 → upsample về input size
      Loss weights: 1.0, 0.4, 0.2, 0.1
    """

    ENC_CHANNELS = [96, 192, 384, 768]

    def __init__(
        self,
        encoder,
        num_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        ch = self.ENC_CHANNELS

        self.bottleneck = nnUNetConvBlock(ch[3], ch[3], p_dropout=dropout)

        self.dec3 = DecoderBlock(ch[3], ch[2], ch[2])
        self.dec2 = DecoderBlock(ch[2], ch[1], ch[1])
        self.dec1 = DecoderBlock(ch[1], ch[0], ch[0])

        # Upsample ×4 để về kích thước input (stem đã stride 4)
        self.up_final = nn.Sequential(
            nn.ConvTranspose3d(ch[0], 32, kernel_size=2, stride=2),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.head = nn.Conv3d(16, num_classes, kernel_size=1)

        # Deep supervision heads
        self.ds3 = nn.Conv3d(ch[2], num_classes, kernel_size=1)
        self.ds2 = nn.Conv3d(ch[1], num_classes, kernel_size=1)
        self.ds1 = nn.Conv3d(ch[0], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]

        skips = self.encoder(x)          # [96ch, 192ch, 384ch, 768ch]
        s0, s1, s2, s3 = skips

        b  = self.bottleneck(s3)         # 768ch

        d3 = self.dec3(b,  s2)           # 384ch
        d2 = self.dec2(d3, s1)           # 192ch
        d1 = self.dec1(d2, s0)           # 96ch

        out = self.head(self.up_final(d1))  # → input_size

        if self.training:
            ds3 = F.interpolate(self.ds3(d3), size=input_size,
                                mode='trilinear', align_corners=False)
            ds2 = F.interpolate(self.ds2(d2), size=input_size,
                                mode='trilinear', align_corners=False)
            ds1 = F.interpolate(self.ds1(d1), size=input_size,
                                mode='trilinear', align_corners=False)
            return out, ds3, ds2, ds1

        return out