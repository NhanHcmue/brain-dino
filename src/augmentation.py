"""
src/augmentation.py — GPU Augmentation 3D
  - GPUAugmentation3D : dùng trong seg training
  - DINOMultiCrop3D   : tạo 2 global + N local crops cho DINO
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPUAugmentation3D(nn.Module):
    """
    Augmentation batched trên GPU. Input/Output: (B, C, D, H, W).
    Không có CPU overhead — GPU không chờ CPU.
    """

    def __init__(self, p_flip=0.5, p_noise=0.5, p_crop=0.5,
                 p_cutout=0.4, p_intensity=0.8):
        super().__init__()
        self.p_flip      = p_flip
        self.p_noise     = p_noise
        self.p_crop      = p_crop
        self.p_cutout    = p_cutout
        self.p_intensity = p_intensity

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._flip(x)
        x = self._rot90(x)
        x = self._intensity(x)
        x = self._noise(x)
        x = self._crop_resize(x)
        x = self._cutout(x)
        return x

    def _flip(self, x):
        for d in [2, 3, 4]:
            if torch.rand(1).item() < self.p_flip:
                x = x.flip(d)
        return x

    def _rot90(self, x):
        k     = torch.randint(0, 4, (1,)).item()
        plane = random.choice([(2, 3), (2, 4), (3, 4)])
        return torch.rot90(x, k=k, dims=plane) if k > 0 else x

    def _intensity(self, x):
        if torch.rand(1).item() < self.p_intensity:
            # Per-channel scale + shift (simulate multi-modal intensity variation)
            B, C = x.shape[:2]
            sc = torch.empty(B, C, 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
            sh = torch.empty(B, C, 1, 1, 1, device=x.device).uniform_(-0.2, 0.2)
            x  = x * sc + sh
        return x

    def _noise(self, x):
        if torch.rand(1).item() < self.p_noise:
            sigma = torch.empty(1, device=x.device).uniform_(0.0, 0.1).item()
            x = x + torch.randn_like(x) * sigma
        return x

    def _crop_resize(self, x):
        if torch.rand(1).item() < self.p_crop:
            B, C, D, H, W = x.shape
            r  = torch.empty(1).uniform_(0.75, 0.95).item()
            cd = max(1, int(D * r))
            ch = max(1, int(H * r))
            cw = max(1, int(W * r))
            sd = torch.randint(0, D - cd + 1, (1,)).item()
            sh = torch.randint(0, H - ch + 1, (1,)).item()
            sw = torch.randint(0, W - cw + 1, (1,)).item()
            x  = F.interpolate(
                x[:, :, sd:sd+cd, sh:sh+ch, sw:sw+cw],
                size=(D, H, W), mode='trilinear', align_corners=False,
            )
        return x

    def _cutout(self, x):
        if torch.rand(1).item() < self.p_cutout:
            B, C, D, H, W = x.shape
            cd = max(1, D // 5)
            ch = max(1, H // 5)
            cw = max(1, W // 5)
            sd = torch.randint(0, D - cd + 1, (1,)).item()
            sh = torch.randint(0, H - ch + 1, (1,)).item()
            sw = torch.randint(0, W - cw + 1, (1,)).item()
            x  = x.clone()
            x[:, :, sd:sd+cd, sh:sh+ch, sw:sw+cw] = 0.0
        return x


class DINOMultiCrop3D(nn.Module):
    """
    Tạo multi-scale crops cho DINO:
      - 2 global crops  (96% patch size)  → teacher + student thấy
      - n_local crops   (50% patch size)  → chỉ student thấy

    DINO học từ sự bất đối xứng global-local này.
    """

    def __init__(
        self,
        global_scale=(0.85, 1.0),
        local_scale=(0.4, 0.6),
        n_local=4,
    ):
        super().__init__()
        self.aug         = GPUAugmentation3D()
        self.global_scale = global_scale
        self.local_scale  = local_scale
        self.n_local      = n_local

    @torch.no_grad()
    def _crop(self, x, scale):
        B, C, D, H, W = x.shape
        r  = torch.empty(1).uniform_(*scale).item()
        cd = max(1, int(D * r))
        ch = max(1, int(H * r))
        cw = max(1, int(W * r))
        sd = torch.randint(0, max(1, D - cd + 1), (1,)).item()
        sh = torch.randint(0, max(1, H - ch + 1), (1,)).item()
        sw = torch.randint(0, max(1, W - cw + 1), (1,)).item()
        cropped = x[:, :, sd:sd+cd, sh:sh+ch, sw:sw+cw]
        return F.interpolate(cropped, size=(D, H, W),
                             mode='trilinear', align_corners=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Returns: list of views
          [global_1, global_2, local_1, ..., local_n]
        """
        views = []
        # 2 global crops
        for _ in range(2):
            v = self._crop(x, self.global_scale)
            v = self.aug(v)
            views.append(v)
        # n local crops
        for _ in range(self.n_local):
            v = self._crop(x, self.local_scale)
            v = self.aug(v)
            views.append(v)
        return views