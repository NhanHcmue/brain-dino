"""
src/dataset.py — Load NPY patches cho DINO pretraining và UNet segmentation
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """
    Load preprocessed .npy patches.
    
    __getitem__ = 2× np.load() → cực nhanh (< 5ms/patch vs ~1.8s với nib.load)
    
    Args:
        patch_dir : output của preprocess.py
        split     : 'train' hoặc 'val'
        mode      : 'seg' → (img, seg) | 'ssl' → img only (cho DINO)
        ram_cache : load tất cả vào RAM nếu RAM >= 32GB
    """

    def __init__(
        self,
        patch_dir: str,
        split: str = 'train',
        mode: str = 'seg',
        ram_cache: bool = False,
    ):
        self.root      = Path(patch_dir)
        self.mode      = mode
        self.ram_cache = ram_cache

        manifest = json.loads((self.root / 'manifest.json').read_text())
        self.entries: List[dict] = manifest[split]
        assert len(self.entries) > 0, f'Không có patch trong split="{split}"'

        self._img_cache: dict = {}
        self._seg_cache: dict = {}

        if ram_cache:
            print(f'[{split}] Caching {len(self.entries)} patches vào RAM...')
            for e in self.entries:
                self._img_cache[e['img']] = np.load(self.root / e['img']).astype(np.float32)
                self._seg_cache[e['seg']] = np.load(self.root / e['seg']).astype(np.float32)
            img_mb = sum(v.nbytes for v in self._img_cache.values()) / 1e6
            seg_mb = sum(v.nbytes for v in self._seg_cache.values()) / 1e6
            print(f'  RAM used: {(img_mb+seg_mb)/1e3:.1f} GB')

        print(f'[PatchDataset:{split}:{mode}] {len(self.entries)} patches '
              f'| {"RAM" if ram_cache else "disk"}')

    def __len__(self) -> int:
        return len(self.entries)

    def _load_img(self, key: str) -> np.ndarray:
        if self.ram_cache:
            return self._img_cache[key]
        return np.load(self.root / key).astype(np.float32)  # fp16→fp32 on load

    def _load_seg(self, key: str) -> np.ndarray:
        if self.ram_cache:
            return self._seg_cache[key]
        return np.load(self.root / key).astype(np.float32)

    def __getitem__(self, idx: int):
        e   = self.entries[idx]
        img = self._load_img(e['img'])  # (4, D, H, W) float32
        img_t = torch.from_numpy(img)

        if self.mode == 'ssl':
            return img_t  # DINO chỉ cần image

        seg = self._load_seg(e['seg'])  # (D, H, W) float32
        seg_t = torch.from_numpy(seg).unsqueeze(0)  # (1, D, H, W)
        return img_t, seg_t