"""
src/dataset.py — BraTSLazyDataset + PatchDataset (legacy)

BraTSLazyDataset: đọc thẳng NII on-the-fly, không cần preprocess.
PatchDataset: giữ lại nếu bạn đã có data NPY cũ.
"""

import json
import os
import random
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ['t1', 't1ce', 't2', 'flair']


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    """Z-score trên brain mask (skull-stripped: background = 0)."""
    brain = v > 0
    if brain.sum() > 100:
        mu  = v[brain].mean()
        std = v[brain].std() + 1e-8
        v   = (v - mu) / std
        v[~brain] = 0.0
    return v.astype(np.float32)


def _load_nii(patient_dir: str, pid: str, suffix: str) -> np.ndarray | None:
    for ext in ['.nii', '.nii.gz']:
        p = os.path.join(patient_dir, f'{pid}_{suffix}{ext}')
        if os.path.exists(p):
            return nib.load(p).get_fdata(dtype=np.float32)
    return None


def _sample_patch(
    imgs: np.ndarray,   # (4, D, H, W) float32
    seg:  np.ndarray,   # (D, H, W)    uint8
    patch: int  = 128,
    fg_prob: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    D, H, W = seg.shape

    # Pad nếu volume nhỏ hơn patch
    if D < patch or H < patch or W < patch:
        imgs = np.pad(imgs, [(0,0),(max(0,patch-D),0),(max(0,patch-H),0),(max(0,patch-W),0)])
        seg  = np.pad(seg,  [(max(0,patch-D),0),(max(0,patch-H),0),(max(0,patch-W),0)])
        D, H, W = seg.shape

    fg = np.argwhere(seg > 0)
    if len(fg) > 0 and random.random() < fg_prob:
        c  = fg[random.randint(0, len(fg) - 1)]
        sd = int(np.clip(c[0] - patch // 2, 0, D - patch))
        sh = int(np.clip(c[1] - patch // 2, 0, H - patch))
        sw = int(np.clip(c[2] - patch // 2, 0, W - patch))
    else:
        sd = random.randint(0, D - patch)
        sh = random.randint(0, H - patch)
        sw = random.randint(0, W - patch)

    return (
        imgs[:, sd:sd+patch, sh:sh+patch, sw:sw+patch],
        seg[     sd:sd+patch, sh:sh+patch, sw:sw+patch],
    )


# ─────────────────────────────────────────────
# BraTSLazyDataset  ← dùng cái này
# ─────────────────────────────────────────────

class BraTSLazyDataset(Dataset):
    """
    Đọc NII on-the-fly. Không cần preprocess hay lưu NPY.

    Mỗi __getitem__ trả về 1 patch ngẫu nhiên từ 1 volume.
    Volume được cache trong RAM theo LRU để tránh đọc lại disk.

    Args:
        data_dir   : thư mục chứa BraTS20_Training_001, 002, ...
        split      : 'train' | 'val'
        val_split  : tỉ lệ val (mặc định 0.2)
        n_patches  : số patches ảo mỗi volume (không lưu, chỉ để tính len)
        patch_size : kích thước patch 3D
        fg_prob    : xác suất sample tại voxel tumor (foreground)
        cache_size : số volumes giữ trong RAM (RTX 3080/4080 16GB → 80)
        mode       : 'seg' → trả (img, seg) | 'ssl' → trả img
        seed       : random seed để split train/val ổn định
    """

    def __init__(
        self,
        data_dir:   str,
        split:      Literal['train', 'val'] = 'train',
        val_split:  float = 0.2,
        n_patches:  int   = 32,
        patch_size: int   = 128,
        fg_prob:    float = 0.85,
        cache_size: int   = 80,
        mode:       Literal['seg', 'ssl'] = 'seg',
        seed:       int   = 42,
    ):
        self.data_dir   = data_dir
        self.patch_size = patch_size
        self.fg_prob    = fg_prob
        self.n_patches  = n_patches
        self.mode       = mode

        # Lấy danh sách patient
        all_pids = sorted(
            d for d in os.listdir(data_dir)
            if (Path(data_dir) / d).is_dir() and not d.startswith('.')
        )
        if len(all_pids) == 0:
            raise ValueError(f"Không tìm thấy patient nào trong: {data_dir}")

        rng = random.Random(seed)
        rng.shuffle(all_pids)
        n_val = max(1, int(len(all_pids) * val_split))

        self.pids = all_pids[:n_val] if split == 'val' else all_pids[n_val:]

        # Index ảo (pid, patch_idx) — không load gì ở đây
        self.index = [(pid, i) for pid in self.pids for i in range(n_patches)]

        # LRU volume cache
        self.cache_size = cache_size
        self._cache: dict[str, tuple] = {}
        self._order: list[str]        = []

        print(f"[BraTSLazyDataset:{split}] {len(self.pids)} volumes "
              f"× {n_patches} patches = {len(self.index)} samples "
              f"| RAM cache = {cache_size} vols")

    def __len__(self) -> int:
        return len(self.index)

    def _load_volume(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        """Load + normalize 1 patient. Kết quả cache LRU trong RAM."""
        if pid in self._cache:
            self._order.remove(pid)
            self._order.append(pid)
            return self._cache[pid]

        pdir = str(Path(self.data_dir) / pid)
        vols = []
        for mod in MODALITIES:
            v = _load_nii(pdir, pid, mod)
            if v is None:
                v = np.zeros((155, 240, 240), dtype=np.float32)
            vols.append(_normalize(v))

        img_vol = np.stack(vols, axis=0)                    # (4, D, H, W)
        seg_raw = _load_nii(pdir, pid, 'seg')
        seg_vol = (seg_raw > 0).astype(np.uint8) if seg_raw is not None \
                  else np.zeros(img_vol.shape[1:], dtype=np.uint8)

        result = (img_vol, seg_vol)

        if self.cache_size > 0:
            if len(self._order) >= self.cache_size:
                del self._cache[self._order.pop(0)]
            self._cache[pid] = result
            self._order.append(pid)

        return result

    def __getitem__(self, idx: int):
        pid, _ = self.index[idx]
        img_vol, seg_vol = self._load_volume(pid)

        # Sample patch, thử lại nếu patch quá ít brain
        for _ in range(20):
            img_p, seg_p = _sample_patch(img_vol, seg_vol, self.patch_size, self.fg_prob)
            if (img_p[1] != 0).mean() >= 0.05:
                break

        img_t = torch.from_numpy(img_p.copy())                       # (4, P, P, P)
        seg_t = torch.from_numpy(seg_p.copy()).unsqueeze(0).float()  # (1, P, P, P)

        return img_t if self.mode == 'ssl' else (img_t, seg_t)


# ─────────────────────────────────────────────
# PatchDataset — giữ lại nếu đã có NPY cũ
# ─────────────────────────────────────────────

class PatchDataset(Dataset):
    """Legacy loader cho data đã preprocess thành NPY."""

    def __init__(
        self,
        patch_dir: str,
        split:     Literal['train', 'val'] = 'train',
        mode:      Literal['seg', 'ssl']   = 'seg',
        ram_cache: bool = False,
    ):
        manifest_path = Path(patch_dir) / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {manifest_path}.\n"
                f"Hãy dùng BraTSLazyDataset — không cần preprocess."
            )
        with open(manifest_path, encoding='utf-8') as f:
            self.entries = json.load(f)[split]
        self.patch_dir = patch_dir
        self.mode      = mode
        self._cache: dict[int, tuple] | None = {} if ram_cache else None
        print(f"[PatchDataset:{split}] {len(self.entries)} patches"
              + (" (RAM cache)" if ram_cache else ""))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        if self._cache is not None and idx in self._cache:
            img_t, seg_t = self._cache[idx]
        else:
            e     = self.entries[idx]
            img   = np.load(Path(self.patch_dir) / e['img']).astype(np.float32)
            seg   = np.load(Path(self.patch_dir) / e['seg']).astype(np.float32)
            img_t = torch.from_numpy(img)
            seg_t = torch.from_numpy(seg).unsqueeze(0)
            if self._cache is not None:
                self._cache[idx] = (img_t, seg_t)

        return img_t if self.mode == 'ssl' else (img_t, seg_t)
