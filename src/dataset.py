"""
dataset.py — Lazy NII loader (không cần preprocess)

Đọc thẳng từ .nii/.nii.gz, patch on-the-fly, cache thông minh.

Ưu điểm so với NPY pipeline:
  - Không tốn 20-25GB disk để lưu patches
  - Setup ~2 phút thay vì 30-60 phút
  - Hoạt động native trên Kaggle Notebooks (BraTS2020 dataset có sẵn)
  - Tốc độ train tương đương nhờ prefetch + cache

Usage (Kaggle Notebook):
    from dataset import BraTSLazyDataset

    # BraTS2020 trên Kaggle: /kaggle/input/brats20-dataset-training-validation/
    train_ds = BraTSLazyDataset(
        data_dir='/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
        split='train',
        n_patches=32,
        patch_size=128,
        cache_size=200,   # cache 200 patches trong RAM (~3GB)
    )

Usage (Colab / Vast.ai với kaggle CLI):
    # Bước 1: tải dataset (chỉ 1 lần, ~2-5 phút)
    # !pip install kaggle -q
    # !kaggle datasets download -d awsaf49/brats20-dataset-training-validation
    # !unzip -q brats20-dataset-training-validation.zip -d data/

    train_ds = BraTSLazyDataset(
        data_dir='data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
        ...
    )
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


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    """Z-score trên brain mask (background = 0 sau skull-strip)."""
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


def _foreground_patch(
    imgs: np.ndarray,
    seg:  np.ndarray,
    patch: int = 128,
    fg_prob: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """
    imgs: (4, D, H, W) float32
    seg : (D, H, W)    uint8
    → img_patch (4, P, P, P), seg_patch (P, P, P)
    """
    D, H, W = seg.shape

    # Pad nếu cần
    if D < patch or H < patch or W < patch:
        pd = max(0, patch - D)
        ph = max(0, patch - H)
        pw = max(0, patch - W)
        imgs = np.pad(imgs, [(0,0),(pd,0),(ph,0),(pw,0)])
        seg  = np.pad(seg,  [(pd,0),(ph,0),(pw,0)])
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


# ──────────────────────────────────────────────────────────────
# Main dataset class
# ──────────────────────────────────────────────────────────────

class BraTSLazyDataset(Dataset):
    """
    Đọc NII on-the-fly, không cần preprocess sang NPY trước.

    Chiến lược cache:
      - cache_size > 0: giữ `cache_size` patches gần nhất trong RAM
      - Trên Kaggle T4 (12GB RAM): cache_size=150-200 là hợp lý
      - Trên A100 (40GB RAM):      cache_size=500-800

    Args:
        data_dir   : thư mục chứa các folder BraTS20_Training_xxx
        split      : 'train' | 'val'
        val_split  : tỉ lệ validation (mặc định 0.2)
        n_patches  : số patches ảo mỗi volume (không lưu thật)
        patch_size : kích thước patch 3D (mặc định 128)
        fg_prob    : xác suất sample tại voxel tumor
        cache_size : số patches cache trong RAM (0 = tắt cache)
        mode       : 'seg' | 'ssl'  (trả về (img, seg) hoặc chỉ img)
        seed       : random seed cho train/val split
    """

    def __init__(
        self,
        data_dir:   str,
        split:      Literal['train', 'val'] = 'train',
        val_split:  float = 0.2,
        n_patches:  int   = 32,
        patch_size: int   = 128,
        fg_prob:    float = 0.85,
        cache_size: int   = 200,
        mode:       Literal['seg', 'ssl'] = 'seg',
        seed:       int   = 42,
    ):
        self.data_dir   = data_dir
        self.patch_size = patch_size
        self.fg_prob    = fg_prob
        self.n_patches  = n_patches
        self.mode       = mode

        # ── Lấy danh sách patient IDs ──
        all_pids = sorted(
            d for d in os.listdir(data_dir)
            if (Path(data_dir) / d).is_dir() and not d.startswith('.')
        )

        rng = random.Random(seed)
        rng.shuffle(all_pids)
        n_val = max(1, int(len(all_pids) * val_split))

        if split == 'val':
            self.pids = all_pids[:n_val]
        else:
            self.pids = all_pids[n_val:]

        # ── Bảng index ảo: (pid, patch_idx) ──
        # Không load gì ở đây — chỉ tạo index
        self.index = [(pid, i) for pid in self.pids for i in range(n_patches)]

        # ── LRU Cache ──
        self.cache_size = cache_size
        self._vol_cache: dict[str, tuple] = {}   # pid → (imgs_vol, seg_vol)
        self._vol_order: list[str]        = []   # LRU order

        print(f"[BraTSLazyDataset] {split}: {len(self.pids)} volumes "
              f"× {n_patches} patches = {len(self.index)} samples "
              f"| cache={cache_size} vols")

    def __len__(self) -> int:
        return len(self.index)

    def _load_volume(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load 4 modalities + seg cho 1 patient.
        Kết quả được cache trong RAM theo LRU.
        """
        # Cache hit
        if pid in self._vol_cache:
            # cập nhật LRU order
            self._vol_order.remove(pid)
            self._vol_order.append(pid)
            return self._vol_cache[pid]

        # Cache miss → load từ disk
        pdir = str(Path(self.data_dir) / pid)
        vols = []
        for mod in MODALITIES:
            v = _load_nii(pdir, pid, mod)
            if v is None:
                # fallback: trả về zeros nếu thiếu modality
                v = np.zeros((155, 240, 240), dtype=np.float32)
            vols.append(_normalize(v))

        img_vol = np.stack(vols, axis=0).astype(np.float32)  # (4, D, H, W)

        seg_raw = _load_nii(pdir, pid, 'seg')
        if seg_raw is None:
            seg_vol = np.zeros(img_vol.shape[1:], dtype=np.uint8)
        else:
            seg_vol = (seg_raw > 0).astype(np.uint8)  # binary

        result = (img_vol, seg_vol)

        # LRU eviction
        if self.cache_size > 0:
            if len(self._vol_order) >= self.cache_size:
                oldest = self._vol_order.pop(0)
                del self._vol_cache[oldest]
            self._vol_cache[pid] = result
            self._vol_order.append(pid)

        return result

    def __getitem__(self, idx: int):
        pid, _ = self.index[idx]

        img_vol, seg_vol = self._load_volume(pid)

        # Sample patch on-the-fly
        max_tries = 20
        for _ in range(max_tries):
            img_p, seg_p = _foreground_patch(
                img_vol, seg_vol, self.patch_size, self.fg_prob
            )
            # Bỏ patch quá ít brain tissue
            if (img_p[1] != 0).mean() >= 0.05:
                break

        img_t = torch.from_numpy(img_p.copy())                         # (4, P, P, P)
        seg_t = torch.from_numpy(seg_p.copy()).unsqueeze(0).float()    # (1, P, P, P)

        if self.mode == 'ssl':
            return img_t
        return img_t, seg_t


# ──────────────────────────────────────────────────────────────
# Backward-compatible wrapper (giữ nguyên interface cũ nếu cần)
# ──────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """
    Drop-in replacement cho PatchDataset cũ (dùng manifest.json + NPY).

    Nếu bạn đã có data NPY cũ → vẫn dùng được.
    Nếu chưa có → dùng BraTSLazyDataset ở trên.
    """

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
                f"manifest.json not found at {patch_dir}.\n"
                f"Hãy dùng BraTSLazyDataset thay thế — không cần preprocess!"
            )

        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)

        self.entries   = manifest[split]
        self.patch_dir = patch_dir
        self.mode      = mode
        self._cache: dict[int, tuple] = {} if ram_cache else None

        print(f"[PatchDataset] {split}: {len(self.entries)} patches"
              + (" (RAM cache ON)" if ram_cache else ""))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        if self._cache is not None and idx in self._cache:
            img_t, seg_t = self._cache[idx]
        else:
            e = self.entries[idx]
            img = np.load(Path(self.patch_dir) / e['img']).astype(np.float32)
            seg = np.load(Path(self.patch_dir) / e['seg']).astype(np.float32)
            img_t = torch.from_numpy(img)
            seg_t = torch.from_numpy(seg).unsqueeze(0)
            if self._cache is not None:
                self._cache[idx] = (img_t, seg_t)

        if self.mode == 'ssl':
            return img_t
        return img_t, seg_t
