"""
src/dataset.py — Load BraTS2020 H5 trực tiếp (không cần preprocess.py)

Format H5 đã xác nhận (BraTS20_Training_001.h5):
  f['image']  shape (134, 172, 136)  float32  — 1 kênh đã crop
  f['label']  shape (134, 172, 136)  uint8    — segmentation mask

Hỗ trợ thêm (tự detect):
  f['image']  shape (4, D, H, W)    — 4 modalities channel-first
  f['image']  shape (D, H, W, 4)    — 4 modalities channel-last
  f['image']  shape (D, H, W)       — 1 kênh (BraTS20 H5 này)
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_volume(v: np.ndarray) -> np.ndarray:
    """Z-score trên brain voxels (> 0)."""
    brain = v > 0
    if brain.sum() > 100:
        mu  = v[brain].mean()
        std = v[brain].std() + 1e-8
        v   = (v - mu) / std
        v[~brain] = 0.0
    return v.astype(np.float32)


def extract_patch(
    img: np.ndarray,
    seg: np.ndarray,
    patch: int = 96,
    fg_prob: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random 3D patch với 85% xác suất center vào tumor.
    Tự pad nếu volume nhỏ hơn patch.
    img: (C, D, H, W), seg: (D, H, W)
    """
    C, D, H, W = img.shape
    ps = patch
    pd = max(0, ps - D); ph = max(0, ps - H); pw = max(0, ps - W)
    if pd or ph or pw:
        img = np.pad(img, [(0,0),(pd,0),(ph,0),(pw,0)])
        seg = np.pad(seg, [(pd,0),(ph,0),(pw,0)])
        C, D, H, W = img.shape

    fg = np.argwhere(seg > 0)
    if len(fg) > 0 and random.random() < fg_prob:
        c  = fg[random.randint(0, len(fg)-1)]
        sd = int(np.clip(c[0]-ps//2, 0, D-ps))
        sh = int(np.clip(c[1]-ps//2, 0, H-ps))
        sw = int(np.clip(c[2]-ps//2, 0, W-ps))
    else:
        sd = random.randint(0, max(0, D-ps))
        sh = random.randint(0, max(0, H-ps))
        sw = random.randint(0, max(0, W-ps))

    return (img[:, sd:sd+ps, sh:sh+ps, sw:sw+ps],
            seg[sd:sd+ps, sh:sh+ps, sw:sw+ps])


class BraTSH5Dataset(Dataset):
    """
    Load BraTS2020 H5 on-the-fly, không cần preprocess.py.

    Args:
        data_dir           : thư mục chứa *.h5 files
        split              : 'train' hoặc 'val'
        val_split          : tỷ lệ val (default 0.2)
        mode               : 'seg' → (img, seg) | 'ssl' → img only
        patch_size         : 96 phù hợp với volume 134x172x136
        patches_per_volume : số random patch mỗi volume
        in_channels        : 1 (đơn kênh) hoặc 4 (multi-modal)
        channel_idx        : kênh dùng khi in_channels=1 và H5 có >1 kênh
        seed               : seed cho train/val split
        ram_cache          : load tất cả vào RAM
    """

    IMG_KEYS = ['image', 'data', 'img', 'volume', 'X']
    SEG_KEYS = ['label', 'mask', 'seg', 'segmentation', 'y', 'Y']

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        val_split: float = 0.2,
        mode: str = 'seg',
        patch_size: int = 96,
        patches_per_volume: int = 8,
        in_channels: int = 1,
        channel_idx: int = 0,
        seed: int = 42,
        ram_cache: bool = False,
    ):
        self.data_dir           = Path(data_dir)
        self.mode               = mode
        self.patch_size         = patch_size
        self.patches_per_volume = patches_per_volume
        self.in_channels        = in_channels
        self.channel_idx        = channel_idx
        self.ram_cache          = ram_cache

        h5_files = sorted(
            list(self.data_dir.rglob('*.h5')) +
            list(self.data_dir.rglob('*.hdf5'))
        )
        if not h5_files:
            raise FileNotFoundError(
                f"Không tìm thấy file .h5 trong '{data_dir}'"
            )

        rng = random.Random(seed)
        files = list(h5_files); rng.shuffle(files)
        n_val = max(1, int(len(files) * val_split))
        self.files = files[:n_val] if split == 'val' else files[n_val:]

        self.samples: List[Tuple[Path, int]] = [
            (f, i) for f in self.files for i in range(patches_per_volume)
        ]

        self._cache = {}
        if ram_cache:
            print(f'[{split}] Caching {len(self.files)} volumes...')
            for f in self.files:
                self._cache[f] = self._load_h5(f)
            mb = sum(v[0].nbytes + (v[1].nbytes if v[1] is not None else 0)
                     for v in self._cache.values()) / 1e6
            print(f'  RAM: {mb/1e3:.2f} GB')

        sample_img, _ = self._load_h5(self.files[0])
        print(
            f'[BraTSH5Dataset:{split}:{mode}] '
            f'{len(self.files)} vols x {patches_per_volume} = {len(self.samples)} samples\n'
            f'  volume={sample_img.shape} patch={patch_size} in_ch={in_channels}'
        )

    def _find_key(self, hf, candidates):
        for k in candidates:
            if k in hf: return k
        keys = list(hf.keys())
        return keys[0] if keys else None

    def _load_h5(self, path: Path):
        import h5py
        with h5py.File(path, 'r') as f:
            img_key = self._find_key(f, self.IMG_KEYS)
            assert img_key, f"No image key in {path}"
            img = f[img_key][()].astype(np.float32)
            seg = None
            if self.mode == 'seg':
                seg_key = self._find_key(f, self.SEG_KEYS)
                if seg_key and seg_key in f:
                    seg = f[seg_key][()].astype(np.float32)

        # Normalize img → (C, D, H, W)
        if img.ndim == 3:
            img = img[np.newaxis]                       # (D,H,W) → (1,D,H,W)
        elif img.ndim == 4:
            ch_axes = [i for i, s in enumerate(img.shape) if s <= 16]
            if ch_axes:
                ch_ax = ch_axes[0]
                if ch_ax != 0:
                    img = np.moveaxis(img, ch_ax, 0)
            else:
                img = np.moveaxis(img, -1, 0)           # channel-last fallback
        else:
            raise ValueError(f"ndim={img.ndim} không hỗ trợ: {path}")

        # Chọn kênh
        C = img.shape[0]
        if self.in_channels == 1:
            ch = min(self.channel_idx, C-1)
            img = img[ch:ch+1]
        elif C < self.in_channels:
            img = np.concatenate([img]*((self.in_channels//C)+1), axis=0)[:self.in_channels]
        else:
            img = img[:self.in_channels]

        for c in range(img.shape[0]):
            img[c] = normalize_volume(img[c])

        if seg is not None:
            if seg.ndim == 4: seg = seg.squeeze()
            seg = (seg > 0).astype(np.float32)

        return img, seg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, patch_idx = self.samples[idx]

        if self.ram_cache and path in self._cache:
            img, seg = self._cache[path]
            img = img.copy()
            if seg is not None: seg = seg.copy()
        else:
            img, seg = self._load_h5(path)

        # Seed riêng cho mỗi (path, patch_idx) — đa dạng nhưng reproducible
        random.seed(hash(str(path)) ^ (patch_idx * 2654435761) & 0xFFFFFFFF)

        if self.mode == 'ssl':
            C, D, H, W = img.shape; ps = self.patch_size
            sd = random.randint(0, max(0, D-ps))
            sh = random.randint(0, max(0, H-ps))
            sw = random.randint(0, max(0, W-ps))
            return torch.from_numpy(img[:, sd:sd+ps, sh:sh+ps, sw:sw+ps].copy())

        assert seg is not None, f"Không tìm thấy label/mask key trong {path}"
        img_p, seg_p = extract_patch(img, seg, self.patch_size)
        return (torch.from_numpy(img_p.copy()),
                torch.from_numpy(seg_p.copy()).unsqueeze(0))


# Alias để không phải sửa import trong train_dino.py / train_seg.py
class PatchDataset(BraTSH5Dataset):
    def __init__(self, patch_dir, split='train', mode='seg',
                 ram_cache=False, **kwargs):
        super().__init__(data_dir=patch_dir, split=split,
                         mode=mode, ram_cache=ram_cache, **kwargs)