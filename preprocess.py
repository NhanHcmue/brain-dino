"""
preprocess.py — NII → NPY (float16) patches
Tiết kiệm disk tối đa: float16 + lưu riêng img/seg

Ước tính disk:
  369 bệnh nhân × 32 patches × 128³ × 4 modalities × 2 bytes (fp16)
  ≈ 369 × 32 × (128³) × 4 × 2 / 1e9 ≈ 20–25 GB

Upload lên Drive 1 lần, train nhiều lần.

Usage:
    python preprocess.py \
        --src data/MICCAI_BraTS2020_TrainingData \
        --dst data/brats_patches \
        --n_patches 32

Layout output:
    data/brats_patches/
        manifest.json          ← danh sách file train/val
        train/
            BraTS20_001_p00_img.npy   ← (4, 128, 128, 128) float16
            BraTS20_001_p00_seg.npy   ← (128, 128, 128) uint8
            ...
        val/
            ...
"""

import argparse, json, os, random
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

MODALITIES = ['t1', 't1ce', 't2', 'flair']


def normalize(v: np.ndarray) -> np.ndarray:
    """Z-score trên brain mask (background=0 sau skull-strip)."""
    brain = v > 0
    if brain.sum() > 100:
        mu  = v[brain].mean()
        std = v[brain].std() + 1e-8
        v   = (v - mu) / std
        v[~brain] = 0.0
    return v.astype(np.float32)


def load_nii(patient_dir: str, pid: str, suffix: str) -> np.ndarray | None:
    for ext in ['.nii', '.nii.gz']:
        p = os.path.join(patient_dir, f'{pid}_{suffix}{ext}')
        if os.path.exists(p):
            return nib.load(p).get_fdata(dtype=np.float32)
    return None


def foreground_patch(imgs, seg, patch=128, fg_prob=0.85):
    """
    Foreground-biased sampling: 85% center tại tumor voxel.
    imgs: (4, D, H, W), seg: (D, H, W)
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
        c  = fg[random.randint(0, len(fg)-1)]
        sd = int(np.clip(c[0] - patch//2, 0, D - patch))
        sh = int(np.clip(c[1] - patch//2, 0, H - patch))
        sw = int(np.clip(c[2] - patch//2, 0, W - patch))
    else:
        sd = random.randint(0, D - patch)
        sh = random.randint(0, H - patch)
        sw = random.randint(0, W - patch)

    img_p = imgs[:, sd:sd+patch, sh:sh+patch, sw:sw+patch]
    seg_p = seg[sd:sd+patch, sh:sh+patch, sw:sw+patch]
    return img_p, seg_p


def preprocess(args):
    src = Path(args.src)
    dst = Path(args.dst)
    (dst / 'train').mkdir(parents=True, exist_ok=True)
    (dst / 'val').mkdir(parents=True, exist_ok=True)

    pids = sorted(
        d for d in os.listdir(src)
        if (src / d).is_dir() and not d.startswith('.')
    )
    random.seed(args.seed)
    random.shuffle(pids)

    n_val   = max(1, int(len(pids) * args.val_split))
    val_set = set(pids[:n_val])
    manifest = {'train': [], 'val': []}

    total_bytes = 0
    skipped = 0

    for pid in tqdm(pids, desc='Processing', ncols=80):
        split   = 'val' if pid in val_set else 'train'
        out_dir = dst / split
        pdir    = str(src / pid)

        # Load 4 modalities
        vols = []
        for mod in MODALITIES:
            v = load_nii(pdir, pid, mod)
            if v is None:
                tqdm.write(f'  SKIP {pid}: missing {mod}')
                break
            vols.append(normalize(v))
        if len(vols) < 4:
            skipped += 1
            continue

        seg_vol = load_nii(pdir, pid, 'seg')
        if seg_vol is None:
            tqdm.write(f'  SKIP {pid}: missing seg')
            skipped += 1
            continue

        img_vol = np.stack(vols, axis=0)  # (4, D, H, W)
        seg_bin = (seg_vol > 0).astype(np.uint8)  # binary: tumor vs background

        saved = 0
        attempts = 0
        while saved < args.n_patches and attempts < args.n_patches * 15:
            attempts += 1
            img_p, seg_p = foreground_patch(img_vol, seg_bin, args.patch)

            # Bỏ patch nếu quá ít foreground brain
            if (img_p[1] != 0).mean() < 0.05:
                continue

            tag      = f'{pid}_p{saved:02d}'
            img_path = str(out_dir / f'{tag}_img.npy')
            seg_path = str(out_dir / f'{tag}_seg.npy')

            np.save(img_path, img_p.astype(np.float16))  # float16 → tiết kiệm 50%
            np.save(seg_path, seg_p)                      # uint8 → rất nhỏ

            manifest[split].append({'img': f'{split}/{tag}_img.npy',
                                    'seg': f'{split}/{tag}_seg.npy'})
            total_bytes += os.path.getsize(img_path) + os.path.getsize(seg_path)
            saved += 1

    # Lưu manifest
    with open(dst / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\n✓ Done!')
    print(f'  Train : {len(manifest["train"])} patches')
    print(f'  Val   : {len(manifest["val"])} patches')
    print(f'  Skipped: {skipped} volumes')
    print(f'  Disk  : {total_bytes/1e9:.1f} GB')
    print(f'  → {dst}/manifest.json')
    print(f'\nTips: upload thư mục {dst}/ lên Google Drive rồi mount khi train.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src',       required=True,   help='BraTS TrainingData dir')
    p.add_argument('--dst',       required=True,   help='Output patch dir')
    p.add_argument('--patch',     type=int, default=128)
    p.add_argument('--n_patches', type=int, default=32, help='Patches per volume')
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--seed',      type=int, default=42)

    args = p.parse_args()
    random.seed(args.seed)

    # Ước tính
    n_train = int(369 * (1 - args.val_split))
    est = n_train * args.n_patches * (args.patch ** 3) * 4 * 2 / 1e9  # 4 mod, fp16
    print(f'Patch size   : {args.patch}³')
    print(f'N patches    : {args.n_patches}/volume')
    print(f'Format       : float16 NPY (4 modalities)')
    print(f'Est. disk    : ~{est:.0f} GB')
    preprocess(args)


if __name__ == '__main__':
    main()