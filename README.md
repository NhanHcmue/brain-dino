# BraTS2020 — DINO + ConvNeXt-Tiny + nnU-Net Decoder

Self-supervised pretraining (DINO) + fine-tuning (nnU-Net decoder) trên BraTS2020.
Chỉ 2 mô hình để so sánh: **DINO-Pretrained** vs **Baseline (random init)**.

## Cấu trúc project

```
├── src/
│   ├── encoder.py       # ConvNeXt-Tiny 3D (backbone)
│   ├── models.py        # ConvNeXtNNUNet (encoder + nnU-Net decoder)
│   ├── dino.py          # DINO, DINOHead, DINOLoss
│   ├── losses.py        # Tversky + Focal + Deep Supervision
│   ├── dataset.py       # PatchDataset (load .npy)
│   ├── augmentation.py  # GPUAugmentation3D, DINOMultiCrop3D
│   └── __init__.py
├── configs/
│   ├── dino.yaml           # DINO pretraining
│   ├── seg_pretrained.yaml # Segmentation với DINO encoder
│   └── seg_baseline.yaml   # Segmentation baseline
├── preprocess.py        # NII → NPY (float16), ~20-25GB
├── train_dino.py        # Step 1: DINO pretraining
├── train_seg.py         # Step 2/3: Segmentation training
├── run_all.sh           # Pipeline đầy đủ
└── requirements.txt
```

## Tại sao thiết kế này cho Dice cao?

| Component | Lý do |
|---|---|
| **4 modalities** (t1,t1ce,t2,flair) | +0.05-0.08 Dice so với chỉ t1ce |
| **ConvNeXt-Tiny 3D** | Depthwise 7×7×7, LayerNorm, GELU → tốt hơn ResNet50 |
| **DINO** | Không cần large batch; representations tốt cho dense prediction |
| **nnU-Net decoder** | Không ASPP; adaptive channels; deep supervision |
| **Tversky+Focal loss** | Chống class imbalance 99% background |
| **2-phase fine-tuning** | Freeze encoder 20 epoch → decoder ổn định → unfreeze |

## Cài đặt

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

## Workflow

### Bước 0 — Preprocess (1 lần duy nhất)

```bash
python preprocess.py \
    --src data/MICCAI_BraTS2020_TrainingData \
    --dst data/brats_patches \
    --n_patches 32

# Ước tính: ~20-25 GB (float16, 4 modalities)
# Sau đó upload data/brats_patches/ lên Google Drive
```

### Bước 1 — DINO Pretraining

```bash
# Chỉnh batch_size trong configs/dino.yaml theo GPU của bạn:
# T4=4, L4/3090=8, A100=16

python train_dino.py --config configs/dino.yaml

# Output: outputs/dino/best_encoder.pth (teacher encoder)
```

### Bước 2 — Segmentation DINO-Pretrained

```bash
python train_seg.py --config configs/seg_pretrained.yaml
# Output: outputs/seg_pretrained/best_model.pth
```

### Bước 3 — Segmentation Baseline

```bash
python train_seg.py --config configs/seg_baseline.yaml
# Output: outputs/seg_baseline/best_model.pth
```

### Hoặc chạy toàn bộ pipeline

```bash
bash run_all.sh \
    --data_dir /path/to/MICCAI_BraTS2020_TrainingData \
    --batch_dino 8 \
    --batch_seg 4 \
    --workers 8
```

## Chạy trên cloud

### Google Colab / Kaggle

```python
# Mount Drive và chỉnh đường dẫn:
!git clone https://github.com/YOUR/REPO.git
%cd REPO
!pip install -r requirements.txt -q

# Nếu data đã preprocess và upload lên Drive:
!python train_dino.py \
    --config configs/dino.yaml \
    --batch 4 \
    --workers 2

!python train_seg.py \
    --config configs/seg_pretrained.yaml \
    --batch 2 \
    --workers 2
```

### Vast.ai / RunPod

```bash
git clone https://github.com/YOUR/REPO.git
cd REPO
pip install -r requirements.txt

# Mount data vào /data/, sau đó:
bash run_all.sh \
    --data_dir /data/MICCAI_BraTS2020_TrainingData \
    --batch_dino 8 \
    --batch_seg 4 \
    --workers 8
```

## Điều chỉnh theo VRAM

| GPU | VRAM | batch_dino | batch_seg | grad_accum | n_local_crops |
|---|---|---|---|---|---|
| T4 | 16GB | 4 | 2 | 4 | 2 |
| L4 / RTX3090 | 24GB | 8 | 4 | 2 | 4 |
| RTX4090 | 24GB | 8 | 4 | 2 | 4 |
| A100-40G | 40GB | 16 | 8 | 1 | 6 |

**Nếu OOM:**
1. Giảm `batch_size` và tăng `grad_accum` (giữ effective batch)
2. Giảm `n_local_crops` xuống 2
3. Giảm patch size từ 128 → 96 trong `preprocess.py`

## Kết quả kỳ vọng

| Model | Val Dice |
|---|---|
| Baseline (random ConvNeXt) | ~0.75–0.82 |
| DINO-Pretrained | **~0.88–0.93** |

## Lý do format NPY thay vì PT

- `.pt` float32: ~100GB (4 mod × 369 vol × 32 patches)
- `.pt` float16: ~50GB
- **`.npy` float16**: **~20-25GB** ← dùng trong project này
- `.npy` float16 + compress: ~8GB nhưng load chậm

NPY float16 được load bằng `np.load()` và cast sang float32 ngay trong `__getitem__` → không overhead so với PT.