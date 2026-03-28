#!/bin/bash
# run_all.sh — Chạy toàn bộ pipeline: preprocess → DINO → Seg → Baseline
#
# Usage:
#   bash run_all.sh --data_dir /path/to/MICCAI_BraTS2020_TrainingData
#
# Skip steps:
#   bash run_all.sh --data_dir /path/to/... --skip_preprocess
#   bash run_all.sh --skip_preprocess --skip_dino   # chỉ train seg
#
# Override GPU settings:
#   bash run_all.sh --data_dir /path/to/... --batch_dino 8 --batch_seg 4 --workers 8

set -e

DATA_DIR=""; PATCH_DIR="data/brats_patches"
BATCH_DINO=4; BATCH_SEG=2; WORKERS=8
N_PATCHES=32
SKIP_PREPROCESS=0; SKIP_DINO=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)       DATA_DIR="$2";       shift 2 ;;
        --patch_dir)      PATCH_DIR="$2";      shift 2 ;;
        --batch_dino)     BATCH_DINO="$2";     shift 2 ;;
        --batch_seg)      BATCH_SEG="$2";      shift 2 ;;
        --workers)        WORKERS="$2";        shift 2 ;;
        --n_patches)      N_PATCHES="$2";      shift 2 ;;
        --skip_preprocess) SKIP_PREPROCESS=1;  shift 1 ;;
        --skip_dino)      SKIP_DINO=1;         shift 1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  BraTS DINO + nnU-Net Pipeline"
echo "========================================"
pip install -r requirements.txt -q

# ── Step 1: Preprocess NII → NPY ──
if [ $SKIP_PREPROCESS -eq 0 ]; then
    if [ -z "$DATA_DIR" ]; then
        echo "ERROR: --data_dir required (hoặc dùng --skip_preprocess)"
        exit 1
    fi
    echo -e "\n=== Step 1: Preprocess NII → NPY (float16) ==="
    echo "  src: $DATA_DIR"
    echo "  dst: $PATCH_DIR"
    echo "  n_patches: $N_PATCHES/volume"
    python preprocess.py \
        --src "$DATA_DIR" \
        --dst "$PATCH_DIR" \
        --n_patches $N_PATCHES
    echo "  ✓ Done → upload $PATCH_DIR/ lên Drive nếu cần"
else
    echo -e "\n=== Step 1: SKIPPED ==="
    [ ! -f "$PATCH_DIR/manifest.json" ] && echo "ERROR: manifest.json not found!" && exit 1
    echo "  ✓ Found: $PATCH_DIR/manifest.json"
fi

# ── Step 2: DINO Pretraining ──
if [ $SKIP_DINO -eq 0 ]; then
    echo -e "\n=== Step 2: DINO Pretraining ==="
    python train_dino.py \
        --config configs/dino.yaml \
        --batch  $BATCH_DINO \
        --workers $WORKERS
    echo "  ✓ Teacher encoder → outputs/dino/best_encoder.pth"
else
    echo -e "\n=== Step 2: DINO — SKIPPED ==="
    [ ! -f "outputs/dino/best_encoder.pth" ] && \
        echo "WARNING: best_encoder.pth not found!" || \
        echo "  ✓ Found: outputs/dino/best_encoder.pth"
fi

# ── Step 3a: Segmentation Pretrained ──
echo -e "\n=== Step 3a: Segmentation — DINO Pretrained ==="
python train_seg.py \
    --config  configs/seg_pretrained.yaml \
    --batch   $BATCH_SEG \
    --workers $WORKERS
echo "  ✓ Best model → outputs/seg_pretrained/best_model.pth"

# ── Step 3b: Segmentation Baseline ──
echo -e "\n=== Step 3b: Segmentation — Baseline (random init) ==="
python train_seg.py \
    --config  configs/seg_baseline.yaml \
    --batch   $BATCH_SEG \
    --workers $WORKERS
echo "  ✓ Best model → outputs/seg_baseline/best_model.pth"

echo -e "\n✓ Pipeline hoàn thành!"
echo "  DINO encoder    : outputs/dino/best_encoder.pth"
echo "  DINO-Pretrained : outputs/seg_pretrained/best_model.pth"
echo "  Baseline        : outputs/seg_baseline/best_model.pth"