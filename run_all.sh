#!/bin/bash
# run_all.sh — Pipeline BraTS2020 H5 (không cần preprocess)
#
# Thay đổi so với bản cũ:
#   - XÓA Step preprocess NII→NPY (không cần nữa, đọc H5 trực tiếp)
#   - Thêm --gdrive_id để tải dataset H5 từ Google Drive
#   - Thêm --in_channels (default=1 cho 1-kênh flair)
#   - Thêm --channel_idx để chọn modality
#
# Usage examples:
#   # Tải data từ Drive rồi train toàn bộ:
#   bash run_all.sh --gdrive_id 1uO6WezH0-qrFH-m0uxNmwlUujOaCCaGs
#
#   # Data đã có sẵn:
#   bash run_all.sh --data_dir data/brats_h5 --batch_dino 4 --batch_seg 2
#
#   # Bỏ qua DINO, chỉ train seg:
#   bash run_all.sh --data_dir data/brats_h5 --skip_dino
#
#   # 4 kênh (tất cả modalities):
#   bash run_all.sh --data_dir data/brats_h5 --in_channels 4

set -e

# ── Defaults ──────────────────────────────────────────────
DATA_DIR=""
GDRIVE_ID=""
BATCH_DINO=4
BATCH_SEG=2
WORKERS=4
IN_CHANNELS=1       # 1 kênh (flair) theo yêu cầu
CHANNEL_IDX=3       # 3=flair, 0=t1, 1=t1ce, 2=t2
SKIP_DINO=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)    DATA_DIR="$2";    shift 2 ;;
        --gdrive_id)   GDRIVE_ID="$2";   shift 2 ;;
        --batch_dino)  BATCH_DINO="$2";  shift 2 ;;
        --batch_seg)   BATCH_SEG="$2";   shift 2 ;;
        --workers)     WORKERS="$2";     shift 2 ;;
        --in_channels) IN_CHANNELS="$2"; shift 2 ;;
        --channel_idx) CHANNEL_IDX="$2"; shift 2 ;;
        --skip_dino)   SKIP_DINO=1;      shift 1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  BraTS H5 — DINO + nnU-Net Pipeline"
echo "  in_channels : $IN_CHANNELS"
echo "  channel_idx : $CHANNEL_IDX (0=t1,1=t1ce,2=t2,3=flair)"
echo "========================================"

# ── Cài packages ──────────────────────────────────────────
pip install torch torchvision nibabel numpy tqdm matplotlib pyyaml scipy h5py -q

# ── Step 0: Download từ Google Drive ──────────────────────
if [ -n "$GDRIVE_ID" ]; then
    echo -e "\n=== Step 0: Download dataset từ Google Drive ==="
    pip install gdown -q
    mkdir -p data
    ZIP_PATH="data/brats_h5.zip"

    echo "  Downloading file ID: $GDRIVE_ID ..."
    gdown "https://drive.google.com/uc?id=$GDRIVE_ID" -O "$ZIP_PATH" \
        || gdown --fuzzy "https://drive.google.com/file/d/$GDRIVE_ID/view" -O "$ZIP_PATH"

    echo "  Extracting..."
    unzip -q "$ZIP_PATH" -d data/
    rm -f "$ZIP_PATH"

    # Auto-detect thư mục chứa .h5 files
    if [ -z "$DATA_DIR" ]; then
        DATA_DIR=$(find data -name "*.h5" -printf "%h\n" | sort -u | head -n 1)
        if [ -z "$DATA_DIR" ]; then
            # Thử tìm thư mục con phổ biến
            DATA_DIR=$(find data -type d -name "*BraTS*" | head -n 1)
        fi
        if [ -z "$DATA_DIR" ]; then
            DATA_DIR="data"
        fi
        echo "  Auto DATA_DIR = $DATA_DIR"
        echo "  H5 files found: $(find $DATA_DIR -name '*.h5' | wc -l)"
    fi
fi

# Kiểm tra DATA_DIR
if [ -z "$DATA_DIR" ]; then
    echo "ERROR: Cần truyền --data_dir hoặc --gdrive_id"
    echo "  Ví dụ: bash run_all.sh --gdrive_id 1uO6WezH0-qrFH-m0uxNmwlUujOaCCaGs"
    exit 1
fi

H5_COUNT=$(find "$DATA_DIR" -name "*.h5" 2>/dev/null | wc -l)
if [ "$H5_COUNT" -eq 0 ]; then
    echo "ERROR: Không tìm thấy file .h5 trong $DATA_DIR"
    exit 1
fi
echo -e "\n✓ Data dir   : $DATA_DIR"
echo   "✓ H5 files   : $H5_COUNT"

# ── Step 1: DINO Pretraining ───────────────────────────────
if [ $SKIP_DINO -eq 0 ]; then
    echo -e "\n=== Step 1: DINO Pretraining ==="
    python train_dino.py \
        --config     configs/dino.yaml \
        --batch      $BATCH_DINO \
        --workers    $WORKERS \
        --patch_dir  "$DATA_DIR" \
        --in_channels $IN_CHANNELS \
        --channel_idx $CHANNEL_IDX
    echo "  ✓ Teacher encoder → outputs/dino/best_encoder.pth"
else
    echo -e "\n=== Step 1: DINO — SKIPPED ==="
    [ -f "outputs/dino/best_encoder.pth" ] && \
        echo "  ✓ Found: outputs/dino/best_encoder.pth" || \
        echo "  WARNING: outputs/dino/best_encoder.pth không tồn tại"
fi

# ── Step 2a: Segmentation DINO Pretrained ─────────────────
echo -e "\n=== Step 2a: Segmentation — DINO Pretrained ==="
python train_seg.py \
    --config      configs/seg_pretrained.yaml \
    --batch       $BATCH_SEG \
    --workers     $WORKERS \
    --patch_dir   "$DATA_DIR" \
    --in_channels $IN_CHANNELS \
    --channel_idx $CHANNEL_IDX
echo "  ✓ Best model → outputs/seg_pretrained/best_model.pth"

# ── Step 2b: Segmentation Baseline ────────────────────────
echo -e "\n=== Step 2b: Segmentation — Baseline ==="
python train_seg.py \
    --config      configs/seg_baseline.yaml \
    --batch       $BATCH_SEG \
    --workers     $WORKERS \
    --patch_dir   "$DATA_DIR" \
    --in_channels $IN_CHANNELS \
    --channel_idx $CHANNEL_IDX
echo "  ✓ Best model → outputs/seg_baseline/best_model.pth"

echo -e "\n✓ Pipeline hoàn thành!"
echo "  DINO encoder    : outputs/dino/best_encoder.pth"
echo "  DINO-Pretrained : outputs/seg_pretrained/best_model.pth"
echo "  Baseline        : outputs/seg_baseline/best_model.pth"