#!/bin/bash
# =============================================================
# run_all.sh — BraTS2020: tải data → DINO → Seg → Baseline
# Dùng cho máy local / GPU thuê kết nối qua VSCode
# =============================================================
#
# LẦN ĐẦU (chưa có data):
#   bash run_all.sh \
#       --data_dir /path/to/lưu/data \
#       --kaggle_user TÊN_KAGGLE \
#       --kaggle_key  API_KEY_KAGGLE
#
#   API key lấy tại: kaggle.com → Settings → API → Create New Token
#
# ĐÃ CÓ DATA (chạy lại train):
#   bash run_all.sh --data_dir /path/to/MICCAI_BraTS2020_TrainingData
#
# CHỈ TRAIN SEG (đã có DINO encoder):
#   bash run_all.sh --data_dir /path/to/... --skip_dino
#
# CHỈ TRAIN SEG PRETRAINED (không train baseline):
#   bash run_all.sh --data_dir /path/to/... --skip_dino --skip_baseline
#
# ĐIỀU CHỈNH GPU (RTX 3080/4080 16GB → defaults đã tối ưu):
#   bash run_all.sh --data_dir /path/to/... --batch_dino 4 --batch_seg 2
#
# =============================================================

set -e

# ── Defaults (tối ưu cho RTX 3080/4080 16GB) ─────────────────
DATA_DIR=""
KAGGLE_USER=""
KAGGLE_KEY=""

BATCH_DINO=4      # effective = 4 × grad_accum 2 = 8
BATCH_SEG=2       # effective = 2 × grad_accum 4 = 8
WORKERS=4
CACHE_SIZE=80     # ~80 volumes trong RAM (~4GB RAM)
N_PATCHES=32
PATCH_SIZE=128
EPOCHS_DINO=100
EPOCHS_SEG=200

SKIP_DINO=0
SKIP_BASELINE=0

# ── Parse args ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)      DATA_DIR="$2";     shift 2 ;;
        --kaggle_user)   KAGGLE_USER="$2";  shift 2 ;;
        --kaggle_key)    KAGGLE_KEY="$2";   shift 2 ;;
        --batch_dino)    BATCH_DINO="$2";   shift 2 ;;
        --batch_seg)     BATCH_SEG="$2";    shift 2 ;;
        --workers)       WORKERS="$2";      shift 2 ;;
        --cache_size)    CACHE_SIZE="$2";   shift 2 ;;
        --epochs_dino)   EPOCHS_DINO="$2";  shift 2 ;;
        --epochs_seg)    EPOCHS_SEG="$2";   shift 2 ;;
        --skip_dino)     SKIP_DINO=1;       shift 1 ;;
        --skip_baseline) SKIP_BASELINE=1;   shift 1 ;;
        *) echo "❌ Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Header ────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║  BraTS2020 — DINO + ConvNeXt-Tiny + nnU-Net  ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""
echo "  batch_dino  : $BATCH_DINO  (effective $(( BATCH_DINO * 2 )))"
echo "  batch_seg   : $BATCH_SEG   (effective $(( BATCH_SEG * 4 )))"
echo "  workers     : $WORKERS"
echo "  cache RAM   : $CACHE_SIZE volumes"
echo "  epochs DINO : $EPOCHS_DINO"
echo "  epochs Seg  : $EPOCHS_SEG"
echo ""

# ── Cài thư viện ─────────────────────────────────────────────
echo "=== Cài đặt thư viện ==="
pip install -r requirements.txt -q
echo "  ✓ Xong"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BƯỚC 0: Tải data (chỉ chạy nếu chưa có)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "=== Bước 0: Chuẩn bị data ==="

# Kiểm tra xem DATA_DIR đã có NII chưa
_data_ready() {
    [ -n "$1" ] && [ -d "$1" ] && \
    [ "$(ls -d "$1"/BraTS20_* 2>/dev/null | wc -l)" -gt 10 ]
}

if _data_ready "$DATA_DIR"; then
    N=$(ls -d "$DATA_DIR"/BraTS20_* | wc -l)
    echo "  ✓ Data đã có: $DATA_DIR ($N bệnh nhân)"

elif [ -n "$KAGGLE_USER" ] && [ -n "$KAGGLE_KEY" ]; then
    # ── Tải từ Kaggle ─────────────────────────────────────
    echo "  → Tải BraTS2020 từ Kaggle (~10GB)..."

    pip install kaggle -q

    mkdir -p ~/.kaggle
    echo "{\"username\":\"$KAGGLE_USER\",\"key\":\"$KAGGLE_KEY\"}" \
        > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json

    # Xác định thư mục đích
    DOWNLOAD_DIR="${DATA_DIR:-data}"
    mkdir -p "$DOWNLOAD_DIR"

    kaggle datasets download \
        -d awsaf49/brats20-dataset-training-validation \
        -p "$DOWNLOAD_DIR" \
        --unzip

    echo "  ✓ Tải và giải nén xong"

    # Tìm thư mục MICCAI_BraTS2020_TrainingData
    FOUND=$(find "$DOWNLOAD_DIR" -type d -name "MICCAI_BraTS2020_TrainingData" \
            2>/dev/null | head -n 1)
    if [ -z "$FOUND" ]; then
        echo "  ❌ Không tìm thấy MICCAI_BraTS2020_TrainingData sau khi giải nén"
        echo "     Kiểm tra nội dung thư mục: ls $DOWNLOAD_DIR"
        exit 1
    fi
    DATA_DIR="$FOUND"
    echo "  ✓ DATA_DIR = $DATA_DIR"

else
    # ── Hướng dẫn nếu thiếu thông tin ──────────────────────
    echo ""
    echo "  ❌ Chưa có data và chưa cung cấp Kaggle credentials."
    echo ""
    echo "  Cách 1 — Cung cấp Kaggle API key (tự động tải):"
    echo "    bash run_all.sh \\"
    echo "        --data_dir ./data \\"
    echo "        --kaggle_user TÊN_CỦA_BẠN \\"
    echo "        --kaggle_key  KEY_TỪ_KAGGLE"
    echo ""
    echo "  Cách 2 — Tải thủ công từ Kaggle rồi giải nén:"
    echo "    https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation"
    echo "    Sau đó: bash run_all.sh --data_dir /path/to/MICCAI_BraTS2020_TrainingData"
    echo ""
    exit 1
fi

# Xác nhận lần cuối
N_PATIENTS=$(ls -d "$DATA_DIR"/BraTS20_* 2>/dev/null | wc -l)
if [ "$N_PATIENTS" -lt 10 ]; then
    echo "  ❌ DATA_DIR có vẻ rỗng hoặc sai đường dẫn: $DATA_DIR"
    exit 1
fi
echo "  ✓ $N_PATIENTS bệnh nhân sẵn sàng"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ghi config files (tự động từ tham số + DATA_DIR)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "=== Ghi config files ==="
mkdir -p configs outputs/dino outputs/seg_pretrained outputs/seg_baseline

# grad_accum tự động để giữ effective batch = 8
ACCUM_DINO=$(python3 -c "print(max(1, 8 // $BATCH_DINO))")
ACCUM_SEG=$(python3  -c "print(max(1, 8 // $BATCH_SEG))")

# n_local_crops: giảm nếu batch nhỏ để tránh OOM
N_LOCAL=$(python3 -c "print(4 if $BATCH_DINO >= 4 else 2)")

cat > configs/dino.yaml << YAML
data:
  data_dir:    "$DATA_DIR"
  patch_dir:   null
  n_patches:   $N_PATCHES
  patch_size:  $PATCH_SIZE
  cache_size:  $CACHE_SIZE

dino:
  out_dim:                    4096
  momentum:                   0.996
  global_scale:               [0.8, 1.0]
  local_scale:                [0.4, 0.6]
  n_local_crops:              $N_LOCAL
  teacher_temp:               0.04
  warmup_teacher_temp:        0.04
  warmup_teacher_temp_epochs: 30
  student_temp:               0.1
  center_momentum:            0.9

train:
  num_epochs:    $EPOCHS_DINO
  batch_size:    $BATCH_DINO
  num_workers:   $WORKERS
  seed:          42
  grad_accum:    $ACCUM_DINO
  warmup_epochs: 10

optimizer:
  base_lr:      5.0e-4
  min_lr:       1.0e-6
  weight_decay: 0.04

output:
  dir:          outputs/dino
  checkpoint:   outputs/dino/checkpoint.pth
  best_encoder: outputs/dino/best_encoder.pth
YAML

cat > configs/seg_pretrained.yaml << YAML
data:
  data_dir:      "$DATA_DIR"
  patch_dir:     null
  n_patches:     $N_PATCHES
  n_patches_val: 16
  patch_size:    $PATCH_SIZE
  cache_size:    $CACHE_SIZE

pretrain:
  encoder_path: outputs/dino/best_encoder.pth

train:
  num_epochs:  $EPOCHS_SEG
  batch_size:  $BATCH_SEG
  num_workers: $WORKERS
  seed:        42
  grad_accum:  $ACCUM_SEG

optimizer:
  lr:           2.0e-4
  weight_decay: 1.0e-5

augmentation:
  p_flip:      0.5
  p_intensity: 0.3
  p_noise:     0.2
  noise_std:   0.05

finetune:
  freeze_encoder_epochs: 20
  encoder_lr_scale:      0.01

output:
  dir:        outputs/seg_pretrained
  checkpoint: outputs/seg_pretrained/checkpoint.pth
  best_model: outputs/seg_pretrained/best_model.pth
YAML

cat > configs/seg_baseline.yaml << YAML
data:
  data_dir:      "$DATA_DIR"
  patch_dir:     null
  n_patches:     $N_PATCHES
  n_patches_val: 16
  patch_size:    $PATCH_SIZE
  cache_size:    $CACHE_SIZE

pretrain:
  encoder_path: null

train:
  num_epochs:  $EPOCHS_SEG
  batch_size:  $BATCH_SEG
  num_workers: $WORKERS
  seed:        42
  grad_accum:  $ACCUM_SEG

optimizer:
  lr:           2.0e-4
  weight_decay: 1.0e-5

augmentation:
  p_flip:      0.5
  p_intensity: 0.3
  p_noise:     0.2
  noise_std:   0.05

finetune:
  freeze_encoder_epochs: 0
  encoder_lr_scale:      1.0

output:
  dir:        outputs/seg_baseline
  checkpoint: outputs/seg_baseline/checkpoint.pth
  best_model: outputs/seg_baseline/best_model.pth
YAML

echo "  ✓ configs/dino.yaml"
echo "  ✓ configs/seg_pretrained.yaml"
echo "  ✓ configs/seg_baseline.yaml"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BƯỚC 1: DINO Pretraining
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ $SKIP_DINO -eq 0 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Bước 1 / 3 — DINO Self-Supervised Pretraining"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  epochs     : $EPOCHS_DINO"
    echo "  batch      : $BATCH_DINO  × grad_accum $ACCUM_DINO = effective $(( BATCH_DINO * ACCUM_DINO ))"
    echo "  n_local    : $N_LOCAL crops"
    echo ""
    python train_dino.py --config configs/dino.yaml
    echo ""
    echo "  ✓ Teacher encoder → outputs/dino/best_encoder.pth"
    echo ""
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Bước 1 / 3 — DINO SKIPPED (--skip_dino)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ ! -f "outputs/dino/best_encoder.pth" ]; then
        echo "  ❌ outputs/dino/best_encoder.pth không tồn tại!"
        echo "     Chạy lại không có --skip_dino để train DINO trước."
        exit 1
    fi
    echo "  ✓ Dùng encoder: outputs/dino/best_encoder.pth"
    echo ""
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BƯỚC 2: Segmentation — DINO Pretrained
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Bước 2 / 3 — Segmentation (DINO Pretrained)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 1 (ep 0-19) : encoder frozen"
echo "  Phase 2 (ep 20+)  : encoder unfreeze, lr ×0.01"
echo ""
python train_seg.py --config configs/seg_pretrained.yaml
echo ""
echo "  ✓ Best model → outputs/seg_pretrained/best_model.pth"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BƯỚC 3: Segmentation — Baseline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ $SKIP_BASELINE -eq 0 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Bước 3 / 3 — Segmentation (Baseline)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  (Random init — để so sánh với DINO pretrained)"
    echo ""
    python train_seg.py --config configs/seg_baseline.yaml
    echo ""
    echo "  ✓ Best model → outputs/seg_baseline/best_model.pth"
    echo ""
else
    echo "  Bước 3 / 3 — Baseline SKIPPED (--skip_baseline)"
    echo ""
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tổng kết
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "╔═══════════════════════════════════════════════╗"
echo "║           ✓ Pipeline hoàn thành!              ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""
[ -f "outputs/dino/best_encoder.pth" ]         && \
    echo "  DINO encoder    : outputs/dino/best_encoder.pth"
[ -f "outputs/seg_pretrained/best_model.pth" ] && \
    echo "  DINO-Pretrained : outputs/seg_pretrained/best_model.pth"
[ -f "outputs/seg_baseline/best_model.pth" ]   && \
    echo "  Baseline        : outputs/seg_baseline/best_model.pth"
echo ""
