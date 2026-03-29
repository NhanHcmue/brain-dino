#!/bin/bash
# =============================================================
# run_all.sh — BraTS2020 DINO + nnU-Net Pipeline (Lazy Loader)
# Không cần preprocess NII → NPY, không tốn 25GB disk thêm
# =============================================================
#
# CÁCH DÙNG:
#
#   [Kaggle Notebook] — data đã có sẵn, chỉ chạy:
#     bash run_all.sh --env kaggle
#
#   [Colab / Vast.ai / Local] — tự tải data qua Kaggle API:
#     bash run_all.sh --env colab \
#         --kaggle_username YOUR_USERNAME \
#         --kaggle_key      YOUR_API_KEY
#
#   [Đã có data sẵn ở local]:
#     bash run_all.sh --env local \
#         --data_dir /path/to/MICCAI_BraTS2020_TrainingData
#
# SKIP BƯỚC:
#   bash run_all.sh --env kaggle --skip_dino        # bỏ qua DINO, chỉ train seg
#   bash run_all.sh --env kaggle --skip_dino --skip_baseline  # chỉ seg pretrained
#
# ĐIỀU CHỈNH GPU:
#   bash run_all.sh --env kaggle --batch_dino 8 --batch_seg 4
#
#   GPU VRAM  | batch_dino | batch_seg | cache_size
#   T4  16GB  |     4      |     2     |    150
#   L4  24GB  |     8      |     4     |    250
#   A100 40GB |    16      |     8     |    500
# =============================================================

set -e

# ── Defaults ──────────────────────────────────────────────────
ENV="kaggle"           # kaggle | colab | local
DATA_DIR=""
KAGGLE_USERNAME=""
KAGGLE_KEY=""

BATCH_DINO=4
BATCH_SEG=2
WORKERS=4
CACHE_SIZE=150         # số volumes cache trong RAM
N_PATCHES=32           # patches ảo mỗi volume (không lưu disk)

SKIP_DINO=0
SKIP_BASELINE=0
EPOCHS_DINO=""
EPOCHS_SEG=""

# ── Parse args ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)              ENV="$2";              shift 2 ;;
        --data_dir)         DATA_DIR="$2";         shift 2 ;;
        --kaggle_username)  KAGGLE_USERNAME="$2";  shift 2 ;;
        --kaggle_key)       KAGGLE_KEY="$2";       shift 2 ;;
        --batch_dino)       BATCH_DINO="$2";       shift 2 ;;
        --batch_seg)        BATCH_SEG="$2";        shift 2 ;;
        --workers)          WORKERS="$2";          shift 2 ;;
        --cache_size)       CACHE_SIZE="$2";       shift 2 ;;
        --n_patches)        N_PATCHES="$2";        shift 2 ;;
        --epochs_dino)      EPOCHS_DINO="$2";      shift 2 ;;
        --epochs_seg)       EPOCHS_SEG="$2";       shift 2 ;;
        --skip_dino)        SKIP_DINO=1;           shift 1 ;;
        --skip_baseline)    SKIP_BASELINE=1;       shift 1 ;;
        *) echo "❌ Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Header ────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   BraTS2020 — DINO + ConvNeXt + nnU-Net     ║"
echo "║   Lazy NII Loader (không cần preprocess)     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  ENV         : $ENV"
echo "  batch_dino  : $BATCH_DINO"
echo "  batch_seg   : $BATCH_SEG"
echo "  workers     : $WORKERS"
echo "  cache_size  : $CACHE_SIZE volumes"
echo "  n_patches   : $N_PATCHES ảo/volume"
echo ""

# ── Cài thư viện ─────────────────────────────────────────────
echo "=== Cài đặt thư viện ==="
pip install -r requirements.txt -q
echo "  ✓ Xong"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BƯỚC 0: Lấy data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "=== Bước 0: Chuẩn bị data ==="

if [ "$ENV" = "kaggle" ]; then
    # ── Kaggle Notebook: data đã mount sẵn tại /kaggle/input ──
    KAGGLE_INPUT="/kaggle/input/brats20-dataset-training-validation"
    BRATS_SUBDIR="BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

    if [ -d "$KAGGLE_INPUT/$BRATS_SUBDIR" ]; then
        DATA_DIR="$KAGGLE_INPUT/$BRATS_SUBDIR"
        echo "  ✓ Kaggle dataset tìm thấy: $DATA_DIR"
    else
        # Thử tìm tự động trong /kaggle/input
        FOUND=$(find /kaggle/input -type d -name "MICCAI_BraTS2020_TrainingData" 2>/dev/null | head -n 1)
        if [ -n "$FOUND" ]; then
            DATA_DIR="$FOUND"
            echo "  ✓ Auto-detect: $DATA_DIR"
        else
            echo "  ❌ Không tìm thấy BraTS2020 data!"
            echo ""
            echo "  → Vào Kaggle Notebook:"
            echo "     Settings → Add Data → tìm 'BraTS 2020 Training Validation' by awsaf49"
            echo "     Sau đó chạy lại script này."
            exit 1
        fi
    fi

elif [ "$ENV" = "colab" ]; then
    # ── Colab / Vast.ai: tải qua Kaggle API ──
    if [ -z "$DATA_DIR" ]; then
        echo "  → Tải BraTS2020 qua Kaggle API..."

        # Cài kaggle CLI
        pip install kaggle -q

        # Thiết lập credentials
        if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
            mkdir -p ~/.kaggle
            echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
            chmod 600 ~/.kaggle/kaggle.json
            echo "  ✓ Kaggle credentials đã thiết lập"
        elif [ ! -f ~/.kaggle/kaggle.json ]; then
            echo "  ❌ Cần truyền --kaggle_username và --kaggle_key"
            echo ""
            echo "  Lấy API key tại: https://www.kaggle.com/settings → API → Create New Token"
            echo ""
            echo "  Ví dụ:"
            echo "    bash run_all.sh --env colab \\"
            echo "        --kaggle_username your_name \\"
            echo "        --kaggle_key      abc123xyz"
            exit 1
        fi

        # Tải dataset (~10GB, mất ~5 phút)
        mkdir -p data
        echo "  → Đang tải (~10GB, ~5 phút)..."
        kaggle datasets download \
            -d awsaf49/brats20-dataset-training-validation \
            -p data/ \
            --unzip

        # Tìm thư mục data
        DATA_DIR=$(find data -type d -name "MICCAI_BraTS2020_TrainingData" | head -n 1)
        if [ -z "$DATA_DIR" ]; then
            echo "  ❌ Extract xong nhưng không tìm thấy MICCAI_BraTS2020_TrainingData"
            exit 1
        fi
        echo "  ✓ Data sẵn sàng: $DATA_DIR"
    else
        echo "  ✓ Dùng data_dir đã cho: $DATA_DIR"
    fi

elif [ "$ENV" = "local" ]; then
    # ── Local: truyền --data_dir trực tiếp ──
    if [ -z "$DATA_DIR" ]; then
        echo "  ❌ --env local cần --data_dir /path/to/MICCAI_BraTS2020_TrainingData"
        exit 1
    fi
    echo "  ✓ Local data: $DATA_DIR"

else
    echo "  ❌ --env phải là: kaggle | colab | local"
    exit 1
fi

# Kiểm tra lần cuối
if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
    echo "  ❌ DATA_DIR không hợp lệ hoặc không tồn tại: $DATA_DIR"
    exit 1
fi

N_PATIENTS=$(ls -d "$DATA_DIR"/BraTS20_* 2>/dev/null | wc -l)
echo "  ✓ Tìm thấy $N_PATIENTS bệnh nhân"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ghi configs tự động từ DATA_DIR và tham số truyền vào
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "=== Ghi config files ==="
mkdir -p configs outputs/dino outputs/seg_pretrained outputs/seg_baseline

# Tính số epochs nếu không truyền
[ -z "$EPOCHS_DINO" ] && EPOCHS_DINO=100
[ -z "$EPOCHS_SEG"  ] && EPOCHS_SEG=200

# grad_accum: giữ effective batch = 8
GRAD_ACCUM_DINO=$(( 8 / BATCH_DINO ))
GRAD_ACCUM_SEG=$(( 8 / BATCH_SEG ))
[ $GRAD_ACCUM_DINO -lt 1 ] && GRAD_ACCUM_DINO=1
[ $GRAD_ACCUM_SEG  -lt 1 ] && GRAD_ACCUM_SEG=1

# n_local_crops theo batch_dino
N_LOCAL=4
[ $BATCH_DINO -le 4 ] && N_LOCAL=2

# ── configs/dino.yaml ──
cat > configs/dino.yaml << YAML
data:
  use_lazy:    true
  data_dir:    "$DATA_DIR"
  patch_dir:   null
  n_patches:   $N_PATCHES
  patch_size:  128
  cache_size:  $CACHE_SIZE
  ram_cache:   false

dino:
  out_dim:                     4096
  momentum:                    0.996
  global_scale:                [0.8, 1.0]
  local_scale:                 [0.4, 0.6]
  n_local_crops:               $N_LOCAL
  teacher_temp:                0.04
  warmup_teacher_temp:         0.04
  warmup_teacher_temp_epochs:  30
  student_temp:                0.1
  center_momentum:             0.9

train:
  num_epochs:   $EPOCHS_DINO
  batch_size:   $BATCH_DINO
  num_workers:  $WORKERS
  seed:         42
  grad_accum:   $GRAD_ACCUM_DINO
  warmup_epochs: 10

optimizer:
  base_lr:       5.0e-4
  min_lr:        1.0e-6
  weight_decay:  0.04

output:
  dir:          outputs/dino
  checkpoint:   outputs/dino/checkpoint.pth
  best_encoder: outputs/dino/best_encoder.pth
YAML

# ── configs/seg_pretrained.yaml ──
cat > configs/seg_pretrained.yaml << YAML
data:
  use_lazy:       true
  data_dir:       "$DATA_DIR"
  patch_dir:      null
  n_patches:      $N_PATCHES
  n_patches_val:  16
  patch_size:     128
  cache_size:     $CACHE_SIZE
  ram_cache:      false

pretrain:
  encoder_path: outputs/dino/best_encoder.pth

train:
  num_epochs:   $EPOCHS_SEG
  batch_size:   $BATCH_SEG
  num_workers:  $WORKERS
  seed:         42
  grad_accum:   $GRAD_ACCUM_SEG

optimizer:
  lr:            2.0e-4
  weight_decay:  1.0e-5

augmentation:
  p_flip:       0.5
  p_intensity:  0.3
  p_noise:      0.2
  noise_std:    0.05

finetune:
  freeze_encoder_epochs: 20
  encoder_lr_scale:      0.01

output:
  dir:         outputs/seg_pretrained
  checkpoint:  outputs/seg_pretrained/checkpoint.pth
  best_model:  outputs/seg_pretrained/best_model.pth
YAML

# ── configs/seg_baseline.yaml ──
cat > configs/seg_baseline.yaml << YAML
data:
  use_lazy:       true
  data_dir:       "$DATA_DIR"
  patch_dir:      null
  n_patches:      $N_PATCHES
  n_patches_val:  16
  patch_size:     128
  cache_size:     $CACHE_SIZE
  ram_cache:      false

pretrain:
  encoder_path: null

train:
  num_epochs:   $EPOCHS_SEG
  batch_size:   $BATCH_SEG
  num_workers:  $WORKERS
  seed:         42
  grad_accum:   $GRAD_ACCUM_SEG

optimizer:
  lr:            2.0e-4
  weight_decay:  1.0e-5

augmentation:
  p_flip:       0.5
  p_intensity:  0.3
  p_noise:      0.2
  noise_std:    0.05

finetune:
  freeze_encoder_epochs: 0
  encoder_lr_scale:      1.0

output:
  dir:         outputs/seg_baseline
  checkpoint:  outputs/seg_baseline/checkpoint.pth
  best_model:  outputs/seg_baseline/best_model.pth
YAML

echo "  ✓ configs/dino.yaml"
echo "  ✓ configs/seg_pretrained.yaml"
echo "  ✓ configs/seg_baseline.yaml"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cập nhật src/dataset.py nếu chưa có BraTSLazyDataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if ! grep -q "BraTSLazyDataset" src/dataset.py 2>/dev/null; then
    echo ""
    echo "⚠  src/dataset.py chưa có BraTSLazyDataset!"
    echo "   Hãy thay src/dataset.py bằng file dataset.py mới (đã cung cấp)."
    echo "   Sau đó chạy lại script."
    exit 1
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cập nhật train_seg.py và train_dino.py để hỗ trợ lazy loader
# (patch nhỏ: thêm import BraTSLazyDataset và logic use_lazy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python - << 'PYEOF'
import re, sys

# ── Patch train_seg.py ──────────────────────────────────────
with open('train_seg.py', 'r', encoding='utf-8') as f:
    src = f.read()

if 'BraTSLazyDataset' not in src:
    # 1. Thêm import
    src = src.replace(
        'from src.dataset      import PatchDataset',
        'from src.dataset      import PatchDataset\nfrom src.dataset      import BraTSLazyDataset'
    )
    # 2. Thay đoạn Dataset
    old = """    train_ds = PatchDataset(patch_dir, 'train', mode='seg', ram_cache=ram_cache)
    val_ds   = PatchDataset(patch_dir, 'val',   mode='seg', ram_cache=False)"""
    new = """    use_lazy   = cfg['data'].get('use_lazy', False)
    data_dir   = cfg['data'].get('data_dir')
    cache_size = cfg['data'].get('cache_size', 150)
    n_patches  = cfg['data'].get('n_patches', 32)
    nv_patches = cfg['data'].get('n_patches_val', 16)
    patch_sz   = cfg['data'].get('patch_size', 128)

    if use_lazy and data_dir:
        train_ds = BraTSLazyDataset(
            data_dir=data_dir, split='train', n_patches=n_patches,
            patch_size=patch_sz, cache_size=cache_size, mode='seg')
        val_ds = BraTSLazyDataset(
            data_dir=data_dir, split='val', n_patches=nv_patches,
            patch_size=patch_sz, cache_size=max(20, cache_size//5), mode='seg')
    else:
        train_ds = PatchDataset(patch_dir, 'train', mode='seg', ram_cache=ram_cache)
        val_ds   = PatchDataset(patch_dir, 'val',   mode='seg', ram_cache=False)"""
    if old in src:
        src = src.replace(old, new)
        with open('train_seg.py', 'w', encoding='utf-8') as f:
            f.write(src)
        print('  ✓ train_seg.py đã được cập nhật (lazy loader)')
    else:
        print('  ⚠  train_seg.py: không tìm thấy đoạn cần patch — hãy cập nhật thủ công')
else:
    print('  ✓ train_seg.py đã có lazy loader')

# ── Patch train_dino.py ─────────────────────────────────────
with open('train_dino.py', 'r', encoding='utf-8') as f:
    src = f.read()

if 'BraTSLazyDataset' not in src:
    src = src.replace(
        'from src.dataset      import PatchDataset',
        'from src.dataset      import PatchDataset\nfrom src.dataset      import BraTSLazyDataset'
    )
    old = """    dataset = PatchDataset(patch_dir, split='train', mode='ssl',
                           ram_cache=cfg['data'].get('ram_cache', False))"""
    new = """    use_lazy   = cfg['data'].get('use_lazy', False)
    data_dir   = cfg['data'].get('data_dir')
    cache_size = cfg['data'].get('cache_size', 150)
    n_patches  = cfg['data'].get('n_patches', 32)
    patch_sz   = cfg['data'].get('patch_size', 128)

    if use_lazy and data_dir:
        dataset = BraTSLazyDataset(
            data_dir=data_dir, split='train', n_patches=n_patches,
            patch_size=patch_sz, cache_size=cache_size, mode='ssl')
    else:
        dataset = PatchDataset(patch_dir, split='train', mode='ssl',
                               ram_cache=cfg['data'].get('ram_cache', False))"""
    if old in src:
        src = src.replace(old, new)
        with open('train_dino.py', 'w', encoding='utf-8') as f:
            f.write(src)
        print('  ✓ train_dino.py đã được cập nhật (lazy loader)')
    else:
        print('  ⚠  train_dino.py: không tìm thấy đoạn cần patch — hãy cập nhật thủ công')
else:
    print('  ✓ train_dino.py đã có lazy loader')
PYEOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BƯỚC 1: DINO Pretraining
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ $SKIP_DINO -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Bước 1: DINO Self-Supervised Pretraining"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  epochs  : $EPOCHS_DINO"
    echo "  batch   : $BATCH_DINO  (effective: $(( BATCH_DINO * GRAD_ACCUM_DINO )))"
    echo ""

    python train_dino.py --config configs/dino.yaml

    echo ""
    echo "  ✓ Teacher encoder → outputs/dino/best_encoder.pth"
else
    echo ""
    echo "=== Bước 1: DINO — SKIPPED ==="
    if [ ! -f "outputs/dino/best_encoder.pth" ]; then
        echo "  ❌ outputs/dino/best_encoder.pth không tồn tại!"
        echo "     Chạy lại không có --skip_dino để train DINO trước."
        exit 1
    fi
    echo "  ✓ Dùng encoder có sẵn: outputs/dino/best_encoder.pth"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BƯỚC 2: Segmentation — DINO Pretrained
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Bước 2: Segmentation — DINO Pretrained"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  epochs  : $EPOCHS_SEG"
echo "  batch   : $BATCH_SEG  (effective: $(( BATCH_SEG * GRAD_ACCUM_SEG )))"
echo "  Phase 1 (20 ep): encoder frozen"
echo "  Phase 2 (ep 20+): unfreeze, lr encoder x0.01"
echo ""

python train_seg.py --config configs/seg_pretrained.yaml

echo ""
echo "  ✓ Best model → outputs/seg_pretrained/best_model.pth"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BƯỚC 3: Segmentation — Baseline (random init)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ $SKIP_BASELINE -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Bước 3: Segmentation — Baseline"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  (Random init, không dùng DINO encoder)"
    echo ""

    python train_seg.py --config configs/seg_baseline.yaml

    echo ""
    echo "  ✓ Best model → outputs/seg_baseline/best_model.pth"
else
    echo ""
    echo "=== Bước 3: Baseline — SKIPPED ==="
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tổng kết
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║              ✓ Pipeline hoàn thành!          ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Outputs:"
[ -f "outputs/dino/best_encoder.pth" ]         && echo "  ✓ DINO encoder    : outputs/dino/best_encoder.pth"
[ -f "outputs/seg_pretrained/best_model.pth" ] && echo "  ✓ DINO-Pretrained : outputs/seg_pretrained/best_model.pth"
[ -f "outputs/seg_baseline/best_model.pth" ]   && echo "  ✓ Baseline        : outputs/seg_baseline/best_model.pth"
echo ""
