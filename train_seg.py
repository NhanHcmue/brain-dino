"""
train_seg.py — Segmentation Training (Pretrained DINO + Baseline)

Usage:
    # DINO pretrained (mô hình chính):
    python train_seg.py --config configs/seg_pretrained.yaml

    # Baseline (random init để so sánh):
    python train_seg.py --config configs/seg_baseline.yaml

    # Override từ CLI:
    python train_seg.py --config configs/seg_pretrained.yaml --batch 4 --workers 8

Training strategy (pretrained):
    Phase 1 (epoch 0-19):   Freeze encoder, chỉ train decoder
    Phase 2 (epoch 20+):    Unfreeze encoder, lr_encoder = lr × 0.01
    
    Lý do 2-phase:
      - Decoder initialized randomly → cần ổn định trước
      - Nếu unfreeze ngay, strong DINO encoder bị "corrupted"
"""

import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentation import GPUAugmentation3D
from src.dataset      import PatchDataset
from src.encoder      import ConvNeXtTiny3D
from src.losses       import CombinedSegLoss, dice_score
from src.models       import ConvNeXtNNUNet


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def load_cfg(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def gpu_mem():
    if torch.cuda.is_available():
        u = torch.cuda.memory_allocated() / 1e9
        t = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{u:.1f}/{t:.0f}GB"
    return "N/A"


def log_system():
    print(f"PyTorch : {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU     : {p.name}  ({p.total_memory/1e9:.1f} GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        print(f"TF32    : enabled")
    else:
        print("GPU     : ✗ No CUDA!")
    print()


def make_optimizer(model, lr, wd, encoder_path, enc_lr_scale):
    """Tạo optimizer với lr khác nhau cho encoder và decoder."""
    if encoder_path:
        enc_params = [p for n, p in model.named_parameters()
                      if 'encoder' in n and p.requires_grad]
        dec_params = [p for n, p in model.named_parameters()
                      if 'encoder' not in n and p.requires_grad]
        return optim.AdamW([
            {'params': enc_params, 'lr': lr * enc_lr_scale},
            {'params': dec_params, 'lr': lr},
        ], weight_decay=wd)
    return optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=wd,
    )


def cosine_lr(optimizer, epoch, num_epochs, warmup=5, base_lr=2e-4, min_lr=1e-6):
    if epoch < warmup:
        factor = (epoch + 1) / warmup
    else:
        factor = min_lr / base_lr + 0.5 * (1 - min_lr / base_lr) * \
                 (1 + np.cos(np.pi * (epoch - warmup) / (num_epochs - warmup)))
    for g in optimizer.param_groups:
        g['lr'] = g.get('initial_lr', base_lr) * factor


def plot_and_save(history, label, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_loss'], label='Train', color='steelblue')
    ax1.plot(history['val_loss'],   label='Val',   color='coral')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(True, alpha=0.4)

    ax2.plot(history['val_dice'], color='green', linewidth=2)
    ax2.axhline(0.90, color='red',    linestyle='--', alpha=0.7, label='0.90')
    ax2.axhline(0.85, color='orange', linestyle='--', alpha=0.7, label='0.85')
    ax2.set_title('Val Dice'); ax2.legend(); ax2.grid(True, alpha=0.4)
    ax2.set_ylim(0, 1)

    plt.suptitle(f'UNet {label}'); plt.tight_layout()
    path = os.path.join(out_dir, 'training_history.png')
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"✓ Plot → {path}")
    if history['val_dice']:
        print(f"  Best Dice: {max(history['val_dice']):.4f}")


def train(cfg):
    out_dir = cfg['output']['dir']
    os.makedirs(out_dir, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_system()
    set_seed(cfg['train']['seed'])

    encoder_path = cfg['pretrain'].get('encoder_path')
    label        = 'DINO-Pretrained' if encoder_path else 'Baseline'
    print(f"Mode: {label}\n")

    # ── Dataset ──
    patch_dir = cfg['data']['patch_dir']
    nw        = cfg['train']['num_workers']
    bs        = cfg['train']['batch_size']
    ram_cache = cfg['data'].get('ram_cache', False)

    train_ds = PatchDataset(patch_dir, 'train', mode='seg', ram_cache=ram_cache)
    val_ds   = PatchDataset(patch_dir, 'val',   mode='seg', ram_cache=False)

    train_loader = DataLoader(
        train_ds, bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, bs, shuffle=False,
        num_workers=max(1, nw // 2), pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
    )
    print(f"Train: {len(train_ds)} patches | Val: {len(val_ds)} patches")
    print(f"Loaders: {len(train_loader)} train | {len(val_loader)} val\n")

    # ── Model ──
    encoder = ConvNeXtTiny3D(in_channels=4).to(DEVICE)

    if encoder_path:
        assert os.path.exists(encoder_path), f"Encoder not found: {encoder_path}"
        state = torch.load(encoder_path, map_location=DEVICE, weights_only=True)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"✓ Loaded DINO encoder: {encoder_path}")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print("[BASELINE] Random init encoder")

    model = ConvNeXtNNUNet(encoder, num_classes=1).to(DEVICE)

    # torch.compile nếu có (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("✓ torch.compile enabled")
        except Exception:
            pass

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {total:.1f}M params | GPU: {gpu_mem()}\n")

    # ── Training setup ──
    gpu_aug    = GPUAugmentation3D(**cfg['augmentation']).to(DEVICE)
    criterion  = CombinedSegLoss()
    scaler     = GradScaler('cuda')
    grad_accum = cfg['train'].get('grad_accum', 1)
    num_epochs = cfg['train']['num_epochs']
    base_lr    = cfg['optimizer']['lr']
    wd         = cfg['optimizer']['weight_decay']
    freeze_ep  = cfg.get('finetune', {}).get('freeze_encoder_epochs', 0) \
                 if encoder_path else 0
    enc_scale  = cfg.get('finetune', {}).get('encoder_lr_scale', 0.01) \
                 if encoder_path else 1.0

    # Phase 1: freeze encoder
    if freeze_ep > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print(f"Phase 1: encoder frozen for {freeze_ep} epochs\n")

    optimizer = make_optimizer(model, base_lr, wd, encoder_path, enc_scale)
    # Store initial lr cho cosine scheduler
    for g in optimizer.param_groups:
        g['initial_lr'] = g['lr']

    # ── Resume ──
    start_epoch = 0; best_dice = 0.0
    ckpt_path   = cfg['output']['checkpoint']
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        best_dice   = ckpt.get('best_dice', 0.0)
        print(f"Resumed epoch {start_epoch}, best_dice={best_dice:.4f}\n")

    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}

    print(f"{'─'*60}")
    print(f"  Segmentation [{label}] — {num_epochs} epochs")
    print(f"  batch={bs} | grad_accum={grad_accum} | "
          f"effective_batch={bs*grad_accum}")
    print(f"{'─'*60}\n")

    for epoch in range(start_epoch, num_epochs):

        # Phase 2: unfreeze encoder
        if epoch == freeze_ep and freeze_ep > 0:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = make_optimizer(model, base_lr, wd, encoder_path, enc_scale)
            for g in optimizer.param_groups:
                g['initial_lr'] = g['lr']
            print(f"\n[Phase 2] Encoder unfrozen at epoch {epoch+1}\n")

        # LR update
        cosine_lr(optimizer, epoch, num_epochs,
                  warmup=5, base_lr=base_lr, min_lr=1e-6)

        # ── Train ──
        model.train()
        tr_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"[{label[:4]}] Ep {epoch+1:3d}/{num_epochs}",
            ncols=85, leave=True,
        )

        for step, (img, seg) in enumerate(pbar):
            img = img.to(DEVICE, non_blocking=True)
            seg = seg.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                img = gpu_aug(img)

            with autocast('cuda'):
                out  = model(img)
                loss = criterion(out, seg) / grad_accum

            if not torch.isfinite(loss):
                print(f"\n⚠ NaN loss ở step {step}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            tr_loss += loss.item() * grad_accum

            if step % 50 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item()*grad_accum:.4f}",
                    gpu=gpu_mem(),
                )

        # ── Validation ──
        model.eval()
        va_loss = va_dice = 0.0
        with torch.no_grad():
            for img, seg in tqdm(val_loader, desc="  val",
                                 ncols=85, leave=False):
                img = img.to(DEVICE, non_blocking=True)
                seg = seg.to(DEVICE, non_blocking=True)
                with autocast('cuda'):
                    out  = model(img)
                    loss = criterion(out, seg)
                va_loss += loss.item()
                va_dice += dice_score(out, seg)

        atl = tr_loss / len(train_loader)
        avl = va_loss / len(val_loader)
        adc = va_dice / len(val_loader)

        cur_lr = optimizer.param_groups[-1]['lr']
        history['train_loss'].append(atl)
        history['val_loss'].append(avl)
        history['val_dice'].append(adc)

        print(
            f"[{label[:4]}] Ep[{epoch+1}/{num_epochs}] "
            f"Loss {atl:.4f}/{avl:.4f} | "
            f"Dice {adc:.4f} | "
            f"LR {cur_lr:.2e} | "
            f"GPU {gpu_mem()}"
        )

        # Checkpoint
        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'best_dice': best_dice,
        }, ckpt_path)

        if adc > best_dice:
            best_dice = adc
            torch.save(model.state_dict(), cfg['output']['best_model'])
            print(f"  ✓ Best model saved (Dice={best_dice:.4f})")

    print(f"\n[{label}] Done!  Best Val Dice: {best_dice:.4f}")
    plot_and_save(history, label, out_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',  required=True)
    p.add_argument('--encoder', default=None)
    p.add_argument('--epochs',  type=int, default=None)
    p.add_argument('--batch',   type=int, default=None)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--accum',   type=int, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = load_cfg(args.config)
    if args.encoder: cfg['pretrain']['encoder_path'] = args.encoder
    if args.epochs:  cfg['train']['num_epochs']      = args.epochs
    if args.batch:   cfg['train']['batch_size']      = args.batch
    if args.workers: cfg['train']['num_workers']     = args.workers
    if args.accum:   cfg['train']['grad_accum']      = args.accum
    train(cfg)