"""
train_dino.py — DINO Self-Supervised Pretraining

Usage:
    python train_dino.py --config configs/dino.yaml

Output: outputs/dino/best_encoder.pth  ← dùng cho train_seg.py

Lưu ý:
  - DINO teacher encoder thường tốt hơn student → lưu teacher
  - Batch size 4-8 là đủ (DINO không cần large batch như SimCLR)
  - 100-200 epochs là đủ cho BraTS với ConvNeXt-Tiny
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

from src.augmentation import DINOMultiCrop3D
from src.dataset      import PatchDataset
from src.dino         import DINO, DINOLoss
from src.encoder      import ConvNeXtTiny3D


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def load_cfg(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def log_system():
    print(f"PyTorch : {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU     : {p.name}  ({p.total_memory/1e9:.1f} GB)")
        print(f"CUDA    : {torch.version.cuda}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    else:
        print("GPU     : ✗ No CUDA!")
    print()


def cosine_scheduler(base_val, final_val, epochs, warmup=10):
    """Cosine schedule với warmup."""
    schedule = np.ones(epochs) * base_val
    if warmup > 0:
        schedule[:warmup] = np.linspace(0, base_val, warmup)
    t = np.arange(epochs - warmup)
    schedule[warmup:] = final_val + 0.5 * (base_val - final_val) * \
                        (1 + np.cos(np.pi * t / (epochs - warmup)))
    return schedule


def train(cfg):
    out_dir = cfg['output']['dir']
    os.makedirs(out_dir, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_system()
    set_seed(cfg['train']['seed'])

    # ── Dataset ──
    patch_dir = cfg['data']['patch_dir']
    nw        = cfg['train']['num_workers']
    bs        = cfg['train']['batch_size']

    dataset = PatchDataset(patch_dir, split='train', mode='ssl',
                           ram_cache=cfg['data'].get('ram_cache', False))
    loader  = DataLoader(
        dataset, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    print(f"Dataset : {len(dataset)} patches | {len(loader)} batches/epoch\n")

    # ── Model ──
    encoder    = ConvNeXtTiny3D(in_channels=4).to(DEVICE)
    model      = DINO(encoder, feat_dim=768,
                      out_dim=cfg['dino']['out_dim'],
                      momentum=cfg['dino']['momentum']).to(DEVICE)
    multi_crop = DINOMultiCrop3D(
        global_scale=tuple(cfg['dino']['global_scale']),
        local_scale=tuple(cfg['dino']['local_scale']),
        n_local=cfg['dino']['n_local_crops'],
    ).to(DEVICE)
    criterion  = DINOLoss(
        out_dim=cfg['dino']['out_dim'],
        n_global_crops=2,
        warmup_teacher_temp=cfg['dino']['warmup_teacher_temp'],
        teacher_temp=cfg['dino']['teacher_temp'],
        warmup_teacher_temp_epochs=cfg['dino']['warmup_teacher_temp_epochs'],
        student_temp=cfg['dino']['student_temp'],
        center_momentum=cfg['dino']['center_momentum'],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.student_encoder.parameters())
    print(f"Encoder : {n_params/1e6:.1f}M params")

    # ── Optimizer: AdamW với cosine LR ──
    num_epochs = cfg['train']['num_epochs']
    warmup_ep  = cfg['train'].get('warmup_epochs', 10)
    base_lr    = cfg['optimizer']['base_lr'] * bs / 256  # linear scaling rule
    min_lr     = cfg['optimizer']['min_lr']
    wd         = cfg['optimizer']['weight_decay']

    # Separate params: no weight decay cho norm + bias
    decay_params    = [p for n, p in model.named_parameters()
                       if 'norm' not in n and 'bias' not in n
                       and 'gamma' not in n and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters()
                       if ('norm' in n or 'bias' in n or 'gamma' in n)
                       and p.requires_grad]

    optimizer = optim.AdamW([
        {'params': decay_params,    'weight_decay': wd},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=base_lr)

    # LR schedule
    lr_schedule = cosine_scheduler(base_lr, min_lr, num_epochs, warmup_ep)
    # Momentum schedule: 0.996 → 1.0
    mom_schedule = cosine_scheduler(
        cfg['dino']['momentum'], 1.0, num_epochs, warmup=0
    )

    scaler     = GradScaler('cuda')
    grad_accum = cfg['train'].get('grad_accum', 1)

    # ── Resume ──
    start_epoch = 0; best_loss = float('inf'); history = []
    ckpt_path   = cfg['output']['checkpoint']
    if os.path.exists(ckpt_path):
        print(f'Resuming from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_loss   = ckpt.get('best_loss', float('inf'))
        history     = ckpt.get('history', [])
        print(f"  epoch={start_epoch}, best_loss={best_loss:.4f}\n")

    print(f"{'─'*55}")
    print(f"  DINO Pretraining — {num_epochs} epochs")
    print(f"  batch={bs} | lr={base_lr:.2e} | momentum={cfg['dino']['momentum']}")
    print(f"{'─'*55}\n")

    for epoch in range(start_epoch, num_epochs):
        # Update LR và momentum
        for g in optimizer.param_groups:
            g['lr'] = lr_schedule[epoch]
        model.momentum = mom_schedule[epoch]

        model.train()
        ep_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"DINO Ep {epoch+1:3d}/{num_epochs}",
                    ncols=85, leave=True)

        for step, x in enumerate(pbar):
            x = x.to(DEVICE, non_blocking=True)

            # Tạo multi-crop views trên GPU
            with torch.no_grad():
                views = multi_crop(x)

            with autocast('cuda'):
                student_out, teacher_out = model(views)
                loss = criterion(student_out, teacher_out, epoch) / grad_accum

            if not torch.isfinite(loss):
                print(f"\n⚠ NaN/Inf loss ở step {step}, skip")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                # Clip student gradients (quan trọng cho DINO stability)
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 3.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Update teacher EMA
                model.update_teacher()

            ep_loss += loss.item() * grad_accum

            if step % 20 == 0:
                mem = torch.cuda.memory_allocated()/1e9 if DEVICE.type=='cuda' else 0
                pbar.set_postfix(
                    loss=f"{loss.item()*grad_accum:.4f}",
                    lr=f"{lr_schedule[epoch]:.2e}",
                    gpu=f"{mem:.1f}GB"
                )

        avg = ep_loss / len(loader)
        history.append(avg)
        print(f"  Ep[{epoch+1}/{num_epochs}] Loss={avg:.4f} | "
              f"LR={lr_schedule[epoch]:.2e} | "
              f"m={model.momentum:.4f}")

        # Checkpoint
        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss, 'history': history,
        }, ckpt_path)

        if avg < best_loss:
            best_loss = avg
            # Lưu TEACHER encoder (thường tốt hơn student)
            torch.save(
                model.teacher_encoder.state_dict(),
                cfg['output']['best_encoder']
            )
            print(f"  ✓ Best teacher encoder saved (loss={best_loss:.4f})")

    # Plot loss
    plt.figure(figsize=(10, 4))
    plt.plot(history, marker='o', linewidth=2, markersize=4, color='steelblue')
    plt.title('DINO Pretraining Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dino_loss.png'), dpi=150)
    print(f"\n✓ Done!  Best loss: {best_loss:.4f}")
    print(f"✓ Teacher encoder → {cfg['output']['best_encoder']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',  default='configs/dino.yaml')
    p.add_argument('--epochs',  type=int, default=None)
    p.add_argument('--batch',   type=int, default=None)
    p.add_argument('--workers', type=int, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = load_cfg(args.config)
    if args.epochs:  cfg['train']['num_epochs']  = args.epochs
    if args.batch:   cfg['train']['batch_size']  = args.batch
    if args.workers: cfg['train']['num_workers'] = args.workers
    train(cfg)