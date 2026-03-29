"""
train_dino.py — DINO Self-Supervised Pretraining

Cập nhật: hỗ trợ H5 dataset 1 kênh, thêm --in_channels / --channel_idx
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
        print("GPU     : CPU mode")
    print()


def cosine_scheduler(base_val, final_val, epochs, warmup=10):
    epochs   = max(epochs, 1)
    warmup   = min(warmup, epochs)
    schedule = np.ones(epochs) * base_val
    if warmup > 0:
        schedule[:warmup] = np.linspace(0, base_val, warmup)
    remaining = epochs - warmup
    if remaining > 0:
        t = np.arange(remaining)
        schedule[warmup:] = final_val + 0.5 * (base_val - final_val) * \
                            (1 + np.cos(np.pi * t / remaining))
    return schedule


def train(cfg):
    out_dir = cfg['output']['dir']
    os.makedirs(out_dir, exist_ok=True)

    DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DEVICE_TYPE = DEVICE.type
    log_system()
    set_seed(cfg['train']['seed'])

    patch_dir   = cfg['data']['patch_dir']
    nw          = cfg['train']['num_workers']
    bs          = cfg['train']['batch_size']
    in_channels = cfg['data'].get('in_channels', 1)
    channel_idx = cfg['data'].get('channel_idx', 3)
    val_split   = cfg['data'].get('val_split', 0.2)
    ppv         = cfg['data'].get('patches_per_volume', 8)

    dataset = PatchDataset(
        patch_dir,
        split='train',
        mode='ssl',
        ram_cache=cfg['data'].get('ram_cache', False),
        in_channels=in_channels,
        channel_idx=channel_idx,
        val_split=val_split,
        patches_per_volume=ppv,
        seed=cfg['train']['seed'],
    )
    loader  = DataLoader(
        dataset, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=(DEVICE_TYPE == 'cuda'), drop_last=True,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )
    print(f"Dataset : {len(dataset)} patches | {len(loader)} batches/epoch\n")

    encoder    = ConvNeXtTiny3D(in_channels=in_channels).to(DEVICE)
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
    print(f"Encoder : {n_params/1e6:.1f}M params | in_channels={in_channels}")

    num_epochs = cfg['train']['num_epochs']
    warmup_ep  = cfg['train'].get('warmup_epochs', 10)
    base_lr    = cfg['optimizer']['base_lr'] * bs / 256
    min_lr     = cfg['optimizer']['min_lr']
    wd         = cfg['optimizer']['weight_decay']

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

    lr_schedule  = cosine_scheduler(base_lr, min_lr, num_epochs, warmup_ep)
    mom_schedule = cosine_scheduler(cfg['dino']['momentum'], 1.0, num_epochs, warmup=0)

    scaler     = GradScaler(DEVICE_TYPE) if DEVICE_TYPE == 'cuda' else None
    grad_accum = cfg['train'].get('grad_accum', 1)

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
    print(f"  DINO Pretraining — {num_epochs} epochs  [{DEVICE_TYPE.upper()}]")
    print(f"  batch={bs} | lr={base_lr:.2e} | grad_accum={grad_accum}")
    print(f"{'─'*55}\n")

    for epoch in range(start_epoch, num_epochs):
        sched_idx = min(epoch, len(lr_schedule) - 1)
        for g in optimizer.param_groups:
            g['lr'] = lr_schedule[sched_idx]
        model.momentum = mom_schedule[sched_idx]

        model.train()
        ep_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"DINO Ep {epoch+1:3d}/{num_epochs}",
                    ncols=85, leave=True)

        for step, x in enumerate(pbar):
            x = x.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                views = multi_crop(x)

            with autocast(DEVICE_TYPE, enabled=(DEVICE_TYPE == 'cuda')):
                student_out, teacher_out = model(views)
                loss = criterion(student_out, teacher_out, epoch) / grad_accum

            if not torch.isfinite(loss):
                print(f"\n⚠ NaN/Inf loss ở step {step}, skip")
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 3.0
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                model.update_teacher()

            ep_loss += loss.item() * grad_accum

            if step % 20 == 0:
                mem = torch.cuda.memory_allocated()/1e9 if DEVICE_TYPE=='cuda' else 0
                pbar.set_postfix(
                    loss=f"{loss.item()*grad_accum:.4f}",
                    lr=f"{lr_schedule[sched_idx]:.2e}",
                    gpu=f"{mem:.1f}GB"
                )

        avg = ep_loss / len(loader)
        history.append(avg)
        print(f"  Ep[{epoch+1}/{num_epochs}] Loss={avg:.4f} | "
              f"LR={lr_schedule[sched_idx]:.2e} | m={model.momentum:.4f}")

        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss, 'history': history,
        }, ckpt_path)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.teacher_encoder.state_dict(),
                       cfg['output']['best_encoder'])
            print(f"  ✓ Best teacher encoder saved (loss={best_loss:.4f})")

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
    p.add_argument('--config',      default='configs/dino.yaml')
    p.add_argument('--epochs',      type=int, default=None)
    p.add_argument('--batch',       type=int, default=None)
    p.add_argument('--workers',     type=int, default=None)
    p.add_argument('--patch_dir',   default=None)
    p.add_argument('--in_channels', type=int, default=None,
                   help='Số kênh input: 1 (flair) hoặc 4 (tất cả modalities)')
    p.add_argument('--channel_idx', type=int, default=None,
                   help='Kênh dùng khi in_channels=1 (0=t1,1=t1ce,2=t2,3=flair)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = load_cfg(args.config)
    if args.epochs:      cfg['train']['num_epochs']     = args.epochs
    if args.batch:       cfg['train']['batch_size']     = args.batch
    if args.workers:     cfg['train']['num_workers']    = args.workers
    if args.patch_dir:   cfg['data']['patch_dir']       = args.patch_dir
    if args.in_channels: cfg['data']['in_channels']     = args.in_channels
    if args.channel_idx is not None:
                         cfg['data']['channel_idx']     = args.channel_idx
    train(cfg)