"""
src/losses.py — Loss functions cho segmentation tumor não

Binary tumor segmentation rất imbalanced:
  Tumor  ≈ 1-3% voxels
  Background ≈ 97-99% voxels

→ Cần loss chống imbalance:
  TverskyLoss: penalize FN nặng hơn FP (đừng bỏ sót tumor)
  FocalLoss:   down-weight easy background voxels
  
  Combined = Tversky + Focal + Deep Supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky index: TP / (TP + alpha*FP + beta*FN)
    
    alpha=0.3, beta=0.7:
      FN penalized 2.3× hơn FP → model học cách không bỏ sót tumor
      Quan trọng vì miss tumor (FN) nguy hiểm hơn false alarm (FP)
    
    smooth=1.0 (thay vì 1e-5): ổn định hơn khi tumor rất nhỏ
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p  = torch.sigmoid(logits)
        # Flatten spatial dims
        p  = p.contiguous().view(-1)
        t  = targets.contiguous().view(-1)

        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()

        tversky = (tp + self.smooth) / \
                  (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class FocalLoss3D(nn.Module):
    """
    Binary Focal Loss:
      FL = -alpha_t * (1-p_t)^gamma * log(p_t)
    
    alpha=0.75: weight cao hơn cho tumor (minority class)
    gamma=2.0:  down-weight easy negatives (background rất dễ đoán)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        p_t = torch.sigmoid(logits) * targets + \
              (1 - torch.sigmoid(logits)) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal   = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class CombinedSegLoss(nn.Module):
    """
    Tversky + Focal + Deep Supervision.
    
    Deep supervision weights (từ nnU-Net paper):
      main output : 1.0
      ds3 (1/16)  : 0.4
      ds2 (1/8)   : 0.2
      ds1 (1/4)   : 0.1
      
    Tổng weights được normalize về 1.0.
    """

    DS_WEIGHTS = [1.0, 0.4, 0.2, 0.1]

    def __init__(
        self,
        tversky_weight: float = 0.7,
        focal_weight: float = 0.3,
    ):
        super().__init__()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.focal   = FocalLoss3D(alpha=0.75, gamma=2.0)
        self.tw      = tversky_weight
        self.fw      = focal_weight

    def _single(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return (
            self.tw * self.tversky(logits, targets) +
            self.fw * self.focal(logits, targets)
        )

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            # outputs = (main, ds3, ds2, ds1) — deep supervision
            total = 0.0
            w_sum = sum(self.DS_WEIGHTS[:len(outputs)])
            for out, w in zip(outputs, self.DS_WEIGHTS[:len(outputs)]):
                total += w * self._single(out, targets)
            return total / w_sum
        return self._single(outputs, targets)


# ─────────────────────────────────────────────
# Metrics (không gradient)
# ─────────────────────────────────────────────

def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> float:
    """Dice Score — dùng trong validation, không có gradient."""
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    preds = (torch.sigmoid(logits) > threshold).float()
    p_flat = preds.contiguous().view(-1)
    t_flat = targets.contiguous().view(-1)

    inter = (p_flat * t_flat).sum()
    union = p_flat.sum() + t_flat.sum()
    return ((2 * inter + smooth) / (union + smooth)).item()


def hausdorff_approx(logits, targets, percentile=95):
    """
    Approximate Hausdorff distance (HD95) — metric phụ để theo dõi.
    Không dùng trong loss, chỉ để log cuối epoch.
    """
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    pred = (torch.sigmoid(logits) > 0.5).float()
    # Simple approximate: dựa trên surface voxels
    # Để đơn giản, dùng scipy nếu có, không thì bỏ qua
    try:
        from scipy.ndimage import distance_transform_edt
        import numpy as np
        p = pred[0, 0].cpu().numpy().astype(bool)
        t = targets[0, 0].cpu().numpy().astype(bool)
        if p.any() and t.any():
            dt_p = distance_transform_edt(~p)
            dt_t = distance_transform_edt(~t)
            hd_pt = np.percentile(dt_t[p],  percentile)
            hd_tp = np.percentile(dt_p[t], percentile)
            return float(max(hd_pt, hd_tp))
    except Exception:
        pass
    return float('nan')