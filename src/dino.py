"""
src/dino.py — DINO Self-Supervised Pretraining

Tại sao DINO tốt hơn SimCLR cho medical imaging?
  1. Không cần large batch (SimCLR cần 256+, DINO ổn với 8-16)
  2. Self-distillation: teacher centering ngăn collapse không cần negatives
  3. Multi-crop: học từ local-global consistency → tốt cho tumor nhỏ
  4. Representations tốt hơn cho dense prediction (segmentation)
     — đã chứng minh trong DINO paper và nhiều medical imaging benchmark

Architecture:
  Student: ConvNeXtTiny3D + DINOHead (được update bằng gradient)
  Teacher: EMA copy của student     (được update bằng momentum)
  
  Loss = cross_entropy(student_softmax, teacher_softmax_centered)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """
    3-layer MLP với weight-normalized last layer.
    Output: logits để so sánh student-teacher (không normalize thành prob ở đây).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        # Weight-norm last layer — không cần BN, ổn định hơn
        self.last = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last.weight_g.data.fill_(1)
        self.last.weight_g.requires_grad = False  # freeze norm coefficient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last(x)


class DINOLoss(nn.Module):
    """
    DINO loss với centering và sharpening.
    
    - Teacher output được center (trừ running mean) → ngăn collapse
    - Student output được sharpen (chia temp thấp)
    - Loss = H(teacher_soft, student_soft) trên tất cả cặp global-* views
    """

    def __init__(
        self,
        out_dim: int = 65536,
        n_global_crops: int = 2,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_global_crops  = n_global_crops
        self.register_buffer('center', torch.zeros(1, out_dim))

        # Teacher temp warmup schedule
        self.teacher_temp_schedule = torch.cat([
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(1000) * teacher_temp,
        ])

    def forward(self, student_out, teacher_out, epoch: int):
        """
        student_out: list of (B, out_dim), len = n_views
        teacher_out: list of (B, out_dim), len = n_global_crops only
        """
        t_temp = self.teacher_temp_schedule[epoch].item()

        # Teacher: center + sharpen
        teacher_probs = [
            F.softmax((t - self.center) / t_temp, dim=-1).detach()
            for t in teacher_out
        ]

        # Student: sharpen
        student_log = [
            F.log_softmax(s / self.student_temp, dim=-1)
            for s in student_out
        ]

        loss = 0.0; n_pairs = 0
        for t_idx, t_soft in enumerate(teacher_probs):
            for s_idx, s_log in enumerate(student_log):
                if s_idx == t_idx:
                    continue  # bỏ qua cặp giống nhau
                loss -= (t_soft * s_log).sum(dim=-1).mean()
                n_pairs += 1

        loss = loss / max(1, n_pairs)

        # Update center (EMA của teacher outputs)
        with torch.no_grad():
            batch_center = torch.cat(teacher_out).mean(dim=0, keepdim=True)
            self.center  = self.center * self.center_momentum + \
                           batch_center * (1 - self.center_momentum)

        return loss


class DINO(nn.Module):
    """
    Full DINO model:
      student_encoder + student_head
      teacher_encoder + teacher_head (EMA, no grad)
    
    Sau khi train xong, chỉ dùng teacher_encoder.state_dict()
    vì teacher thường có representation tốt hơn student.
    """

    def __init__(
        self,
        encoder,               # ConvNeXtTiny3D
        feat_dim: int = 768,
        out_dim: int = 65536,
        momentum: float = 0.996,
    ):
        super().__init__()
        self.student_encoder = encoder
        self.teacher_encoder = copy.deepcopy(encoder)

        self.student_head = DINOHead(feat_dim, out_dim)
        self.teacher_head = DINOHead(feat_dim, out_dim)

        # Teacher không train bằng gradient
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # Copy weights từ student sang teacher
        self.teacher_encoder.load_state_dict(encoder.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        self.momentum = momentum

    @torch.no_grad()
    def update_teacher(self):
        """EMA update: teacher = m*teacher + (1-m)*student."""
        for s, t in zip(self.student_encoder.parameters(),
                        self.teacher_encoder.parameters()):
            t.data = t.data * self.momentum + s.data * (1.0 - self.momentum)
        for s, t in zip(self.student_head.parameters(),
                        self.teacher_head.parameters()):
            t.data = t.data * self.momentum + s.data * (1.0 - self.momentum)

    def forward(self, views):
        """
        views: list [global_1, global_2, local_1, ..., local_n]
        Returns: (student_outputs, teacher_outputs)
        """
        global_views = views[:2]

        # Student: process ALL views
        student_out = [
            self.student_head(self.student_encoder.forward_flat(v))
            for v in views
        ]

        # Teacher: process global views only (no_grad já setado nos params)
        with torch.no_grad():
            teacher_out = [
                self.teacher_head(self.teacher_encoder.forward_flat(v))
                for v in global_views
            ]

        return student_out, teacher_out