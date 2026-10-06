"""HCNet model core reconstructed from manuscript Sections 2.6–2.9 and Fig. 1.

Inputs: aligned single-channel pseudo-images [B, 1, H, W].
Classes: 0 = low risk, 1 = high risk. Outputs are two raw logits.
Undisclosed settings use configurable defaults; see README.md.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class ResidualBlock(nn.Module):
    """Two 3×3 convolutions; elementwise dropout as in Section 2.7."""

    def __init__(self, channels: int, dropout: float = 0.3):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.output = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x):
        return self.output(x + self.residual(x))


class DownsampleStage(nn.Module):
    """Stages 1/2: strided conv → optional CA → BN → ReLU → add → pool."""

    def __init__(self, cin: int, cout: int, kernel: int, attention: nn.Module):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(cin, cout, kernel, stride=2, padding=kernel // 2, bias=False),
            attention,
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        # Projection resolves channel/spatial mismatches at residual addition.
        self.skip = nn.Conv2d(cin, cout, 1, stride=2, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        return self.pool(self.main(x) + self.skip(x))


class GeneCNN(nn.Module):
    def __init__(self, reduction: int = 16, dropout: float = 0.3):
        super().__init__()
        if reduction < 1:
            raise ValueError("reduction must be positive")
        self.stage1 = DownsampleStage(1, 32, 7, ChannelAttention(32, reduction))
        self.stage2 = DownsampleStage(32, 64, 5, nn.Identity())
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128, dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, 2)

    def forward_features(self, x):
        """Return stage-3 feature maps, before global pooling."""
        if x.ndim != 4 or x.shape[1] != 1 or min(x.shape[-2:]) < 11:
            raise ValueError("Expected [B, 1, H, W] with H,W >= 11")
        return self.stage3(self.stage2(self.stage1(x)))

    def forward(self, x, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.classifier(self.pool(features).flatten(1))
        return (logits, features) if return_features else logits


class FocalLoss(nn.Module):
    """Stable two-class focal loss; alpha gives weights for classes (0, 1)."""

    def __init__(self, gamma: float = 2.0, alpha=(1.0, 1.0)):
        super().__init__()
        weights = torch.as_tensor(alpha, dtype=torch.float32)
        if (gamma < 0 or weights.shape != (2,)
                or not torch.isfinite(weights).all() or (weights < 0).any()):
            raise ValueError("gamma must be nonnegative; alpha needs two finite nonnegative weights")
        self.gamma = gamma
        self.register_buffer("alpha", weights)

    def forward(self, logits, labels):
        # labels must be torch.long, shape [B], and contain only 0 or 1.
        if logits.ndim != 2 or logits.shape[1] != 2 or labels.shape != logits.shape[:1]:
            raise ValueError("Expected logits [B, 2] and labels [B]")
        log_pt = F.log_softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
        return (-self.alpha[labels] * (1 - log_pt.exp()).pow(self.gamma) * log_pt).mean()


class HCNet(nn.Module):
    """Student/EMA teacher with focal classification and stage-3 MSE losses."""

    def __init__(self, ema_decay: float = 0.99, consistency_weight: float = 1.0,
                 gamma: float = 2.0, alpha=(1.0, 1.0),
                 reduction: int = 16, dropout: float = 0.3):
        super().__init__()
        if not 0 <= ema_decay < 1 or consistency_weight < 0:
            raise ValueError("Require 0 <= ema_decay < 1 and consistency_weight >= 0")
        self.student = GeneCNN(reduction, dropout)
        self.teacher = deepcopy(self.student).requires_grad_(False).eval()
        self.focal = FocalLoss(gamma, alpha)
        self.ema_decay = ema_decay
        self.consistency_weight = consistency_weight

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()  # Keep dropout off and use EMA BatchNorm statistics.
        return self

    def forward(self, x, return_features: bool = False):
        """Student inference; high-risk probability = logits.softmax(1)[:, 1]."""
        return self.student(x, return_features=return_features)

    def loss(self, weak, strong, labels):
        """Views must be from the same patients, in the same batch order."""
        if weak.shape != strong.shape:
            raise ValueError("Weak/strong views must have identical shapes")
        logits, student_features = self.student(weak, return_features=True)
        with torch.no_grad():
            teacher_features = self.teacher.forward_features(strong)
        classification = self.focal(logits, labels)
        consistency = F.mse_loss(student_features, teacher_features)
        return {
            "total": classification + self.consistency_weight * consistency,
            "classification": classification,
            "consistency": consistency,
        }

    @torch.no_grad()
    def update_teacher(self):
        """Call once AFTER each optimizer.step() on the student."""
        decay = self.ema_decay
        for teacher, student in zip(self.teacher.parameters(), self.student.parameters()):
            teacher.mul_(decay).add_(student, alpha=1 - decay)
        # BN buffers also need updates; integer batch counters cannot be averaged.
        for teacher, student in zip(self.teacher.buffers(), self.student.buffers()):
            if teacher.is_floating_point():
                teacher.mul_(decay).add_(student, alpha=1 - decay)
            else:
                teacher.copy_(student)
