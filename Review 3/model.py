"""
Review 3 — Model
================
Architecture (per Review 1 proposal and survey of the cited papers):

    Log-mel spectrogram (3, 224, 224)
        → ResNet50 backbone (ImageNet pretrained, FC + avgpool stripped)
        → 2D feature map (B, 2048, 7, 7)
        → Squeeze-and-Excitation block (channel attention; Hu et al. 2018)
        → flatten spatial → 49 tokens of dim 2048
        → Multi-Head Self-Attention (spatial context; Vaswani et al. 2017)
        → residual + LayerNorm
        → GAP over tokens
        → MLP classifier (Dropout → Linear → ReLU → Dropout → Linear)
        → logits (B, num_classes)

The model exposes hooks that the XAI module needs:
* ``backbone_last_conv`` — for Grad-CAM
* ``last_attention_weights`` — for attention-map visualisation
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import ModelConfig


# ─────────────────────────────────────────────────────────────
# SE BLOCK (Hu et al. — CVPR 2018)
# ─────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns per-channel weights and rescales."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ─────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────

class HeartSoundModel(nn.Module):
    """ResNet50 + SE + Multi-Head Attention + classifier head."""

    def __init__(self, cfg: ModelConfig, pretrained: bool = True):
        super().__init__()
        self.cfg = cfg

        # Pretrained ResNet50 backbone, FC + avgpool stripped.
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Children up to (but excluding) avgpool + fc
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        # Cached reference for Grad-CAM. ``layer4[-1]`` is the last
        # Bottleneck block of ResNet50 — the conventional Grad-CAM target.
        self.backbone_last_conv = self.feature_extractor[-1][-1]

        if cfg.use_se:
            self.se = SEBlock(cfg.feature_dim, reduction=cfg.se_reduction)
        else:
            self.se = nn.Identity()

        if cfg.use_mha:
            self.mha = nn.MultiheadAttention(
                embed_dim=cfg.feature_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.mha_dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(cfg.feature_dim)
        else:
            self.mha = nn.Identity()
            self.norm = nn.Identity()

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.head_dropout / 2),
            nn.Linear(512, cfg.num_classes),
        )

        # Exposed for attention-map visualisation
        self.last_attention_weights: Optional[torch.Tensor] = None

    # ── freeze helpers ────────────────────────────────────────
    def freeze_backbone(self) -> None:
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.feature_extractor.parameters():
            p.requires_grad = True

    # ── forward ──────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        f = self.feature_extractor(x)            # (B, 2048, 7, 7)
        if self.cfg.use_se:
            f = self.se(f)                       # SE channel attention

        B, C, H, W = f.shape
        tokens = f.view(B, C, H * W).permute(0, 2, 1)   # (B, 49, 2048)

        if self.cfg.use_mha:
            # Self-attention with attention-weights kept for XAI
            attn_out, attn_weights = self.mha(
                tokens, tokens, tokens,
                need_weights=True, average_attn_weights=True,
            )
            self.last_attention_weights = attn_weights.detach()   # (B, 49, 49)
            tokens = self.norm(tokens + attn_out)               # residual + LN
        else:
            self.last_attention_weights = None

        pooled = self.gap(tokens.permute(0, 2, 1)).squeeze(-1)   # (B, 2048)
        return self.classifier(pooled)                            # (B, C)


# ─────────────────────────────────────────────────────────────
# BUILDER
# ─────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_model(cfg: ModelConfig,
                device: str = "cpu",
                pretrained: bool = True,
                freeze_backbone: bool = True
               ) -> HeartSoundModel:
    """Build, optionally freeze the backbone, and move to device."""
    model = HeartSoundModel(cfg, pretrained=pretrained)
    if freeze_backbone:
        model.freeze_backbone()
    model = model.to(device)

    trainable, total = count_parameters(model)
    print(f"  Backbone frozen: {freeze_backbone}")
    print(f"  Trainable params: {trainable:,} / Total: {total:,}")
    return model
