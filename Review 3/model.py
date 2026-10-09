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

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# MAIN MODEL
# ─────────────────────────────────────────────────────────────

class HeartSoundModel(nn.Module):
    """ResNet50 + optional SE + optional Multi-Head Attention + classifier head.

    Supports five variants:
      - 'baseline': ResNet50 -> GAP -> classifier head.
      - 'se': ResNet50 -> SEBlock -> GAP -> classifier head.
      - 'mha': ResNet50 -> MHA + residual + LayerNorm -> GAP -> classifier head.
      - 'se_mha': ResNet50 -> SEBlock -> MHA + residual + LayerNorm -> GAP -> classifier head (legacy architecture).
      - 'se_mha_dilated': same as se_mha but ResNet50 layer4 uses dilated convolutions (stride replaced by dilation),
        yielding a 2048x14x14 feature map (196 spatial tokens, ~0.57 s per cell) instead of 7x7.

    Modules not used by a variant are NOT created so parameter counts are exact.
    Module attribute names (feature_extractor, se, mha, norm, classifier) match the legacy checkpoint.
    """

    def __init__(self, cfg: ModelConfig, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg

        # Pretrained ResNet50 backbone, FC + avgpool stripped.
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        replace_dilation = [False, False, True] if cfg.variant == "se_mha_dilated" else None
        backbone = models.resnet50(
            weights=weights,
            replace_stride_with_dilation=replace_dilation,
        )

        # Children up to (but excluding) avgpool + fc
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        # Cached reference for Grad-CAM: layer4[-1] is the last Bottleneck block of ResNet50
        self.backbone_last_conv = self.feature_extractor[-1][-1]

        # Only create modules used by the specified variant
        if cfg.use_se:
            self.se = SEBlock(cfg.feature_dim, reduction=cfg.se_reduction)

        if cfg.use_mha:
            self.mha = nn.MultiheadAttention(
                embed_dim=cfg.feature_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.mha_dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(cfg.feature_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.head_dropout / 2),
            nn.Linear(512, cfg.num_classes),
        )

        # Exposed for attention-map visualisation (None when variant lacks MHA)
        self.last_attention_weights: Optional[torch.Tensor] = None

    # ── freeze helpers ────────────────────────────────────────
    def freeze_backbone(self) -> None:
        """Freeze feature extractor parameters for Phase A warm-up."""
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze feature extractor parameters for Phase B fine-tuning."""
        for p in self.feature_extractor.parameters():
            p.requires_grad = True

    # ── forward ──────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input batch of spectrograms of shape (B, 3, 224, 224).

        Returns:
            Logits of shape (B, num_classes).
        """
        f = self.feature_extractor(x)            # (B, 2048, 7, 7) or (B, 2048, 14, 14)

        if hasattr(self, "se"):
            f = self.se(f)                       # SE channel attention

        B, C, H, W = f.shape

        if hasattr(self, "mha"):
            tokens = f.view(B, C, H * W).permute(0, 2, 1)   # (B, num_tokens, 2048)
            attn_out, attn_weights = self.mha(
                tokens, tokens, tokens,
                need_weights=True, average_attn_weights=True,
            )
            self.last_attention_weights = attn_weights.detach()   # (B, num_tokens, num_tokens)
            tokens = self.norm(tokens + attn_out)               # residual + LN
            pooled = tokens.mean(dim=1)                          # GAP over tokens: (B, 2048)
        else:
            self.last_attention_weights = None
            pooled = f.mean(dim=(2, 3))                          # GAP over spatial grid: (B, 2048)

        return self.classifier(pooled)                            # (B, num_classes)


# ─────────────────────────────────────────────────────────────
# BUILDER & PARAMETERS
# ─────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count trainable and total parameters of a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_variant_param_details(model: HeartSoundModel) -> dict[str, int]:
    """Compute exact parameter counts per component, total, and Phase-A trainable."""
    backbone_params = sum(p.numel() for p in model.feature_extractor.parameters())
    se_params = sum(p.numel() for p in model.se.parameters()) if hasattr(model, "se") else 0
    mha_params = sum(p.numel() for p in model.mha.parameters()) if hasattr(model, "mha") else 0
    norm_params = sum(p.numel() for p in model.norm.parameters()) if hasattr(model, "norm") else 0
    clf_params = sum(p.numel() for p in model.classifier.parameters())
    total = backbone_params + se_params + mha_params + norm_params + clf_params
    phase_a_trainable = se_params + mha_params + norm_params + clf_params
    return {
        "feature_extractor": backbone_params,
        "se": se_params,
        "mha": mha_params,
        "norm": norm_params,
        "classifier": clf_params,
        "total": total,
        "phase_a_trainable": phase_a_trainable,
    }


def compute_all_variant_params() -> dict[str, dict[str, int]]:
    """Compute parameter counts for all five variants."""
    variants = ["baseline", "se", "mha", "se_mha", "se_mha_dilated"]
    out: dict[str, dict[str, int]] = {}
    for var in variants:
        cfg = ModelConfig(variant=var)
        model = HeartSoundModel(cfg, pretrained=False)
        out[var] = get_variant_param_details(model)
    return out


def export_param_counts(save_path: Path) -> dict[str, dict[str, int]]:
    """Export parameter counts for all variants to JSON."""
    import json
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    counts = compute_all_variant_params()
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    return counts


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
    print(f"  Variant: {cfg.variant}")
    print(f"  Backbone frozen: {freeze_backbone}")
    print(f"  Trainable params: {trainable:,} / Total: {total:,}")
    return model
