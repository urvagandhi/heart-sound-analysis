"""Adebayo cascading weight randomization sanity check for Grad-CAM explanations."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from model import HeartSoundModel
from xai import GradCAM


def reinit_module(module: nn.Module) -> None:
    """Re-initialize weights of a module using standard random initializations."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight)
            if m.out_proj.weight is not None:
                nn.init.xavier_uniform_(m.out_proj.weight)


def get_cascading_stages(model: HeartSoundModel) -> List[Tuple[str, List[nn.Module]]]:
    """Define top-down cascading stages, skipping modules not present in the variant."""
    stages = []

    # 1. Classifier
    stages.append(("classifier", [model.classifier]))

    # 2. norm + mha (if present)
    mha_mods = []
    if hasattr(model, "norm"):
        mha_mods.append(model.norm)
    if hasattr(model, "mha"):
        mha_mods.append(model.mha)
    if mha_mods:
        stages.append(("mha_norm", mha_mods))

    # 3. se (if present)
    if hasattr(model, "se"):
        stages.append(("se", [model.se]))

    # ResNet50 children in feature_extractor:
    # 0: conv1, 1: bn1, 2: relu, 3: maxpool, 4: layer1, 5: layer2, 6: layer3, 7: layer4
    fe = model.feature_extractor
    # 4. layer4
    stages.append(("layer4", [fe[7]]))

    # 5. layer3
    stages.append(("layer3", [fe[6]]))

    # 6. layer2
    stages.append(("layer2", [fe[5]]))

    # 7. layer1 + conv1
    early_layers = [fe[0], fe[1], fe[4]]  # conv1, bn1, layer1
    stages.append(("layer1_conv1", early_layers))

    return stages


def run_sanity_check(model: HeartSoundModel,
                     test_dataset: Any,
                     test_df: Any,
                     device: str,
                     n_samples: int = 100,
                     seed: int = 42) -> Dict[str, Any]:
    """Execute Adebayo cascading weight randomization test on stratified samples."""
    # Select 100 stratified test samples
    labels = test_df["label"].values.astype(int)
    rng = np.random.default_rng(seed)
    n_per = max(1, n_samples // 2)
    norm_idx = np.where(labels == 0)[0]
    abn_idx = np.where(labels == 1)[0]
    sel_norm = rng.choice(norm_idx, size=min(n_per, len(norm_idx)), replace=False)
    sel_abn = rng.choice(abn_idx, size=min(n_per, len(abn_idx)), replace=False)
    chosen_indices = np.concatenate([sel_norm, sel_abn])

    # Clone model so original checkpoint remains pristine
    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()

    # 1. Compute original Grad-CAM maps
    print(f"  [Sanity Check] Generating baseline Grad-CAM maps for {len(chosen_indices)} samples ...")
    orig_maps: List[np.ndarray] = []
    target_classes: List[int] = []

    for idx in chosen_indices:
        x_norm, _ = test_dataset[int(idx)]
        x_batch = x_norm.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model_copy(x_batch).argmax(dim=1).item())
        target_classes.append(pred)

        cam_ext = GradCAM(model_copy, model_copy.backbone_last_conv)
        cam = cam_ext.generate(x_batch, target_class=pred)
        cam_ext.remove()
        orig_maps.append(cam)

    # 2. Cascading randomization
    stages = get_cascading_stages(model_copy)
    stage_correlations: Dict[str, Dict[str, float]] = {}

    print(f"  [Sanity Check] Executing {len(stages)} cascading randomization stages ...")
    for stage_name, modules in stages:
        for m in modules:
            reinit_module(m)

        corrs: List[float] = []
        for i, idx in enumerate(chosen_indices):
            x_norm, _ = test_dataset[int(idx)]
            x_batch = x_norm.unsqueeze(0).to(device)
            pred = target_classes[i]

            cam_ext = GradCAM(model_copy, model_copy.backbone_last_conv)
            rand_cam = cam_ext.generate(x_batch, target_class=pred)
            cam_ext.remove()

            # Spearman correlation between original and randomized map
            r, _ = stats.spearmanr(orig_maps[i].flatten(), rand_cam.flatten())
            corrs.append(float(r) if not np.isnan(r) else 0.0)

        mean_corr = float(np.mean(corrs))
        sd_corr = float(np.std(corrs, ddof=1)) if len(corrs) > 1 else 0.0
        stage_correlations[stage_name] = {
            "mean_spearman": mean_corr,
            "sd_spearman": sd_corr,
        }
        print(f"    Stage '{stage_name:12s}': mean Spearman = {mean_corr:+.4f} +/- {sd_corr:.4f}")

    final_stage = stages[-1][0]
    final_spearman = stage_correlations[final_stage]["mean_spearman"]
    passed_check = bool(final_spearman < 0.30)

    return {
        "n_samples": len(chosen_indices),
        "stages": list(stage_correlations.keys()),
        "stage_correlations": stage_correlations,
        "final_stage_mean_spearman": final_spearman,
        "passed_sanity_check": passed_check,
    }
