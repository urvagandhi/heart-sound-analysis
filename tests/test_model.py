"""Unit tests for model variants, parameter counts, and legacy key compatibility."""

import json
from pathlib import Path
import pytest
import torch

from config import ModelConfig
from model import HeartSoundModel, get_variant_param_details


def test_legacy_keys_compatibility() -> None:
    """Verify that se_mha state_dict keys exactly match the legacy checkpoint keys."""
    legacy_keys_path = Path(__file__).resolve().parent / "legacy_keys.json"
    assert legacy_keys_path.exists(), "tests/legacy_keys.json must exist"
    with open(legacy_keys_path, "r", encoding="utf-8") as f:
        expected_keys = json.load(f)

    cfg = ModelConfig(variant="se_mha")
    model = HeartSoundModel(cfg, pretrained=False)
    actual_keys = list(model.state_dict().keys())

    assert actual_keys == expected_keys, f"se_mha keys mismatch: expected {len(expected_keys)}, got {len(actual_keys)}"


def test_variant_parameter_counts_and_uncreated_modules() -> None:
    """Verify that unused modules are omitted and parameter counts are exact."""
    # Baseline
    cfg_base = ModelConfig(variant="baseline")
    m_base = HeartSoundModel(cfg_base, pretrained=False)
    assert not hasattr(m_base, "se")
    assert not hasattr(m_base, "mha")
    assert not hasattr(m_base, "norm")
    p_base = get_variant_param_details(m_base)
    assert p_base["se"] == 0
    assert p_base["mha"] == 0
    assert p_base["norm"] == 0

    # SE
    cfg_se = ModelConfig(variant="se")
    m_se = HeartSoundModel(cfg_se, pretrained=False)
    assert hasattr(m_se, "se")
    assert not hasattr(m_se, "mha")
    assert not hasattr(m_se, "norm")

    # MHA
    cfg_mha = ModelConfig(variant="mha")
    m_mha = HeartSoundModel(cfg_mha, pretrained=False)
    assert not hasattr(m_mha, "se")
    assert hasattr(m_mha, "mha")
    assert hasattr(m_mha, "norm")

    # SE+MHA vs SE+MHA Dilated
    cfg_semha = ModelConfig(variant="se_mha")
    m_semha = HeartSoundModel(cfg_semha, pretrained=False)
    cfg_dil = ModelConfig(variant="se_mha_dilated")
    m_dil = HeartSoundModel(cfg_dil, pretrained=False)

    p_semha = get_variant_param_details(m_semha)
    p_dil = get_variant_param_details(m_dil)
    assert p_semha["total"] == p_dil["total"]
    assert p_semha["phase_a_trainable"] == p_dil["phase_a_trainable"]


def test_dilated_grid_resolution() -> None:
    """Verify that se_mha_dilated produces a 14x14 grid (196 tokens) vs 7x7 (49 tokens)."""
    x = torch.randn(2, 3, 224, 224)

    m_standard = HeartSoundModel(ModelConfig(variant="se_mha"), pretrained=False)
    m_standard.eval()
    with torch.no_grad():
        out_std = m_standard(x)
    assert out_std.shape == (2, 2)
    assert m_standard.last_attention_weights is not None
    assert m_standard.last_attention_weights.shape == (2, 49, 49)

    m_dilated = HeartSoundModel(ModelConfig(variant="se_mha_dilated"), pretrained=False)
    m_dilated.eval()
    with torch.no_grad():
        out_dil = m_dilated(x)
    assert out_dil.shape == (2, 2)
    assert m_dilated.last_attention_weights is not None
    assert m_dilated.last_attention_weights.shape == (2, 196, 196)


@pytest.mark.parametrize("variant", ["baseline", "se", "mha", "se_mha", "se_mha_dilated"])
def test_forward_pass_output_shape(variant: str) -> None:
    """Verify forward pass output shape (1, 2) on dummy tensor (1, 3, 224, 224) for every variant."""
    cfg = ModelConfig(variant=variant)
    model = HeartSoundModel(cfg, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2), f"Expected shape (1, 2), got {out.shape} for {variant}"

