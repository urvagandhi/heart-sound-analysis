"""
Review 3 — Explainable AI
=========================
Three complementary explanations, all reported in the cited literature on
heart-sound classification:

1. Grad-CAM         — spatial saliency on the spectrogram, hooked on the last
                      ResNet50 bottleneck (Selvaraju et al., ICCV 2017).
2. SHAP             — feature attributions via GradientExplainer (robust
                      with custom architectures like the SE+MHA head).
3. Attention map    — the multi-head self-attention weights captured during
                      forward pass, averaged over heads, reshaped to 7×7.

All visualisations are saved as PNG and overlaid on the original spectrogram
so a clinician can read time × frequency directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from config import AudioConfig, CLASS_NAMES, ModelConfig
from data import HeartSoundDataset
from model import HeartSoundModel


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation and return a single-channel HxW image
    in [0, 1] for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor.detach().cpu() * std + mean
    img  = img.clamp(0, 1)[0].numpy()          # all channels equal — take ch 0
    return img


def _heatmap_overlay(image: np.ndarray, heatmap: np.ndarray,
                     alpha: float = 0.5) -> np.ndarray:
    """Blend a 2D image (H,W) with a 2D heatmap (H,W) using the jet colormap."""
    import matplotlib.pyplot as plt
    rgb_img = np.stack([image] * 3, axis=-1)
    hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    hm_rgb  = plt.cm.jet(hm_norm)[:, :, :3]
    return alpha * rgb_img + (1 - alpha) * hm_rgb


def _mel_frequencies(n_mels: int = 128, fmin: float = 20.0, fmax: float = 1000.0) -> np.ndarray:
    """Return center frequencies for n_mels mel filterbanks.
    Uses librosa if available; falls back to pure numpy implementation matching librosa.
    """
    try:
        import librosa
        return librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    except ImportError:
        f_min, f_sp = 0.0, 200.0 / 3
        min_log_hz, min_log_mel = 1000.0, (1000.0 - 0.0) / (200.0 / 3)
        logstep = np.log(6.4) / 27.0

        def hz_to_mel(f):
            f = np.asanyarray(f)
            m = (f - f_min) / f_sp
            log_t = f >= min_log_hz
            if np.any(log_t):
                m[log_t] = min_log_mel + np.log(f[log_t] / min_log_hz) / logstep
            return m

        def mel_to_hz(m):
            m = np.asanyarray(m)
            f = f_min + f_sp * m
            log_t = m >= min_log_mel
            if np.any(log_t):
                f[log_t] = min_log_hz * np.exp(logstep * (m[log_t] - min_log_mel))
            return f

        mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels)
        return mel_to_hz(mels)


def _setup_mel_axis(ax, audio_cfg: AudioConfig, img_h: int = 224,
                    ticks_hz: Sequence[int] = (100, 200, 400, 600, 800, 1000)):
    """Configure y-axis ticks and labels on a resized mel spectrogram
    so ticks reflect the non-linear mel filterbank center frequencies."""
    mf = _mel_frequencies(n_mels=audio_cfg.n_mels, fmin=audio_cfg.fmin, fmax=audio_cfg.fmax)
    rows = np.interp(ticks_hz, mf, np.arange(audio_cfg.n_mels)) / (audio_cfg.n_mels - 1) * (img_h - 1)
    ax.set_yticks(rows)
    ax.set_yticklabels(ticks_hz)


# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """Minimal Grad-CAM implementation that hooks any 2D conv layer.

    We avoid the external ``pytorch-grad-cam`` dependency so the module is
    self-contained, but the algorithm is identical to Selvaraju et al.
    """

    def __init__(self, model: HeartSoundModel, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor]   = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_act)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _module, _inp, out):
        self.activations = out.detach()

    def _save_grad(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        was_training = self.model.training
        self.model.eval()

        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        # (1, C, h, w)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global-avg
        cam = (weights * self.activations).sum(dim=1).squeeze(0) # (h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        if was_training:
            self.model.train()
        return cam


def visualise_gradcam(model: HeartSoundModel,
                      dataset: HeartSoundDataset,
                      indices: Sequence[int],
                      device: str,
                      audio_cfg: AudioConfig,
                      class_names: List[str],
                      save_path: Path) -> None:
    """Render Grad-CAM heatmaps for the supplied dataset indices."""
    import matplotlib.pyplot as plt

    cam_engine = GradCAM(model, model.backbone_last_conv)
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.6 * n))
    if n == 1:
        axes = axes[None, :]

    try:
        for row, idx in enumerate(indices):
            spec_t, true_label = dataset[idx]
            spec_t_batch = spec_t.unsqueeze(0).to(device)

            # Prediction
            with torch.no_grad():
                logits = model(spec_t_batch)
                pred_class = int(logits.argmax(1).item())
                confidence = float(F.softmax(logits, dim=1)[0, pred_class])

            cam = cam_engine(spec_t_batch, target_class=pred_class)
            img = _denormalise(spec_t)

            true_str = class_names[int(true_label)]
            pred_str = class_names[pred_class]
            correct  = (pred_class == int(true_label))

            img_h = img.shape[0]
            extent = [0, audio_cfg.duration, 0, img_h - 1]

            axes[row, 0].imshow(img, cmap="magma", origin="lower",
                                aspect="auto", extent=extent)
            _setup_mel_axis(axes[row, 0], audio_cfg, img_h=img_h)
            axes[row, 0].set_title(f"Spectrogram\nTrue: {true_str}", fontsize=10)
            axes[row, 0].set_xlabel("Time (s)"); axes[row, 0].set_ylabel("Mel freq (Hz)")

            axes[row, 1].imshow(cam, cmap="jet", origin="lower", aspect="auto",
                                extent=extent)
            _setup_mel_axis(axes[row, 1], audio_cfg, img_h=img_h)
            axes[row, 1].set_title("Grad-CAM (warm = high)", fontsize=10)
            axes[row, 1].set_xlabel("Time (s)")

            axes[row, 2].imshow(_heatmap_overlay(img, cam), origin="lower",
                                aspect="auto", extent=extent)
            _setup_mel_axis(axes[row, 2], audio_cfg, img_h=img_h)
            colour = "green" if correct else "red"
            axes[row, 2].set_title(
                f"Overlay\nPred: {pred_str} ({confidence:.1%})",
                fontsize=10, color=colour,
            )
            axes[row, 2].set_xlabel("Time (s)")
    finally:
        cam_engine.close()

    plt.suptitle("Grad-CAM — spectrogram regions driving the prediction",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# ATTENTION VISUALISATION
# ─────────────────────────────────────────────────────────────

def visualise_attention(model: HeartSoundModel,
                        dataset: HeartSoundDataset,
                        indices: Sequence[int],
                        device: str,
                        audio_cfg: AudioConfig,
                        class_names: List[str],
                        save_path: Path) -> None:
    """Average MHA weights across heads and query tokens to obtain a per-key
    attention map; reshape 49 → 7×7 → 224×224 and overlay on the spectrogram.
    """
    import matplotlib.pyplot as plt

    model.eval()
    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.6 * n))
    if n == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        spec_t, true_label = dataset[idx]
        spec_t_batch = spec_t.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(spec_t_batch)
            pred_class = int(logits.argmax(1).item())

        # (B, Q, K) after average_attn_weights=True
        attn = model.last_attention_weights[0].cpu()            # (Q, K)
        per_key = attn.mean(dim=0)                              # (K,) = (49,)
        side = int(np.sqrt(per_key.numel()))
        attn_map = per_key.view(side, side).numpy()

        # Upsample 7×7 to 224×224
        attn_t  = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0)
        attn_up = F.interpolate(attn_t, size=spec_t.shape[-2:],
                                mode="bilinear", align_corners=False)
        attn_up = attn_up.squeeze().numpy()
        attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)

        img = _denormalise(spec_t)
        true_str = class_names[int(true_label)]
        pred_str = class_names[pred_class]

        img_h = img.shape[0]
        extent = [0, audio_cfg.duration, 0, img_h - 1]

        axes[row, 0].imshow(img, cmap="magma", origin="lower", aspect="auto",
                            extent=extent)
        _setup_mel_axis(axes[row, 0], audio_cfg, img_h=img_h)
        axes[row, 0].set_title(f"Spectrogram — True: {true_str}", fontsize=10)
        axes[row, 0].set_xlabel("Time (s)"); axes[row, 0].set_ylabel("Mel freq (Hz)")

        axes[row, 1].imshow(_heatmap_overlay(img, attn_up), origin="lower",
                            aspect="auto", extent=extent)
        _setup_mel_axis(axes[row, 1], audio_cfg, img_h=img_h)
        axes[row, 1].set_title(f"MHA attention — Pred: {pred_str}", fontsize=10)
        axes[row, 1].set_xlabel("Time (s)")

    plt.suptitle("Multi-Head Attention — average spatial focus", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────

def compute_shap_values(model: HeartSoundModel,
                        background_loader,
                        explain_loader,
                        device: str,
                        n_background: int = 32,
                        n_explain: int = 8):
    """Use SHAP's GradientExplainer — robust with custom heads (MHA), unlike
    DeepExplainer.  Returns: (shap_values, inputs, labels).
    """
    import shap

    bg, used = [], 0
    for X, _ in background_loader:
        bg.append(X)
        used += len(X)
        if used >= n_background:
            break
    background = torch.cat(bg)[:n_background].to(device)

    inp, lab, used = [], [], 0
    for X, y in explain_loader:
        inp.append(X); lab.append(y)
        used += len(X)
        if used >= n_explain:
            break
    inputs = torch.cat(inp)[:n_explain].to(device)
    labels = torch.cat(lab)[:n_explain]

    model.eval()
    explainer  = shap.GradientExplainer(model, background)
    raw = explainer.shap_values(inputs)

    # Normalise to a list of per-class arrays of shape (N, C_in, H, W).
    # SHAP < 0.42 returns ``list[class]`` of (N, C_in, H, W).
    # SHAP >= 0.42 returns a single ``ndarray`` of shape
    #     (N, C_in, H, W, n_classes)  ← class axis is last
    # — convert the new format back to a list so callers behave identically.
    if isinstance(raw, list):
        shap_values = raw
    else:
        arr = np.asarray(raw)
        if arr.ndim == 5:
            shap_values = [arr[..., c] for c in range(arr.shape[-1])]
        else:
            shap_values = [arr]
    return shap_values, inputs.detach().cpu(), labels


def plot_shap_frequency_importance(shap_values,
                                   audio_cfg: AudioConfig,
                                   class_names: List[str],
                                   save_path: Path) -> None:
    """Aggregate SHAP attributions into a mean-|importance| profile per Mel
    frequency band, annotated with cardiac-frequency regions of interest."""
    import matplotlib.pyplot as plt

    # Normalise to a list of per-class (N, C_in, H, W) arrays.
    # SHAP >= 0.42 may pass one ndarray (N, C_in, H, W, n_classes).
    if isinstance(shap_values, list):
        pass
    else:
        arr = np.asarray(shap_values)
        shap_values = ([arr[..., c] for c in range(arr.shape[-1])]
                       if arr.ndim == 5 else [arr])

    n_classes = len(shap_values)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    img_size = shap_values[0].shape[-2]  # height H = 224
    mf = _mel_frequencies(n_mels=audio_cfg.n_mels, fmin=audio_cfg.fmin, fmax=audio_cfg.fmax)
    ticks_hz = [100, 200, 400, 600, 800, 1000]
    rows = np.interp(ticks_hz, mf, np.arange(audio_cfg.n_mels)) / (audio_cfg.n_mels - 1) * (img_size - 1)
    y_axis = np.arange(img_size)

    for c, ax in enumerate(axes):
        sv = np.asarray(shap_values[c])           # (N, 3, H, W)
        # Mean |SHAP| across samples and channels, then mean across time
        importance = np.abs(sv).mean(axis=(0, 1)).mean(axis=1)    # (H,)

        ax.plot(importance, y_axis, color="#1F3A5F", linewidth=2)
        ax.fill_betweenx(y_axis, 0, importance, color="#378ADD", alpha=0.3)

        # Cardiac frequency bands mapped to mel row positions
        for lo, hi, colour, label in [
            (25,  150, "#E24B4A", "S1/S2 (25–150 Hz)"),
            (150, 500, "#EF9F27", "Murmur (150–500 Hz)"),
            (500, 900, "#6CA0DC", "High freq (500–900 Hz)"),
        ]:
            y_lo = float(np.interp(lo, mf, np.arange(audio_cfg.n_mels)) / (audio_cfg.n_mels - 1) * (img_size - 1))
            y_hi = float(np.interp(hi, mf, np.arange(audio_cfg.n_mels)) / (audio_cfg.n_mels - 1) * (img_size - 1))
            ax.axhspan(y_lo, y_hi, alpha=0.10, color=colour, label=label)

        ax.set_ylim(0, img_size - 1)
        ax.set_yticks(rows)
        ax.set_yticklabels(ticks_hz)
        ax.set_title(f"SHAP frequency importance — {class_names[c]}", fontsize=11)
        ax.set_xlabel("Mean |SHAP|"); ax.set_ylabel("Mel frequency (Hz)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    plt.suptitle("SHAP — Where the model looks across the frequency axis",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


def plot_shap_sample_overlays(shap_values, inputs: torch.Tensor,
                              labels: torch.Tensor,
                              audio_cfg: AudioConfig,
                              class_names: List[str],
                              save_path: Path,
                              n_show: int = 4) -> None:
    """For each of the first ``n_show`` samples, plot the spectrogram side
    by side with the SHAP attribution map for the predicted class."""
    import matplotlib.pyplot as plt

    if isinstance(shap_values, list):
        pass
    else:
        arr = np.asarray(shap_values)
        shap_values = ([arr[..., c] for c in range(arr.shape[-1])]
                       if arr.ndim == 5 else [arr])

    n_show = min(n_show, inputs.shape[0])
    fig, axes = plt.subplots(n_show, 2, figsize=(10, 3.5 * n_show))
    if n_show == 1:
        axes = axes[None, :]

    for i in range(n_show):
        true_lab = int(labels[i].item())
        # Use the SHAP map of the TRUE class so the interpretation matches
        # the ground-truth biology (irrespective of model error).
        sv = np.asarray(shap_values[true_lab])[i]   # (3, H, W)
        attribution = np.abs(sv).mean(axis=0)         # (H, W)

        img = _denormalise(inputs[i])
        img_h = img.shape[0]
        extent = [0, audio_cfg.duration, 0, img_h - 1]

        axes[i, 0].imshow(img, cmap="magma", origin="lower", aspect="auto",
                          extent=extent)
        _setup_mel_axis(axes[i, 0], audio_cfg, img_h=img_h)
        axes[i, 0].set_title(f"Spectrogram — {class_names[true_lab]}", fontsize=10)
        axes[i, 0].set_xlabel("Time (s)"); axes[i, 0].set_ylabel("Mel freq (Hz)")

        axes[i, 1].imshow(_heatmap_overlay(img, attribution), origin="lower",
                          aspect="auto", extent=extent)
        _setup_mel_axis(axes[i, 1], audio_cfg, img_h=img_h)
        axes[i, 1].set_title("SHAP overlay (|attribution|)", fontsize=10)
        axes[i, 1].set_xlabel("Time (s)")

    plt.suptitle("SHAP — per-sample spectrogram attributions", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")
