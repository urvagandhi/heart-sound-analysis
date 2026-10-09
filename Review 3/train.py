"""
Review 3 — Training
===================
Two-phase fine-tuning:

  Phase A (warm-up): backbone frozen, train SE + MHA + classifier head only
                     at ``lr_head`` for ``warmup_epochs``.
  Phase B (fine-tune): unfreeze the whole network, drop to ``lr_full``,
                       continue for the remaining epochs.

Class-imbalance handling uses cross-entropy with class weights (loaded from
class_weights.pt or computed on the fly) plus optional label smoothing.
"""

from __future__ import annotations

import json
import warnings
from contextlib import nullcontext

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Paths, ModelConfig, TrainConfig
from model import HeartSoundModel


# ─────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """Saves the model whenever the monitored metric improves."""

    def __init__(self, patience: int, min_delta: float, ckpt_path: Path,
                 mode: str = "min"):
        assert mode in ("min", "max")
        self.patience  = patience
        self.min_delta = min_delta
        self.ckpt_path = Path(ckpt_path)
        self.mode      = mode
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.counter   = 0
        self.best_epoch: Optional[int] = None

    def _is_better(self, value: float) -> bool:
        if self.mode == "min":
            return value < (self.best - self.min_delta)
        return value > (self.best + self.min_delta)

    def __call__(self, value: float, model: nn.Module, epoch: int) -> bool:
        """Returns True when early-stop should trigger."""
        if self._is_better(value):
            self.best       = value
            self.best_epoch = epoch
            self.counter    = 0
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.ckpt_path)
            return False
        self.counter += 1
        print(f"  EarlyStopping: no improvement {self.counter}/{self.patience}")
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────
# EPOCH LOOPS
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model: HeartSoundModel,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: str,
                    grad_clip: float,
                    scaler: Optional["torch.amp.GradScaler"] = None,
                    use_amp: bool = False,
                    freeze_bn: bool = False
                   ) -> Dict[str, float]:
    model.train()
    if freeze_bn:
        model.feature_extractor.eval()
    total_loss, correct, total = 0.0, 0, 0

    autocast_ctx = (torch.amp.autocast("cuda") if (use_amp and device == "cuda")
                    else nullcontext())

    for X, y in tqdm(loader, desc="  Train", leave=False):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            logits = model(X)
            loss   = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def validate(model: HeartSoundModel,
             loader: DataLoader,
             criterion: nn.Module,
             device: str
            ) -> Dict[str, float]:
    from sklearn.metrics import f1_score

    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)

        total_loss += loss.item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total   += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    return {"loss": total_loss / total, "acc": correct / total, "f1": f1}


# ─────────────────────────────────────────────────────────────
# DRIVER
# ─────────────────────────────────────────────────────────────

def _make_optimizer(model: HeartSoundModel, lr: float, weight_decay: float,
                    only_trainable: bool = False) -> AdamW:
    params = (filter(lambda p: p.requires_grad, model.parameters())
              if only_trainable else model.parameters())
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def train(model: HeartSoundModel,
          train_loader: DataLoader,
          val_loader: DataLoader,
          class_weights: torch.Tensor,
          paths: Paths,
          train_cfg: TrainConfig,
          device: str
         ) -> Dict[str, List[float]]:
    """End-to-end two-phase training. Saves best checkpoint to
    ``paths.checkpoint_dir / 'best_model.pth'`` and the training history
    JSON next to it."""
    import time
    from config import set_seed

    set_seed(train_cfg.seed)
    start_time = time.time()

    paths.ensure_output_dirs()
    ckpt_path = paths.checkpoint_dir / "best_model.pth"

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if train_cfg.use_class_weights else None,
        label_smoothing=train_cfg.label_smoothing,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "val_f1":     [], "lr":       [],
    }

    use_amp = train_cfg.use_amp and (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    stopper = EarlyStopping(
        patience  = train_cfg.early_stop_patience,
        min_delta = 1e-4,
        ckpt_path = ckpt_path,
        mode      = "min",
    )

    # ── Phase A: warm-up (backbone frozen) ────────────────────
    warmup = min(train_cfg.warmup_epochs, train_cfg.num_epochs)
    print(f"\n[Phase A] Warm-up: backbone frozen, {warmup} epochs, lr={train_cfg.lr_head:g}")
    optimizer = _make_optimizer(model, train_cfg.lr_head,
                                train_cfg.weight_decay, only_trainable=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=warmup, eta_min=1e-6)

    for epoch in range(warmup):
        tr = train_one_epoch(model, train_loader, optimizer, criterion,
                             device, train_cfg.grad_clip, scaler, use_amp,
                             freeze_bn=train_cfg.freeze_bn)
        va = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr["loss"]); history["val_loss"].append(va["loss"])
        history["train_acc"].append(tr["acc"]);   history["val_acc"].append(va["acc"])
        history["val_f1"].append(va["f1"]);       history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"  Epoch {epoch+1:02d}/{train_cfg.num_epochs} | "
              f"loss {tr['loss']:.4f}/{va['loss']:.4f} | "
              f"acc {tr['acc']:.3f}/{va['acc']:.3f} | "
              f"f1 {va['f1']:.3f}")
        stopper(va["loss"], model, epoch + 1)   # track best even in phase A

    # ── Phase B: full fine-tune ──────────────────────────────
    remaining = train_cfg.num_epochs - warmup
    if remaining > 0:
        print(f"\n[Phase B] Fine-tune: backbone unfrozen, {remaining} epochs, lr={train_cfg.lr_full:g}")
        model.unfreeze_backbone()
        model.feature_extractor.train()
        stopper.counter = 0  # Reset EarlyStopping counter so warm-up epochs do not carry over

        optimizer = _make_optimizer(model, train_cfg.lr_full,
                                    train_cfg.weight_decay, only_trainable=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-7)

        for epoch in range(warmup, train_cfg.num_epochs):
            tr = train_one_epoch(model, train_loader, optimizer, criterion,
                                 device, train_cfg.grad_clip, scaler, use_amp,
                                 freeze_bn=False)
            va = validate(model, val_loader, criterion, device)
            scheduler.step()

            history["train_loss"].append(tr["loss"]); history["val_loss"].append(va["loss"])
            history["train_acc"].append(tr["acc"]);   history["val_acc"].append(va["acc"])
            history["val_f1"].append(va["f1"]);       history["lr"].append(optimizer.param_groups[0]["lr"])

            print(f"  Epoch {epoch+1:02d}/{train_cfg.num_epochs} | "
                  f"loss {tr['loss']:.4f}/{va['loss']:.4f} | "
                  f"acc {tr['acc']:.3f}/{va['acc']:.3f} | "
                  f"f1 {va['f1']:.3f}")

            if stopper(va["loss"], model, epoch + 1):
                print(f"  Early stop at epoch {epoch+1}. "
                      f"Best epoch: {stopper.best_epoch}")
                break

    # Load best weights back in
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True))
    print(f"\n  Loaded best checkpoint (epoch {stopper.best_epoch}, "
          f"val_loss={stopper.best:.4f}) from {ckpt_path}")

    elapsed_seconds = float(time.time() - start_time)

    # Persist history
    hist_path = paths.checkpoint_dir / "history.json"
    with open(hist_path, "w") as f:
        json.dump({"history": history,
                   "best_epoch": stopper.best_epoch,
                   "best_val_loss": stopper.best,
                   "elapsed_seconds": elapsed_seconds,
                   "train_config": asdict(train_cfg)}, f, indent=2)

    return history


# ─────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_training_history(history: Dict[str, List[float]],
                          save_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train", color="#1F3A5F")
    axes[0].plot(history["val_loss"],   label="Val",   color="#E24B4A")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train acc", color="#1F3A5F")
    axes[1].plot(history["val_acc"],   label="Val acc",   color="#E24B4A")
    axes[1].plot(history["val_f1"],    label="Val macro F1",
                 color="#2E7D32", linestyle="--")
    axes[1].set_title("Accuracy & F1"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["lr"], color="#378ADD")
    axes[2].set_title("Learning rate"); axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log"); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")
