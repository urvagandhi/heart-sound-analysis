"""
PhysioNet / CinC 2016 Heart Sound Dataset
Download + Preprocess + PyTorch Dataset
========================================
Run this in Google Colab (free T4 GPU) or locally.

Install dependencies first:
    pip install librosa scipy numpy pandas torch torchvision tqdm requests
"""

import os
import requests
import zipfile
import tarfile
import numpy as np
import pandas as pd
import librosa
import librosa.display
from scipy.signal import butter, filtfilt
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# STEP 1 — DOWNLOAD
# ─────────────────────────────────────────────
# The dataset has 6 training sub-sets (a–f) + validation
# Each sub-set has .wav files + a REFERENCE.csv with labels
# Labels: 1 = normal, -1 = abnormal

DATA_URL_BASE = "https://physionet.org/files/challenge-2016/1.0.0/"

# Sub-sets available
SUBSETS = ["training-a", "training-b", "training-c",
           "training-d", "training-e", "training-f",
           "validation"]

DATA_DIR = Path("physionet_2016")
RAW_DIR  = DATA_DIR / "raw"
SPEC_DIR = DATA_DIR / "spectrograms"  # processed output


def download_dataset(subsets=None, data_dir=RAW_DIR):
    """
    Download PhysioNet CinC 2016 sub-sets.
    If you're on Colab, this is the fastest approach.
    Alternatively, download manually from:
      https://physionet.org/content/challenge-2016/1.0.0/
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    if subsets is None:
        subsets = SUBSETS  # download all

    for subset in subsets:
        dest = data_dir / f"{subset}.zip"
        if dest.exists():
            print(f"  Already downloaded: {subset}")
            continue

        url = f"{DATA_URL_BASE}{subset}.zip"
        print(f"Downloading {subset} ...", end=" ", flush=True)

        r = requests.get(url, stream=True)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        print("done")

        # Unzip
        with zipfile.ZipFile(dest, "r") as z:
            z.extractall(data_dir)
        dest.unlink()  # remove zip after extraction


# ─────────────────────────────────────────────
# STEP 2 — LOAD LABELS
# ─────────────────────────────────────────────

def load_labels(data_dir=RAW_DIR):
    """
    Reads all REFERENCE.csv files and combines into one DataFrame.
    Columns: filename, label (0=normal, 1=abnormal), subset
    """
    rows = []
    for subset in SUBSETS:
        ref_path = data_dir / subset / "REFERENCE.csv"
        if not ref_path.exists():
            continue
        df = pd.read_csv(ref_path, header=None, names=["filename", "label"])
        df["subset"] = subset
        df["wav_path"] = df["filename"].apply(
            lambda f: data_dir / subset / f"{f}.wav"
        )
        rows.append(df)

    df = pd.concat(rows, ignore_index=True)

    # PhysioNet/CinC 2016 convention: raw labels are -1 = normal, +1 = abnormal
    # (Clifford et al., 2016 — verified against the per-subset distribution).
    # Remap to integer class indices: 0 = Normal, 1 = Abnormal.
    df["label"] = df["label"].map({-1: 0, 1: 1})   # 0=normal, 1=abnormal

    # For 4-class: you can split abnormal into subtypes if annotation exists
    # PhysioNet 2016 is natively binary — use this for 2-class
    # See Note below for 4-class strategy

    print(f"Total recordings: {len(df)}")
    print(df["label"].value_counts().rename({0: "Normal", 1: "Abnormal"}))
    return df


# ─────────────────────────────────────────────
# STEP 3 — AUDIO PREPROCESSING
# ─────────────────────────────────────────────

SR_TARGET  = 4000    # resample all audio to 4000 Hz (enough for heart sounds <1000 Hz)
DURATION   = 8.0     # fixed clip length in seconds (pad/trim to this)
N_FFT      = 512     # STFT window size
HOP_LENGTH = 128     # STFT hop
N_MELS     = 128     # number of Mel filterbanks → final spectrogram height
FMIN       = 20      # min frequency (Hz)
FMAX       = 1000    # max frequency (heart sounds live here)


def butter_bandpass(lowcut=25, highcut=900, fs=SR_TARGET, order=4):
    """4th-order Butterworth bandpass — removes noise outside heart sound range."""
    nyq = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def preprocess_audio(wav_path, sr_target=SR_TARGET, duration=DURATION):
    """
    Full preprocessing pipeline for one .wav file:
      1. Load + resample
      2. Bandpass filter (Butterworth)
      3. Normalize amplitude
      4. Pad or trim to fixed length
    Returns: numpy array of shape (sr_target * duration,)
    """
    # Load audio (librosa resamples on the fly)
    y, sr = librosa.load(str(wav_path), sr=sr_target, mono=True)

    # Bandpass filter
    b, a = butter_bandpass(fs=sr_target)
    y = filtfilt(b, a, y)

    # Normalize to [-1, 1]
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    # Pad or trim to fixed duration
    target_len = int(sr_target * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]

    return y, sr_target


def audio_to_logmel(y, sr=SR_TARGET):
    """
    Convert preprocessed waveform → Log-Mel Spectrogram.
    Output shape: (N_MELS, time_frames) = (128, ~250)
    This is the 2D "image" fed into ResNet.
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    # Convert to log scale (dB), matches human perception
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1] for CNN input
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

    return log_mel.astype(np.float32)   # shape: (128, time_frames)


def process_and_save(df, spec_dir=SPEC_DIR):
    """
    Process all recordings and save spectrograms as .npy files.
    Call this once — takes ~5–10 min on Colab.
    """
    spec_dir.mkdir(parents=True, exist_ok=True)
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        out_path = spec_dir / f"{row['filename']}.npy"
        if out_path.exists():
            continue  # already done

        try:
            y, sr = preprocess_audio(row["wav_path"])
            spec   = audio_to_logmel(y, sr)
            np.save(out_path, spec)
        except Exception as e:
            failed.append((row["filename"], str(e)))

    if failed:
        print(f"\nFailed: {len(failed)} files")
        for name, err in failed[:5]:
            print(f"  {name}: {err}")

    print(f"Saved {len(list(spec_dir.glob('*.npy')))} spectrograms to {spec_dir}/")


# ─────────────────────────────────────────────
# STEP 4 — PyTorch DATASET
# ─────────────────────────────────────────────

class HeartSoundDataset(Dataset):
    """
    PyTorch Dataset that loads pre-saved .npy spectrograms.
    Each spectrogram is resized to (224, 224) to match ResNet50 input.
    3-channel: replicate grayscale spectrogram across R, G, B channels.
    """

    IMG_SIZE = 224  # ResNet50 expects 224×224

    def __init__(self, df, spec_dir=SPEC_DIR, augment=False):
        self.df       = df.reset_index(drop=True)
        self.spec_dir = Path(spec_dir)
        self.augment  = augment

        # ImageNet normalization (since we fine-tune from ImageNet weights)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        spec_path = self.spec_dir / f"{row['filename']}.npy"

        # Load spectrogram (128, T)
        spec = np.load(spec_path)

        # Resize to 224×224 using interpolation
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, 128, T)
        spec_tensor = torch.nn.functional.interpolate(
            spec_tensor.unsqueeze(0),
            size=(self.IMG_SIZE, self.IMG_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)  # (1, 224, 224)

        # Repeat grayscale across 3 channels → (3, 224, 224)
        spec_tensor = spec_tensor.repeat(3, 1, 1)

        # ImageNet normalize
        spec_tensor = self.normalize(spec_tensor)

        # Data augmentation (training only)
        if self.augment:
            spec_tensor = self._augment(spec_tensor)

        label = torch.tensor(row["label"], dtype=torch.long)
        return spec_tensor, label

    def _augment(self, x):
        """Simple time/frequency masking (SpecAugment-style)."""
        # Frequency masking: zero out random frequency band
        f_start = np.random.randint(0, self.IMG_SIZE - 20)
        f_width = np.random.randint(5, 20)
        x[:, f_start:f_start + f_width, :] = 0

        # Time masking: zero out random time segment
        t_start = np.random.randint(0, self.IMG_SIZE - 30)
        t_width = np.random.randint(10, 30)
        x[:, :, t_start:t_start + t_width] = 0

        return x


def get_dataloaders(df, spec_dir=SPEC_DIR, batch_size=32,
                    val_split=0.15, test_split=0.15, seed=42):
    """
    Split into train / val / test and return DataLoaders.
    Stratified split to keep class balance.
    """
    from sklearn.model_selection import train_test_split

    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=val_split + test_split,
        stratify=df["label"], random_state=seed
    )
    # Second split: val vs test
    relative_test = test_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["label"], random_state=seed
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = HeartSoundDataset(train_df, spec_dir, augment=True)
    val_ds   = HeartSoundDataset(val_df,   spec_dir, augment=False)
    test_ds  = HeartSoundDataset(test_df,  spec_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# STEP 5 — QUICK VISUALISATION CHECK
# ─────────────────────────────────────────────

def visualise_samples(df, spec_dir=SPEC_DIR, n=4):
    """Plot spectrograms for a few samples — sanity check."""
    fig, axes = plt.subplots(1, n, figsize=(14, 3))
    sample = df.sample(n, random_state=1)

    for ax, (_, row) in zip(axes, sample.iterrows()):
        spec = np.load(spec_dir / f"{row['filename']}.npy")
        label_str = "Normal" if row["label"] == 0 else "Abnormal"
        librosa.display.specshow(
            spec, sr=SR_TARGET, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="mel", ax=ax, fmin=FMIN, fmax=FMAX
        )
        ax.set_title(f"{row['filename']}\n{label_str}", fontsize=9)
        ax.set_xlabel("")

    plt.suptitle("Log-Mel Spectrograms — Sample Check", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig("sample_spectrograms.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: sample_spectrograms.png")


# ─────────────────────────────────────────────
# MAIN — RUN THIS END-TO-END
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Download (comment out if you already have the data)
    print("=== Step 1: Download ===")
    download_dataset(subsets=["training-a", "training-b",
                               "training-c", "training-d",
                               "training-e", "training-f"])

    # 2. Load labels
    print("\n=== Step 2: Load Labels ===")
    df = load_labels()

    # 3. Process all audio → spectrograms (run once, ~5–10 min)
    print("\n=== Step 3: Process Audio ===")
    process_and_save(df)

    # 4. Visualise a few to sanity check
    print("\n=== Step 4: Visualise Samples ===")
    visualise_samples(df)

    # 5. Build PyTorch DataLoaders
    print("\n=== Step 5: DataLoaders ===")
    train_loader, val_loader, test_loader = get_dataloaders(df)

    # 6. Check one batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch shape : {X_batch.shape}")   # (32, 3, 224, 224)
    print(f"Labels      : {y_batch[:8]}")
    print("\nReady for model training!")


# ─────────────────────────────────────────────
# NOTE: 4-CLASS STRATEGY
# ─────────────────────────────────────────────
# PhysioNet 2016 is natively binary. For 4 classes you have 2 options:
#
# Option A) Use CirCor DigiScope 2022 instead — natively has 3-4 label types.
#   URL: https://physionet.org/content/circor-heart-sound/1.0.3/
#   Labels: murmur_present, murmur_absent, murmur_unknown
#
# Option B) Stay binary but add a second dimension via segmentation:
#   Classify whether the abnormality is in the systolic or diastolic phase
#   by analyzing which segment (S1→S2 vs S2→S1) triggers the highest Grad-CAM activation.
#   This lets you report 4 "effective" categories post-hoc without changing the dataset.
#
# For Review 2 minimum viable: go binary first, get it working, then expand.
