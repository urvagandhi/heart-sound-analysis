# Explainable AI for Heart Sound Analysis
### Leakage-Aware Benchmarking, Architectural Ablations and Population-Level Explanation Audits

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PhysioNet%20CinC%202016-green.svg)](https://physionet.org/content/challenge-2016/1.0.0/)
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)]()

> **Research Methodology and Seminar (RMS) Project**  
> **Department of Computer Science and Engineering, Nirma University, Ahmedabad**  
> **Authors:** Urva Gandhi (`23BCE078`) & Rakshit Gajnotar (`23BCE077`)  
> **Under the Guidance of:** Dr. Sapan Mankad  

---

## 📌 Executive Summary

Cardiovascular diseases (CVDs) remain the leading cause of global mortality, responsible for approximately 17.9 million deaths annually. While traditional cardiac auscultation using phonocardiograms (PCG) is the primary frontline screening technique, it is subjective, examiner-dependent and vulnerable to ambient noise.

Automated PCG classifiers have achieved high reported accuracy on public datasets, yet published numbers are frequently fragile because of unaddressed data leakage and cross-site distribution shifts. Furthermore, deep learning interpretability studies often showcase single cherry-picked saliency maps without quantitative population-level validation.

This repository provides an end-to-end framework implementing:
1. **Leakage-Aware Evaluation:** Quantifying diagnostic inflation from duplicate validation entries and conducting leave-one-database-out (LODO) cross-site testing.
2. **Component Ablations (3 Seeds):** Systematically isolating the contributions of ResNet50, Squeeze-and-Excitation (SE), and Multi-Head Self-Attention (MHSA).
3. **Dilated Backbone (14×14 Grid):** Replacing standard layer4 strides with atrous convolutions to refine spatial-temporal resolution to 0.57 s per cell.
4. **Population-Level Explanation Audit:** Quantitatively auditing frequency occlusion, perturbation faithfulness, cascading model randomization sanity checks, multi-method agreement, and PhysioNet acoustic state annotation enrichment ($E_s$).
5. **Model Calibration and Clinical Operating Point:** Evaluating Expected Calibration Error (ECE) and optimizing decision thresholds on validation data targeting sensitivity $\ge 0.90$.

---

## 🔬 System Specifications

### 1. Audio Preprocessing Pipeline
- **Sampling Rate:** Resampled to 4,000 Hz (providing bandwidth up to 2,000 Hz, well above the 1,000 Hz cardiac ceiling).
- **Filtering:** 4th-order zero-phase Butterworth bandpass filter (25 Hz to 900 Hz), suppressing baseline drift below 25 Hz and acoustic hiss above 900 Hz.
- **Duration Normalization:** Standardized to 8.0 s (32,000 samples) via zero-padding or onset cropping (capturing 8 to 13 resting cardiac cycles).
- **Time-Frequency Representation:** Log-mel spectrograms with 128 mel bands computed via STFT:
  - FFT window size ($n_{\text{fft}}$): 512 samples (128 ms, Hann window)
  - Hop length: 128 samples (32 ms)
  - Scale: Default librosa (Slaney) mel scale from 20 Hz to 1,000 Hz (linear below 1 kHz, 7.7 Hz per bin)
  - Output shape: $128 \times 251$, scaled to $[0, 1]$ in dB, bilinearly resized to $224 \times 224$, replicated across 3 channels, and normalized with ImageNet statistics.
- **Data Augmentation:** SpecAugment (frequency masking width 1 to 17 bands, time masking width 1 to 24 frames).

### 2. Training Protocol & Hyperparameters
- **Optimization:** AdamW optimizer with weight decay $10^{-4}$ and gradient clipping (max norm 1.0).
- **Loss Function:** Class-weighted cross-entropy with label smoothing ($\alpha = 0.05$).
- **Batch Size:** 32.
- **Two-Stage Schedule:**
  - *Phase A (Warm-up, 10 epochs):* Backbone weights frozen; training SE, MHA, and classification head at base learning rate $3\times 10^{-4}$ decaying to $10^{-6}$ via cosine annealing. Backbone BatchNorm running statistics are frozen (`freeze_bn=True`) to preserve ImageNet statistics.
  - *Phase B (Fine-tuning, 30 epochs):* Joint end-to-end training of all parameters at learning rate $3\times 10^{-5}$ decaying to $10^{-7}$. Early stopping counter is reset to 0 at the start of Phase B (patience 8 on validation loss).
- **Regularization:** Head dropout of 0.4 and 0.2; attention weight dropout of 0.1.

---

## 🏗️ Model Variants & Parameter Counts

| Model Variant | Key Components | Total Parameters | Phase A Trainable | Spatial Grid | Temporal Resolution |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `baseline` | ResNet50 Backbone | 23,508,034 | 2,050 | $7\times 7$ (49 tokens) | 1.14 s / cell |
| `se` | ResNet50 + SE Block ($r=16$) | 26,038,466 | 2,532,482 | $7\times 7$ (49 tokens) | 1.14 s / cell |
| `mha` | ResNet50 + MHSA (8 heads, $d_k=256$) | 39,341,506 | 15,835,522 | $7\times 7$ (49 tokens) | 1.14 s / cell |
| `se_mha` | ResNet50 + SE + MHSA (Reference [2]) | 41,871,938 | 18,365,954 | $7\times 7$ (49 tokens) | 1.14 s / cell |
| `se_mha_dilated` | Dilated ResNet50 Layer4 + SE + MHSA | 41,871,938 | 18,365,954 | $14\times 14$ (196 tokens) | **0.57 s / cell** |

---

## 🧪 Evaluation Protocols

1. **Protocol 1 — Legacy Split (Data Leakage Measurement):**
   - Evaluated on the 3,541 entries containing 301 duplicated validation entries.
   - Measures performance on 70 duplicate twin test rows versus 462 unique test rows.
2. **Protocol 2 — Deduplicated Benchmark (Component Ablation):**
   - Evaluated on 3,240 unique recordings across 3 random seeds (42, 43, and 44) under stratified 70/15/15 splits.
   - Measures multi-seed mean and sample standard deviation across all 5 model variants (15 runs).
3. **Protocol 3 — Leave-One-Database-Out (LODO Cross-Site Generalization):**
   - Cross-site evaluation holding out independent recording sites (`training-a`, `training-b`, `training-e`, `training-f`) for testing (8 runs).

---

## 🔍 Population-Level Explanation Audit Suite

1. **Frequency Band Occlusion:**
   - Evaluates reliance on physiological bands: B1 (20–150 Hz, S1/S2), B2 (150–500 Hz, murmurs), and B3 (500–1,000 Hz, high frequencies).
   - Compares performance drops against 20 contiguous random row-window controls ($\mu_{\text{rnd}} - 2\sigma_{\text{rnd}}$).
2. **Faithfulness Deletion and Insertion:**
   - Computes deletion and insertion curves across 0% to 50% feature perturbation.
   - Contrasts deletion AUC against random attribution baselines using paired Wilcoxon signed-rank tests.
3. **Adebayo Cascading Randomization Sanity Check:**
   - Progressively randomizes network weights top-down from classifier to early conv layers.
   - Verifies whether Spearman rank correlation with the intact Grad-CAM map collapses (< 0.30).
4. **Multi-Method Pairwise Agreement:**
   - Quantifies pairwise agreement between Grad-CAM, self-attention maps, and Gradient SHAP on pooled $7\times 7$ grids using Spearman rank correlation and top-20% IoU.
5. **PhysioNet State Annotation Enrichment ($E_s$):**
   - Evaluates temporal alignment against PhysioNet `*_StateAns.mat` labels (S1, systole, S2, diastole).
   - Computes enrichment ratio: $E_s = \frac{\sum_{t \in s} \text{CAM}(t) / \sum_t \text{CAM}(t)}{T_s / T}$, where $E_s > 1$ denotes focus exceeding chance.

---

## 🚀 Reproduction Guide (Google Colab & CLI)

The entire experimental benchmark consists of 24 planned runs executed sequentially without hyperparameter tuning loops.

### Step 1: Clone Repository & Install Dependencies
```bash
git clone https://github.com/urvagandhi/heart-sound-analysis.git
cd heart-sound-analysis

pip install -r requirements.txt
```

### Step 2: Generate Splits & Check Leakage
```bash
# Generate deduplicated splits (seeds 42, 43, 44) and LODO splits (holdouts a, b, e, f)
python "Review 3/make_splits.py" --mode dedup --seeds 42 43 44
python "Review 3/make_splits.py" --mode lodo --holdouts a b e f --seed 42

# Verify duplicates and compute legacy twin leakage
python "Review 3/verify_duplicates.py"
```

### Step 3: Run the 24 Benchmark Experiments Sequentially
```bash
# Verify the 24 planned runs in dry-run mode
python "Review 3/run_all.py" --dry-run

# Execute all 24 training/evaluation runs sequentially (resume-safe)
python "Review 3/run_all.py"
```

### Step 4: Run the Population-Level Explanation Audit
```bash
# Execute explanation audit suite on primary model (dedup_se_mha_seed42)
python "Review 3/audit/run_audit.py" --tag dedup_se_mha_seed42
```

### Step 5: Aggregate Deliverables & Generate LaTeX
```bash
# Compile all metrics into LaTeX tables, macros, and rule-based findings
python "Review 3/aggregate_results.py"
```

### Step 6: Run Fast Unit Test Suite
```bash
# Execute fast (<60 s) synthetic test suite
python -m pytest -v
```

---

## 📁 Repository Structure

```
.
├── README.md                                     # Project Documentation & Benchmark Guide
├── requirements.txt                              # Python Dependencies
├── .gitignore                                    # Build & cache exclusion rules
│
├── physionet_2016/                               # Dataset Manifests & Splits
│   ├── metadata.csv                              # Original 3,541 recording entries
│   ├── metadata_dedup.csv                        # Deduplicated 3,240 recording entries
│   ├── split_indices.json                        # Legacy 70/15/15 split indices
│   └── splits/                                   # Generated dedup & LODO split manifests
│
├── Review 2/                                     # RMS Review 2: Preprocessing Pipeline
│   ├── physionet_preprocess.py                   # Data ingestion, filter & spectrogram pipeline
│   └── main.tex                                  # Review 2 Beamer Presentation
│
├── Review 3/                                     # RMS Review 3: Core Implementation
│   ├── config.py                                 # Configuration dataclasses (5 variants, paths)
│   ├── model.py                                  # ResNet50, SE, MHA, and Dilated Backbone
│   ├── train.py                                  # Two-stage training engine (frozen BN, reset ES)
│   ├── evaluate.py                               # Evaluation engine & metric computation
│   ├── xai.py                                    # Grad-CAM, attention maps, and SHAP
│   ├── make_splits.py                            # Deduplicated & LODO split generator
│   ├── verify_duplicates.py                      # Leakage verifier & twin performance audit
│   ├── run_all.py                                # Sequential orchestrator (24 planned runs)
│   ├── aggregate_results.py                      # Deliverables aggregator -> Paper Writing/generated/
│   └── audit/                                    # Population-Level Explanation Audit Suite
│       ├── common.py                             # Mel-Hz mapping, unit conversion, bootstrap CI
│       ├── occlusion.py                          # Band occlusion with random-window controls
│       ├── faithfulness.py                       # Deletion/insertion curves & Wilcoxon tests
│       ├── sanity.py                             # Adebayo cascading randomization check
│       ├── agreement.py                          # Pairwise Spearman & IoU agreement
│       ├── enrichment.py                         # PhysioNet state annotation enrichment (E_s)
│       ├── calibration.py                        # ECE & clinical operating point selection
│       └── run_audit.py                          # Audit orchestration runner
│
├── Paper Writing/                                # Publication Manuscript
│   ├── paper.tex                                 # IEEEtran Conference Manuscript
│   └── generated/                                # Auto-generated LaTeX tables, macros & findings
│
└── tests/                                        # Synthetic Fast Unit Test Suite (pytest)
    ├── test_model.py                             # Model variants, legacy keys & parameter counts
    ├── test_splits.py                            # Disjointness & completeness of splits
    ├── test_occlusion.py                         # Toy band occlusion audit
    ├── test_faithfulness.py                      # Toy deletion AUC perturbation ordering
    ├── test_sanity.py                            # Cascading randomization correlation drop
    ├── test_enrichment.py                        # Acoustic state enrichment E_s = 1.0 test
    ├── test_calibration.py                       # ECE & operating point threshold selection
    └── test_aggregation.py                      # Results aggregator mock end-to-end test
```

---

## 📑 Citation & Research Paper

```bibtex
@article{gandhi2026xaiheartsound,
  title   = {Leakage-Aware Evaluation, Architectural Ablations and Population-Level Explanation Audits for Phonocardiogram Classification},
  author  = {Urva Gandhi and Rakshit Gajnotar and Dr. Sapan Mankad},
  journal = {Department of Computer Science and Engineering, Nirma University},
  year    = {2026}
}
```
