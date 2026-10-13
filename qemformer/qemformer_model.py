"""
qemformer_model.py
==================
QEMFormer — Transformer-Based Quantum Error Mitigation Model.

Inspired by "QEM-Bench: Benchmarking Learning-based Quantum Error Mitigation
and QEMFormer as a Multi-ranged Context Learning Baseline" (ICML 2025, Bao et al.)

Architecture (Two-Branch Fusion):
    ┌─────────────────────────────────────────────────────────┐
    │  Input Features (25 raw → 26 augmented after pruning)   │
    │  (survival_factor removed; T1/T2 retained; no ZNE       │
    │   shortcut features — Transformer must learn them)      │
    ├─────────────────────────┬───────────────────────────────┤
    │  Static Features (21)   │  ZNE Sequence (5 values)      │
    │         │                │         │                     │
    │   MLP Branch             │  Transformer Branch           │
    │  ResBlock(512)           │  LearnablePosEmbed(λ)         │
    │  ResBlock(512)           │  4× TransformerEncoderLayer   │
    │  ResBlock(256)           │  Attention Pooling             │
    │         │                │         │                     │
    │         └────────┬───────┘         │                     │
    │                  │                 │                     │
    │          Fusion (Concat)           │                     │
    │          ResBlock(512)                                    │
    │          ResBlock(256)                                    │
    │          Linear(1)  → mitigated expectation value         │
    └─────────────────────────────────────────────────────────┘

Features:
    - Identical data pipeline to mlp_mitiq.py (same feature engineering,
      survival filtering, train/val/test splits) for fair comparison
    - Physics-informed loss (MSE + boundary penalty)
    - Tabular feature dropout regularization
    - AMP + OneCycleLR + early stopping
    - Comprehensive visualization suite (6 plots)

Dataset:
    zne/dataset/vector_dataset.npz
    X : float32 (500000, 25) — circuit metadata + noise params + ZNE exp values
    y : float32 (500000,)    — ideal expectation value

Usage:
    python qemformer/qemformer_model.py
    python qemformer/qemformer_model.py --epochs 150 --lr 5e-4 --batch 512

Outputs (saved to qemformer/outputs/):
    best_qemformer_model.pth    — best checkpoint (lowest val loss)
    final_qemformer_model.pth   — model after all epochs
    training_curves.png         — loss, MSE, MAE, R² curves
    predictions_scatter.png     — predicted vs ideal scatter
    predictions_residuals_kde.png — error distribution
    error_vs_survival.png       — residual error vs noise difficulty
    error_by_scale_factor.png   — correction quality per ZNE scale
    feature_importance.png      — permutation importance
    training_log.csv            — per-epoch metrics
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR    = Path(__file__).resolve().parent.parent          # project root
DATASET_NPZ = ROOT_DIR / "zne" / "dataset" / "vector_dataset.npz"
OUTPUT_DIR  = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS      = 200    # expanded budget for dual-branch convergence (Fix 5)
DEFAULT_LR          = 5e-4
DEFAULT_BATCH       = 8192
DEFAULT_SEED        = 42
DEFAULT_VAL_SPLIT   = 0.20
DEFAULT_TEST_SPLIT  = 0.10
DEFAULT_LAMBDA_BNDRY= 0.10
DEFAULT_DROPOUT     = 0.05
DEFAULT_WEIGHT_DECAY= 1e-4

# Transformer-specific
DEFAULT_D_MODEL     = 128     # Transformer internal dimension
DEFAULT_NHEAD       = 8       # Number of attention heads
DEFAULT_NUM_LAYERS  = 8       # Transformer encoder layers
DEFAULT_DIM_FF      = 256     # Feed-forward inner dimension

# Minimum noise-survival factor for sample inclusion
MIN_SURVIVAL = 0.1

# ---------------------------------------------------------------------------
# Feature column indices — vector_dataset.npz (31 columns)
# ---------------------------------------------------------------------------
# 0:num_qubits  1:depth  2:idle_frac  3:obs_weight  4:sample_shots
# 5:error_1q  6:error_2q
# 7:T1_us  8:T2_us  9:time_1q_ns  10:time_2q_ns
# 11:p_damp_1q  12:p_deph_1q  13:p_damp_2q  14:p_deph_2q
# 15:p_leak  16:p_01  17:p_10  18:p_xtalk  19:beta_nm  20:P_th  21:zeta_zz  22:tls_drop
# 23:theta_error  24:observable_alignment  25:survival_factor
# 26:exp_val_scale_1  27:exp_val_scale_3  28:exp_val_scale_5
# 29:exp_val_scale_7  30:exp_val_scale_9

IDX_NQUBITS      = 0
IDX_DEPTH        = 1
IDX_IDLE_FRAC    = 2
IDX_OBS_WEIGHT   = 3
IDX_SAMPLE_SHOTS = 4
IDX_ERROR_1Q     = 5
IDX_ERROR_2Q     = 6
IDX_T1           = 7
IDX_T2           = 8
IDX_TIME_1Q      = 9
IDX_TIME_2Q      = 10
IDX_P_DAMP_1Q    = 11
IDX_P_DEPH_1Q    = 12
IDX_P_DAMP_2Q    = 13
IDX_P_DEPH_2Q    = 14
IDX_P_LEAK       = 15
IDX_P_01         = 16
IDX_P_10         = 17
IDX_P_XTALK      = 18
IDX_BETA_NM      = 19
IDX_P_TH         = 20
IDX_ZETA_ZZ      = 21
IDX_TLS_DROP     = 22
IDX_THETA_ERROR  = 23
IDX_OBS_ALIGN    = 24
IDX_SURVIVAL     = 25

# ZNE exp-value columns: indices 26-30
NOISY_EXP_INDICES = [26, 27, 28, 29, 30]

# Scale factors corresponding to the ZNE columns
SCALE_FACTORS = [1, 3, 5, 7, 9]


# ===========================================================================
# Data Loading & Preprocessing (mirrors mlp_mitiq.py for consistency)
# ===========================================================================

class TabularFeatureDropout(torch.utils.data.Dataset):
    """
    Tabular feature dropout dataset for regularization.
    Accepts a dropout_schedule: dict mapping column index tuples to dropout probs.
    Training samples have feature groups randomly zeroed; val/test receive full data.

    Fix 1 — The λ=1 unmitigated baseline is stored separately and is immune to
    feature dropout so residual predictions can always be reconstructed back into
    ideal-value space during training and evaluation.
    """
    def __init__(self, X, y, baseline, is_train, dropout_schedule=None):
        self.X = X
        self.y = y
        self.baseline = baseline  # (N,) — λ=1 unmitigated value, dropout-immune
        self.is_train = is_train
        self.dropout_schedule = []
        if dropout_schedule and is_train:
            for cols, prob in dropout_schedule.items():
                col_t = torch.tensor(list(cols), dtype=torch.long, device=X.device)
                self.dropout_schedule.append((col_t, prob))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        baseline = self.baseline[idx]
        if self.dropout_schedule:
            cloned = False
            for col_indices, prob in self.dropout_schedule:
                if torch.rand(1).item() < prob:
                    if not cloned:
                        x = x.clone()
                        cloned = True
                    x[col_indices] = 0.0
        return x, y, baseline


class GPUDataLoader:
    """
    Zero-copy GPU-native batch iterator with vectorized feature dropout.

    Fix 1 — Yields (X_batch, y_residual_batch, baseline_batch) 3-tuples.
    The baseline (λ=1 unmitigated value) bypasses feature dropout so residual
    predictions can always be reconstructed into ideal-value space for metrics.
    """
    def __init__(self, dataset, batch_size, shuffle):
        self.X = dataset.X
        self.y = dataset.y
        self.baseline = dataset.baseline
        self.is_train = dataset.is_train
        self.dropout_schedule = dataset.dropout_schedule
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(dataset)
        self.dataset = dataset

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        device = self.X.device
        idx = (torch.randperm(self.n, device=device) if self.shuffle
               else torch.arange(self.n, device=device))

        for start in range(0, self.n, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            X_batch       = self.X[batch_idx]
            y_batch       = self.y[batch_idx]
            baseline_batch = self.baseline[batch_idx]

            if self.is_train and self.dropout_schedule:
                cloned = False
                for col_indices, prob in self.dropout_schedule:
                    drop_mask = torch.rand(len(batch_idx), device=device) < prob
                    if drop_mask.any():
                        if not cloned:
                            X_batch = X_batch.clone()
                            cloned = True
                        for c in col_indices:
                            X_batch[drop_mask, c] = 0.0

            yield X_batch, y_batch, baseline_batch


def load_and_split(
    npz_path,
    val_split=DEFAULT_VAL_SPLIT,
    test_split=DEFAULT_TEST_SPLIT,
    seed=DEFAULT_SEED,
    min_survival=MIN_SURVIVAL,
    device=None,
):
    """
    Load dataset, engineer derived physics features, filter irrecoverable
    samples, scale, and split into train/val/test.

    Mirrors the exact same pipeline as mlp_mitiq.py for fair comparison.

    Returns: train_ds, val_ds, test_ds, scaler, feature_names, zne_col_indices
    """
    data = np.load(npz_path, allow_pickle=True)
    X_raw = data["X"].astype(np.float32)   # (N, 25)
    y_raw = data["y"].astype(np.float32)   # (N,)
    feature_names = data["feature_names"].tolist()

    print(f"  Raw dataset: X={X_raw.shape}, y={y_raw.shape}")
    print(f"  Features: {feature_names}")

    # ------------------------------------------------------------------
    # Derived features (same physics as mlp_mitiq.py)
    # ------------------------------------------------------------------
    nq        = X_raw[:, IDX_NQUBITS   ].astype(np.float64)
    d         = X_raw[:, IDX_DEPTH     ].astype(np.float64)
    e1q       = X_raw[:, IDX_ERROR_1Q  ].astype(np.float64)
    e2q       = X_raw[:, IDX_ERROR_2Q  ].astype(np.float64)
    T1        = X_raw[:, IDX_T1        ].astype(np.float64)
    T2        = X_raw[:, IDX_T2        ].astype(np.float64)
    t1q_ns    = X_raw[:, IDX_TIME_1Q   ].astype(np.float64)
    t2q_ns    = X_raw[:, IDX_TIME_2Q   ].astype(np.float64)
    p_leak    = X_raw[:, IDX_P_LEAK    ].astype(np.float64)
    p_01      = X_raw[:, IDX_P_01      ].astype(np.float64)
    p_10      = X_raw[:, IDX_P_10      ].astype(np.float64)
    p_xt      = X_raw[:, IDX_P_XTALK   ].astype(np.float64)
    beta_nm   = X_raw[:, IDX_BETA_NM   ].astype(np.float64)
    theta_err = X_raw[:, IDX_THETA_ERROR].astype(np.float64)

    FRAC_1Q = 2.0 / 3.0
    FRAC_2Q = 1.0 / 3.0

    t1q_us = t1q_ns * 1e-3
    t2q_us = t2q_ns * 1e-3

    N_total  = nq * d
    t_eff_us = FRAC_1Q * N_total * t1q_us + FRAC_2Q * N_total * t2q_us

    eps_eff    = FRAC_1Q * e1q + FRAC_2Q * e2q
    eps_N      = eps_eff * N_total
    gamma_dep  = np.exp(-(np.where(eps_N > 0, eps_N, 0.0) ** beta_nm))

    gamma_T1   = np.exp(-t_eff_us / np.maximum(T1, 1e-9))
    p_decay_T1 = (1.0 - gamma_T1).astype(np.float32)
    gamma_T2   = np.exp(-t_eff_us / np.maximum(T2, 1e-9))

    gamma_leak  = (1.0 - p_leak) ** N_total
    n_pairs     = nq * (nq - 1) / 2.0
    gamma_xtalk = np.exp(-p_xt * n_pairs * d)

    readout_fac = 1.0 - p_01 - p_10
    gamma_coh   = np.cos(theta_err * N_total)

    survival_combined = (
        gamma_dep * gamma_T1 * gamma_T2 * gamma_leak
        * gamma_xtalk * gamma_coh * readout_fac
    ).astype(np.float32)
    log_surv = np.log(np.clip(np.abs(survival_combined), 1e-10, 1.0)).astype(np.float32)

    e_noisy = X_raw[:, NOISY_EXP_INDICES]
    e1, e3, e5 = e_noisy[:, 0], e_noisy[:, 1], e_noisy[:, 2]

    zne_extrap = ((3.0 * e1 - e3) / 2.0).astype(np.float32)

    lam_dev    = np.array([-4., -2., 0., 2., 4.], dtype=np.float32)
    e_mean     = e_noisy.mean(axis=1)
    b_coef     = (e_noisy * lam_dev).sum(axis=1) / 40.0
    zne_linear = (e_mean - b_coef * 5.0).astype(np.float32)

    # FIX: Do NOT compute or add zne_quad / zne_extrap / zne_linear to X_aug.
    # These are mathematical derivatives of the ZNE sequence and handing them
    # to the MLP branch renders the Transformer branch completely obsolete:
    # the MLP can simply read the finished extrapolation instead of attending
    # to the raw sequence. The Transformer must learn this on its own.

    # shot_noise_variance uses zne_extrap only internally; still compute it
    # for the variance estimate but do NOT expose zne_extrap as a feature.
    shot_noise_variance = ((1.0 - zne_extrap**2) / 8192.0).astype(np.float32)

    # FIX: Remove raw survival_factor from X_aug.
    # survival_combined is a physically richer composite of all noise channels.
    # Keeping the raw survival_factor alongside it creates a perfect information
    # bypass: when survival_combined is masked by 40% dropout, the network
    # just reads the identical information from survival_factor instead.
    # We drop survival_factor (col 19) from X_raw before concatenating.
    raw_cols_to_drop = {"survival_factor"}
    raw_keep_mask = [i for i, n in enumerate(feature_names) if n not in raw_cols_to_drop]
    X_raw_pruned   = X_raw[:, raw_keep_mask]
    feature_names_pruned = [feature_names[i] for i in raw_keep_mask]

    # Augment: pruned raw (24) + 4 derived = 28 total
    # (survival_combined, log_survival, p_decay_T1, shot_noise_variance)
    X_aug = np.concatenate(
        [X_raw_pruned,
         survival_combined[:, None], log_surv[:, None],
         p_decay_T1[:, None], shot_noise_variance[:, None]],
        axis=1,
    )
    aug_names = feature_names_pruned + [
        "survival_combined", "log_survival",
        "p_decay_T1", "shot_noise_variance"
    ]

    # Column pruning
    # FIX: Removed T1_us and T2_us from PRUNE_COLS.
    # These are the fundamental hardware coherence times that the StaticMLPBranch
    # needs to understand the root physical causes of noise. Deleting them forces
    # the model to rely exclusively on derived features, starving it of the
    # underlying physics. Only the raw gate durations (time_1q/2q_ns) are pruned
    # since they are already folded into t_eff_us (used to compute p_decay_T1,
    # p_damp/p_deph) and would otherwise be collinear with those derived features.
    # Fix 2: Eliminate physics multicollinearity.
    # Keep raw hardware parameters (T1, T2, error_1q/2q) as the single level of
    # abstraction. Drop the analytically-derived intermediate probability columns
    # that are fully determined by those raw parameters — they create a "God Feature"
    # shortcut where the network ignores the root physics and reads only the cleanest
    # composite. time_1q/2q_ns are pruned for the same reason (folded into t_eff_us).
    PRUNE_COLS = {
        "time_1q_ns", "time_2q_ns",   # folded into t_eff_us; collinear with derived probs
        "p_damp_1q",  "p_deph_1q",    # derived from T1/T2+time → collinear with T1/T2
        "p_damp_2q",  "p_deph_2q",    # same for 2-qubit gates
        "p_decay_T1",                  # direct T1-decay composite → fully derived from T1
    }
    prune_mask = [i for i, n in enumerate(aug_names) if n in PRUNE_COLS]
    if prune_mask:
        keep = [i for i in range(len(aug_names)) if i not in prune_mask]
        dropped = [aug_names[i] for i in prune_mask]
        print(f"  [Pruning] Dropping: {dropped}")
        X_aug     = X_aug[:, keep]
        aug_names = [aug_names[i] for i in keep]
    print(f"  [Pruning] Active feature count: {len(aug_names)} columns.")

    # Survival filter
    mask   = np.abs(survival_combined) >= min_survival
    X_aug  = X_aug[mask]
    y_raw  = y_raw[mask]
    N_kept = int(mask.sum())
    print(f"  Survival filter (>= {min_survival}): kept {N_kept:,} / {len(mask):,} "
          f"({100.0 * N_kept / len(mask):.1f}%)")

    # Identify ZNE column indices post-pruning (for the Transformer branch)
    zne_names = {"exp_val_scale_1", "exp_val_scale_3", "exp_val_scale_5",
                 "exp_val_scale_7", "exp_val_scale_9"}
    zne_col_indices = [i for i, n in enumerate(aug_names) if n in zne_names]
    static_feature_indices = [i for i in range(len(aug_names)) if i not in zne_col_indices]

    print(f"  ZNE cols (Transformer): {[aug_names[i] for i in zne_col_indices]}")
    print(f"  Static cols (MLP): {len(static_feature_indices)} features")

    # Move to device
    print("  Moving data to device...")
    X_aug_pt = torch.tensor(X_aug, device=device, dtype=torch.float32)
    y_raw_pt = torch.tensor(y_raw, device=device, dtype=torch.float32)

    N_data  = len(y_raw_pt)
    n_test  = int(N_data * test_split)
    n_val   = int(N_data * val_split)
    n_train = N_data - n_test - n_val

    if device.type == 'cpu':
        g = torch.Generator()
    else:
        g = torch.Generator(device=device)
    g.manual_seed(seed)
    idx = torch.randperm(N_data, generator=g, device=device)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train : n_train + n_val]
    test_idx  = idx[n_train + n_val :]

    # StandardScaler (same no-scale logic as MLP)
    # NOTE: zne_extrap / zne_linear / zne_quad have been removed from X_aug
    # entirely — they no longer appear in aug_names and need no special handling.
    no_scale_names = {
        "exp_val_scale_1", "exp_val_scale_3", "exp_val_scale_5",
        "exp_val_scale_7", "exp_val_scale_9",
        "shot_noise_variance"
    }
    no_scale_idx = [i for i, n in enumerate(aug_names) if n in no_scale_names]
    scale_idx    = [i for i in range(X_aug_pt.shape[1]) if i not in no_scale_idx]

    X_train_static = X_aug_pt[train_idx][:, scale_idx]
    mean = X_train_static.mean(dim=0)
    std  = X_train_static.std(dim=0, unbiased=False)
    std[std == 0] = 1.0

    def scale_dataset(X):
        X_scaled = torch.empty_like(X, dtype=torch.float32, device=device)
        X_scaled[:, scale_idx]    = (X[:, scale_idx] - mean) / std
        X_scaled[:, no_scale_idx] = X[:, no_scale_idx]
        return X_scaled

    X_train = scale_dataset(X_aug_pt[train_idx])
    X_val   = scale_dataset(X_aug_pt[val_idx])
    X_test  = scale_dataset(X_aug_pt[test_idx])

    y_train = y_raw_pt[train_idx]
    y_val   = y_raw_pt[val_idx]
    y_test  = y_raw_pt[test_idx]

    # ------------------------------------------------------------------
    # Fix 1: Residual Learning — target = y_ideal − baseline(λ=1)
    # The unmitigated λ=1 value is the strongest single predictor of y_ideal.
    # Subtracting it from the target destroys the algebraic shortcut so the
    # optimizer is forced to explain the *error* using hardware metadata.
    # ZNE columns are not scaled (in no_scale_idx), so X[:, baseline_col_index]
    # is always the raw λ=1 expectation value, safe to use as a baseline.
    # ------------------------------------------------------------------
    baseline_col_index = next(
        i for i, n in enumerate(aug_names) if n == "exp_val_scale_1"
    )
    baseline_train = X_train[:, baseline_col_index]  # (N_train,) raw λ=1 values
    baseline_val   = X_val[:,   baseline_col_index]
    baseline_test  = X_test[:,  baseline_col_index]

    y_train_res = y_train - baseline_train   # residual = ideal − unmitigated
    y_val_res   = y_val   - baseline_val
    y_test_res  = y_test  - baseline_test

    print(f"  [Residual] baseline col = '{aug_names[baseline_col_index]}' "
          f"(idx {baseline_col_index})")
    print(f"  [Residual] train target  mean={y_train_res.mean():.5f} "
          f"std={y_train_res.std():.5f}")

    # Feature dropout schedule
    # With survival_factor removed entirely and zne_extrap/linear/quad gone,
    # the bypass routes are closed. The remaining schedule keeps regularization
    # on the still-powerful composite survival signal and on exp_val_scale_1
    # (the lowest-noise, highest-information ZNE point).
    def _cols(*names):
        return tuple(i for i, n in enumerate(aug_names) if n in set(names))

    dropout_schedule = {
        _cols("exp_val_scale_1")                                      : 0.50,
        _cols("survival_combined", "log_survival")                    : 0.40,
        _cols("exp_val_scale_3", "exp_val_scale_5",
              "exp_val_scale_7", "exp_val_scale_9")                   : 0.25,
    }
    dropout_schedule = {k: v for k, v in dropout_schedule.items() if k}

    print(f"  [Dropout] Feature dropout schedule:")
    for cols, prob in dropout_schedule.items():
        names_in_group = [aug_names[c] for c in cols]
        print(f"    prob={prob:.0%}  cols={names_in_group}")

    # Datasets: store residuals as y; baseline stored separately and immune to dropout
    train_ds = TabularFeatureDropout(X_train, y_train_res, baseline_train,
                                     is_train=True, dropout_schedule=dropout_schedule)
    val_ds   = TabularFeatureDropout(X_val,   y_val_res,   baseline_val,   is_train=False)
    test_ds  = TabularFeatureDropout(X_test,  y_test_res,  baseline_test,  is_train=False)

    print(f"  Train : {len(train_ds):,} samples")
    print(f"  Val   : {len(val_ds):,} samples")
    print(f"  Test  : {len(test_ds):,} samples")

    class DummyScaler:
        def __init__(self, m, s):
            self.mean_ = m.cpu().numpy()
            self.scale_ = s.cpu().numpy()

    scaler = DummyScaler(mean, std)

    return train_ds, val_ds, test_ds, scaler, aug_names, zne_col_indices, baseline_col_index


# ===========================================================================
# Model Definition — QEMFormer
# ===========================================================================

class ResBlock(nn.Module):
    """Residual block with BatchNorm, GELU, and skip connection."""
    def __init__(self, in_dim, out_dim, dropout=0.05):
        super().__init__()
        self.fc   = nn.Linear(in_dim, out_dim)
        self.bn   = nn.BatchNorm1d(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.act  = nn.GELU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.bn(self.fc(x))
        out = out + self.proj(x)
        out = self.act(out)
        return self.drop(out)


class ZNETransformerBranch(nn.Module):
    """
    Transformer encoder branch for the ZNE scale-factor sequence.

    Treats the 5 ZNE expectation values (at λ=1,3,5,7,9) as a length-5
    sequence. Each value is projected to d_model dimensions and augmented
    with learnable positional embeddings indexed by the scale factor values.

    Fix 3 — Static scale-factor tensors are registered as persistent buffers
    so they live permanently in GPU VRAM with no per-batch CPU→GPU transfer.

    Fix 4 — Replaces single-query attention pooling with a [CLS] token.
    The CLS token is prepended to the 5 ZNE tokens (sequence length = 6).
    Transformer self-attention at every layer routes information from ALL noise
    steps into the CLS position, naturally enforcing gradient flow across the
    full ZNE profile instead of collapsing to just λ=1 and λ=3.

    Architecture:
        Input (batch, 5) → Linear(1, d_model) per token
        Prepend [CLS] → (batch, 6, d_model)
        + Learnable positional embedding (6 positions: CLS + 5 ZNE)
        + Scale factor embedding (λ=0 for CLS, λ=1,3,5,7,9 for ZNE tokens)
        → N × TransformerEncoderLayer (d_model, nhead, dim_ff)
        → Extract CLS output → (batch, d_model)
    """
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_ff=256, dropout=0.05):
        super().__init__()
        self.d_model = d_model

        # Project each scalar ZNE value to d_model dimensions
        self.token_proj = nn.Linear(1, d_model)

        # Fix 4: [CLS] token — accumulates full sequence representation via self-attention
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable positional embedding for 6 positions (CLS + 5 ZNE)
        self.pos_embed = nn.Parameter(torch.randn(1, 6, d_model) * 0.02)

        # Scale factor encoding: embed the actual λ values as additional context.
        # Fix 3: pre-compute the (1, 6, 1) scale-factor tensor as a persistent buffer.
        # CLS token gets λ=0; ZNE tokens get λ=1,3,5,7,9.
        self.scale_factor_embed = nn.Linear(1, d_model)
        self.register_buffer(
            "scale_factors_buf",
            torch.tensor([0.0] + SCALE_FACTORS, dtype=torch.float32)
            .unsqueeze(0)   # (1, 6)
            .unsqueeze(-1)  # (1, 6, 1)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, zne_values):
        """
        Args:
            zne_values: (batch, 5) — ZNE exp values at scales [1, 3, 5, 7, 9]
        Returns:
            (batch, d_model) — CLS token output after transformer encoding
        """
        B = zne_values.shape[0]

        # (batch, 5) → (batch, 5, 1) → (batch, 5, d_model)
        zne_tokens = self.token_proj(zne_values.unsqueeze(-1))

        # Fix 4: prepend [CLS] token → (batch, 6, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, zne_tokens], dim=1)

        # Add positional embedding (covers all 6 positions)
        tokens = tokens + self.pos_embed

        # Add scale factor encoding using pre-computed buffer (Fix 3)
        # scale_factors_buf: (1, 6, 1) → expand to (batch, 6, 1) → embed
        scale_enc = self.scale_factor_embed(
            self.scale_factors_buf.expand(B, -1, -1)
        )  # (batch, 6, d_model)
        tokens = tokens + scale_enc

        # Transformer encoding: self-attention at every layer routes ZNE info into CLS
        encoded = self.transformer(tokens)  # (batch, 6, d_model)

        # Fix 4: extract CLS token (position 0) as the sequence summary
        cls_out = encoded[:, 0, :]  # (batch, d_model)

        return self.layer_norm(cls_out)


class StaticMLPBranch(nn.Module):
    """
    MLP branch for static circuit/noise features (short-range context).
    3 residual blocks with decreasing width.
    """
    def __init__(self, input_dim, hidden1=512, hidden2=512, hidden3=256, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(input_dim, hidden1, dropout),
            ResBlock(hidden1,  hidden2, dropout),
            ResBlock(hidden2,  hidden3, dropout),
        )

    def forward(self, x):
        return self.net(x)


class QEMFormer(nn.Module):
    """
    QEMFormer: Two-branch transformer model for quantum error mitigation.

    Branch 1 (MLP): Processes static circuit metadata and noise parameters
    Branch 2 (Transformer): Processes the 5 ZNE expectation values as a sequence

    The branches are fused via concatenation and passed through a fusion MLP
    to predict the ideal (noiseless) expectation value.
    """
    def __init__(
        self,
        n_static_features,
        zne_col_indices,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_ff=256,
        mlp_hidden1=512,
        mlp_hidden2=512,
        mlp_hidden3=256,
        dropout=0.05,
    ):
        super().__init__()

        self.zne_col_indices = zne_col_indices
        self.n_total_cols = n_static_features + len(zne_col_indices)

        # Branch 1: Static MLP
        self.mlp_branch = StaticMLPBranch(
            input_dim=n_static_features,
            hidden1=mlp_hidden1,
            hidden2=mlp_hidden2,
            hidden3=mlp_hidden3,
            dropout=dropout,
        )

        # Branch 2: ZNE Transformer
        self.transformer_branch = ZNETransformerBranch(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )

        # Fusion head: concat MLP output (mlp_hidden3) + transformer output (d_model)
        fusion_input_dim = mlp_hidden3 + d_model
        self.fusion = nn.Sequential(
            ResBlock(fusion_input_dim, 512, dropout),
            ResBlock(512, 256, dropout),
            ResBlock(256, 128, dropout),
            nn.Linear(128, 1),
        )

        # Fix 3: Pre-compute static/ZNE index buffers as persistent GPU tensors.
        # Registering them here eliminates per-batch torch.tensor() + .to(device)
        # calls inside forward(), removing the CPU→GPU stall on every batch.
        n_total = n_static_features + len(zne_col_indices)
        zne_buf = torch.tensor(zne_col_indices, dtype=torch.long)
        all_buf = torch.arange(n_total, dtype=torch.long)
        static_buf = all_buf[~torch.isin(all_buf, zne_buf)]
        self.register_buffer("_zne_indices_buf",    zne_buf)
        self.register_buffer("_static_indices_buf", static_buf)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, n_total_cols) — full feature matrix
        Returns:
            (batch, 1) — predicted error residual (add λ=1 baseline to get ideal value)
        """
        # Fix 3: use pre-computed persistent buffers — no runtime tensor allocation
        x_static = x[:, self._static_indices_buf]   # (batch, n_static)
        x_zne    = x[:, self._zne_indices_buf]       # (batch, 5)

        # Branch 1: MLP on static features
        mlp_out = self.mlp_branch(x_static)  # (batch, mlp_hidden3)

        # Branch 2: Transformer on ZNE sequence
        tf_out = self.transformer_branch(x_zne)  # (batch, d_model)

        # Fusion
        fused = torch.cat([mlp_out, tf_out], dim=1)
        out = self.fusion(fused)  # (batch, 1)

        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Physics-Informed Loss
# ===========================================================================

class PhysicsInformedLoss(nn.Module):
    """
    L_total = MSE(y_pred, y_ideal) + lambda_bndry * boundary_penalty

    Boundary penalty: quantum expectation values must lie in [-1, 1].
    """
    def __init__(self, lambda_bndry=0.10):
        super().__init__()
        self.lambda_bndry = lambda_bndry
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true, X=None):
        mse_loss = self.mse(y_pred.squeeze(-1), y_true)
        boundary_penalty = torch.relu(y_pred.abs() - 1.0).mean()
        total_loss = mse_loss + self.lambda_bndry * boundary_penalty
        return total_loss, mse_loss, boundary_penalty


# ===========================================================================
# Metrics
# ===========================================================================

def r2_score(y_pred, y_true):
    """R² coefficient of determination."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1.0 - ss_res / ss_tot).item()


# ===========================================================================
# Training & Evaluation
# ===========================================================================

def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, scheduler=None):
    model.train()
    total_loss = mse_sum = bndry_sum = 0.0
    all_pred, all_true = [], []
    use_amp = (device.type == "cuda" and scaler is not None)

    # Fix 1: loader yields (X, y_residual, baseline) 3-tuples.
    # The model predicts residuals; loss and metrics are computed in ideal-value space.
    for X_batch, y_residual_batch, baseline_batch in loader:
        X_batch          = X_batch.to(device)
        y_residual_batch = y_residual_batch.to(device)
        baseline_batch   = baseline_batch.to(device)

        # Reconstruct ideal ground-truth for loss / metrics
        y_ideal_batch = y_residual_batch + baseline_batch

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            # Model predicts residual; reconstruct ideal prediction
            y_pred_residual = model(X_batch)
            y_pred_ideal    = y_pred_residual + baseline_batch.unsqueeze(-1)
            loss, mse, bndry = criterion(y_pred_ideal, y_ideal_batch, X_batch)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs = len(y_residual_batch)
        total_loss += loss.item() * bs
        mse_sum    += mse.item() * bs
        bndry_sum  += bndry.item() * bs

        all_pred.append(y_pred_ideal.squeeze(-1).detach())
        all_true.append(y_ideal_batch.detach())

    n = len(loader.dataset)
    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    mae  = (all_pred - all_true).abs().mean().item()
    r2   = r2_score(all_pred, all_true)
    acc  = ((all_pred - all_true).abs() <= 0.05).float().mean().item() * 100.0

    return {
        "loss"  : total_loss / n,
        "mse"   : mse_sum / n,
        "bndry" : bndry_sum / n,
        "mae"   : mae,
        "r2"    : r2,
        "acc"   : acc,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = mse_sum = bndry_sum = 0.0
    all_pred, all_true = [], []

    # Fix 1: loader yields (X, y_residual, baseline) 3-tuples.
    for X_batch, y_residual_batch, baseline_batch in loader:
        X_batch          = X_batch.to(device)
        y_residual_batch = y_residual_batch.to(device)
        baseline_batch   = baseline_batch.to(device)

        y_ideal_batch = y_residual_batch + baseline_batch

        with torch.autocast(device_type=device.type, enabled=use_amp):
            y_pred_residual = model(X_batch)
            y_pred_ideal    = y_pred_residual + baseline_batch.unsqueeze(-1)
            loss, mse, bndry = criterion(y_pred_ideal, y_ideal_batch, X_batch)

        bs = len(y_residual_batch)
        total_loss += loss.item() * bs
        mse_sum    += mse.item() * bs
        bndry_sum  += bndry.item() * bs

        all_pred.append(y_pred_ideal.squeeze(-1))
        all_true.append(y_ideal_batch)

    n = len(loader.dataset)
    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    mae  = (all_pred - all_true).abs().mean().item()
    r2   = r2_score(all_pred, all_true)
    acc  = ((all_pred - all_true).abs() <= 0.05).float().mean().item() * 100.0

    return {
        "loss"  : total_loss / n,
        "mse"   : mse_sum / n,
        "bndry" : bndry_sum / n,
        "mae"   : mae,
        "r2"    : r2,
        "acc"   : acc,
        "pred"  : all_pred.cpu().numpy(),
        "true"  : all_true.cpu().numpy(),
    }


# ===========================================================================
# Visualization Suite
# ===========================================================================

# Dark theme configuration for all plots
DARK_BG     = "#0f0f1a"
DARK_PANEL  = "#1a1a2e"
DARK_SPINE  = "#333355"
LIGHT_TEXT  = "#aaaacc"
WHITE_TEXT  = "white"
COLOR_TRAIN = "#7ec8e3"
COLOR_VAL   = "#f4a261"
COLOR_ACCENT = "#e76f51"


def _setup_dark_ax(ax):
    """Apply dark theme to an axis."""
    ax.set_facecolor(DARK_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(DARK_SPINE)
    ax.tick_params(colors=LIGHT_TEXT)


def plot_training_curves(history, out_dir):
    """2×2 grid of training curves: Loss, MSE, MAE, R²."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    axes_cfg = [
        (0, 0, "loss",  "Total Loss",  True ),
        (0, 1, "mse",   "MSE Loss",    True ),
        (1, 0, "mae",   "MAE",         False),
        (1, 1, "r2",    "R² Score",    False),
    ]

    for row, col, key, title, logy in axes_cfg:
        ax = fig.add_subplot(gs[row, col])
        _setup_dark_ax(ax)

        tr_vals  = history[f"train_{key}"]
        val_vals = history[f"val_{key}"]

        ax.plot(epochs, tr_vals,  color=COLOR_TRAIN, lw=1.8,
                label="Train", alpha=0.9)
        ax.plot(epochs, val_vals, color=COLOR_VAL, lw=1.8,
                label="Validation", alpha=0.9, linestyle="--")

        if logy:
            ax.set_yscale("log")

        ax.set_title(title, color=WHITE_TEXT, fontsize=12, pad=8)
        ax.set_xlabel("Epoch", color=LIGHT_TEXT, fontsize=10)
        ax.legend(fontsize=9, framealpha=0.2, labelcolor=WHITE_TEXT)
        ax.grid(True, alpha=0.15, color="#ffffff")

    fig.suptitle("QEMFormer — Training Curves",
                 color=WHITE_TEXT, fontsize=15, y=1.01, fontweight="bold")

    path = out_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_predictions(pred, true, out_dir):
    """Predicted vs Ideal scatter colored by |error|."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    _setup_dark_ax(ax)

    sc = ax.scatter(true, pred, c=np.abs(pred - true), cmap="plasma",
                    alpha=0.45, s=12, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("|Error|", color=LIGHT_TEXT)
    cbar.ax.yaxis.set_tick_params(color=LIGHT_TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=LIGHT_TEXT)

    lims = [-1.05, 1.05]
    ax.plot(lims, lims, color=COLOR_TRAIN, lw=1.5, linestyle="--",
            label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    r2  = r2_score(torch.tensor(pred), torch.tensor(true))
    mae = float(np.abs(pred - true).mean())
    ax.set_title(f"QEMFormer — Test Set: Predicted vs Ideal\nR²={r2:.4f}   MAE={mae:.5f}",
                 color=WHITE_TEXT, fontsize=12)
    ax.set_xlabel("Ideal Expectation Value", color=LIGHT_TEXT, fontsize=11)
    ax.set_ylabel("Mitigated (Predicted)", color=LIGHT_TEXT, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor=WHITE_TEXT)
    ax.grid(True, alpha=0.12, color="#ffffff")

    path = out_dir / "predictions_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_residuals_kde(pred, true, out_dir):
    """KDE plot of error residuals."""
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    _setup_dark_ax(ax)

    residuals = pred - true

    kde = gaussian_kde(residuals)
    x_range = np.linspace(min(residuals), max(residuals), 500)
    y_kde = kde(x_range)

    ax.plot(x_range, y_kde, color=COLOR_TRAIN, lw=2)
    ax.fill_between(x_range, y_kde, color=COLOR_TRAIN, alpha=0.3)
    ax.axvline(0, color='#ff6b6b', linestyle='--', linewidth=2, label="Zero Error")

    mae = float(np.abs(residuals).mean())
    std = float(np.std(residuals))

    ax.set_title(f"QEMFormer — Error Residuals Distribution\n"
                 f"Mean: {np.mean(residuals):.5f} | Std: {std:.5f} | MAE: {mae:.5f}",
                 color=WHITE_TEXT, fontsize=12)
    ax.set_xlabel("Error (Predicted - Ideal)", color=LIGHT_TEXT, fontsize=11)
    ax.set_ylabel("Density", color=LIGHT_TEXT, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor=WHITE_TEXT)
    ax.grid(True, alpha=0.12, color="#ffffff")

    path = out_dir / "predictions_residuals_kde.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_error_vs_survival(pred, true, X_test_np, feature_names, out_dir):
    """Plot |error| vs survival factor to assess difficulty-dependent performance."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    _setup_dark_ax(ax)

    # Find survival_combined column
    surv_idx = None
    for i, name in enumerate(feature_names):
        if name == "survival_combined":
            surv_idx = i
            break

    if surv_idx is None:
        # Fallback: use survival_factor from raw features
        for i, name in enumerate(feature_names):
            if name == "survival_factor":
                surv_idx = i
                break

    if surv_idx is None:
        print("  [WARNING] Could not find survival column for error_vs_survival plot.")
        plt.close()
        return

    survival = X_test_np[:, surv_idx]
    abs_error = np.abs(pred - true)

    # Bin by survival factor
    n_bins = 20
    bin_edges = np.linspace(np.min(survival), np.max(survival), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = []
    bin_stds = []

    for i in range(n_bins):
        mask = (survival >= bin_edges[i]) & (survival < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means.append(abs_error[mask].mean())
            bin_stds.append(abs_error[mask].std())
        else:
            bin_means.append(0)
            bin_stds.append(0)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    ax.bar(bin_centers, bin_means, width=(bin_edges[1] - bin_edges[0]) * 0.85,
           color=COLOR_TRAIN, alpha=0.7, edgecolor=DARK_SPINE)
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='none',
                ecolor=COLOR_VAL, capsize=3, alpha=0.6)

    ax.set_title("QEMFormer — |Error| vs Noise Survival Factor",
                 color=WHITE_TEXT, fontsize=12)
    ax.set_xlabel("Survival Factor (higher = less noise)", color=LIGHT_TEXT, fontsize=11)
    ax.set_ylabel("Mean |Error|", color=LIGHT_TEXT, fontsize=11)
    ax.grid(True, alpha=0.12, color="#ffffff")

    path = out_dir / "error_vs_survival.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_error_by_scale(pred, true, X_test_np, feature_names, out_dir):
    """
    Bar chart showing how well QEMFormer corrects compared to each raw
    ZNE scale factor: |E_noisy_λ - E_ideal| vs |E_predicted - E_ideal|.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    _setup_dark_ax(ax)

    scale_names = [f"exp_val_scale_{lam}" for lam in SCALE_FACTORS]
    scale_indices = []
    for sn in scale_names:
        for i, fn in enumerate(feature_names):
            if fn == sn:
                scale_indices.append(i)
                break

    # Also include ZNE extrapolation baselines if available
    zne_baselines = {}
    for extra_name in ["zne_extrap", "zne_linear", "zne_quad"]:
        for i, fn in enumerate(feature_names):
            if fn == extra_name:
                zne_baselines[extra_name] = i
                break

    abs_error_pred = np.abs(pred - true).mean()
    labels = []
    errors = []
    colors = []

    for si, sn in zip(scale_indices, scale_names):
        noisy = X_test_np[:, si]
        err = np.abs(noisy - true).mean()
        labels.append(sn.replace("exp_val_scale_", "λ="))
        errors.append(err)
        colors.append("#8ecae6")

    for name, idx in zne_baselines.items():
        baseline = X_test_np[:, idx]
        err = np.abs(baseline - true).mean()
        labels.append(name)
        errors.append(err)
        colors.append("#219ebc")

    labels.append("QEMFormer")
    errors.append(abs_error_pred)
    colors.append(COLOR_ACCENT)

    bars = ax.barh(labels, errors, color=colors, edgecolor=DARK_SPINE, height=0.6)

    # Highlight QEMFormer bar
    bars[-1].set_edgecolor(COLOR_VAL)
    bars[-1].set_linewidth(2)

    ax.set_title("QEMFormer — Mean |Error| vs Baselines",
                 color=WHITE_TEXT, fontsize=12)
    ax.set_xlabel("Mean |Error|", color=LIGHT_TEXT, fontsize=11)
    ax.grid(axis='x', alpha=0.15, color="#ffffff")

    # Annotate values
    for bar, err in zip(bars, errors):
        ax.text(err + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{err:.4f}", va='center', color=LIGHT_TEXT, fontsize=9)

    path = out_dir / "error_by_scale_factor.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def compute_permutation_importance(model, loader, criterion, device,
                                   feature_names, out_dir):
    """Compute and plot permutation feature importance."""
    print("\n  Computing Permutation Feature Importance ...")
    model.eval()

    X_all, y_all, baseline_all = [], [], []
    for X, y_res, bl in loader:
        X_all.append(X)
        y_all.append(y_res)
        baseline_all.append(bl)
    X_all        = torch.cat(X_all,        dim=0).to(device)
    y_res_all    = torch.cat(y_all,        dim=0).to(device)
    baseline_all = torch.cat(baseline_all, dim=0).to(device)
    y_ideal_all  = y_res_all + baseline_all

    with torch.no_grad():
        baseline_pred_res = model(X_all)
        baseline_pred     = baseline_pred_res + baseline_all.unsqueeze(-1)
        _, baseline_mse, _ = criterion(baseline_pred, y_ideal_all, X_all)
        baseline_mse = baseline_mse.item()

        importances = []
        for i in range(X_all.shape[1]):
            X_shuffled = X_all.clone()
            perm = torch.randperm(X_all.shape[0], device=device)
            X_shuffled[:, i] = X_all[perm, i]

            pred_res = model(X_shuffled)
            pred     = pred_res + baseline_all.unsqueeze(-1)
            _, mse, _ = criterion(pred, y_ideal_all, X_shuffled)
            importances.append(mse.item() - baseline_mse)

    importances = np.array(importances)
    idx = np.argsort(importances)

    sorted_names = [feature_names[j] for j in idx]
    sorted_imps  = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(8, len(sorted_names) * 0.35)))
    fig.patch.set_facecolor(DARK_BG)
    _setup_dark_ax(ax)

    ax.barh(range(len(sorted_imps)), sorted_imps, color=COLOR_VAL,
            edgecolor=DARK_SPINE)
    ax.set_yticks(range(len(sorted_imps)))
    ax.set_yticklabels(sorted_names, color=LIGHT_TEXT)

    ax.set_xlabel("Increase in MSE (Permutation Importance)",
                  color=LIGHT_TEXT, fontsize=11)
    ax.set_title("QEMFormer — Feature Importance (Test Set)",
                 color=WHITE_TEXT, fontsize=14)
    ax.grid(axis='x', alpha=0.15, color="#ffffff")

    path = out_dir / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


# ===========================================================================
# Main Training Pipeline
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train QEMFormer for Quantum Error Mitigation"
    )
    p.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",            type=float, default=DEFAULT_LR)
    p.add_argument("--batch",         type=int,   default=DEFAULT_BATCH)
    p.add_argument("--seed",          type=int,   default=DEFAULT_SEED)
    p.add_argument("--val_split",     type=float, default=DEFAULT_VAL_SPLIT)
    p.add_argument("--test_split",    type=float, default=DEFAULT_TEST_SPLIT)
    p.add_argument("--lambda_bndry",  type=float, default=DEFAULT_LAMBDA_BNDRY)
    p.add_argument("--dropout",       type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--weight_decay",  type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--d_model",       type=int,   default=DEFAULT_D_MODEL)
    p.add_argument("--nhead",         type=int,   default=DEFAULT_NHEAD)
    p.add_argument("--num_layers",    type=int,   default=DEFAULT_NUM_LAYERS)
    p.add_argument("--dim_ff",        type=int,   default=DEFAULT_DIM_FF)
    p.add_argument("--dataset",       type=str,   default=str(DATASET_NPZ))
    p.add_argument("--min_survival",  type=float, default=MIN_SURVIVAL)
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  [Optimization] Fast mode: TF32 + cuDNN benchmark ON.")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n  [Device] CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n  [Device] Apple MPS")
    else:
        device = torch.device("cpu")
        print(f"\n  [Device] CPU")

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  QEMFormer — Transformer-Based Quantum Error Mitigation")
    print("=" * 70)
    print(f"  Dataset       : {args.dataset}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Learning rate : {args.lr}")
    print(f"  Batch size    : {args.batch}")
    print(f"  Min survival  : {args.min_survival}")
    print(f"  lambda_bndry  : {args.lambda_bndry}")
    print(f"  Dropout       : {args.dropout}")
    print(f"  Weight decay  : {args.weight_decay}")
    print(f"  d_model       : {args.d_model}")
    print(f"  nhead         : {args.nhead}")
    print(f"  num_layers    : {args.num_layers}")
    print(f"  dim_ff        : {args.dim_ff}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading & splitting dataset ...")
    train_ds, val_ds, test_ds, scaler, feature_names, zne_col_indices, baseline_col_index = load_and_split(
        npz_path     = Path(args.dataset),
        val_split    = args.val_split,
        test_split   = args.test_split,
        seed         = args.seed,
        min_survival = args.min_survival,
        device       = device,
    )

    train_loader = GPUDataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = GPUDataLoader(val_ds,   batch_size=args.batch, shuffle=False)
    test_loader  = GPUDataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\n[2/4] Building QEMFormer model ...")

    n_total_features = train_ds.X.shape[1]
    n_static = n_total_features - len(zne_col_indices)

    model = QEMFormer(
        n_static_features = n_static,
        zne_col_indices   = zne_col_indices,
        d_model           = args.d_model,
        nhead             = args.nhead,
        num_layers        = args.num_layers,
        dim_ff            = args.dim_ff,
        mlp_hidden1       = 512,
        mlp_hidden2       = 512,
        mlp_hidden3       = 256,
        dropout           = args.dropout,
    ).to(device)

    print(f"\n  Architecture:")
    print(f"  {model}")
    print(f"\n  Total features    : {n_total_features}")
    print(f"  Static (MLP)      : {n_static}")
    print(f"  ZNE (Transformer) : {len(zne_col_indices)}")
    print(f"  Trainable params  : {model.count_parameters():,}")

    criterion = PhysicsInformedLoss(lambda_bndry=args.lambda_bndry)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )
    if device.type == "cuda":
        print("  [Optimization] Fused AdamW enabled.")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )

    use_amp = (device.type == "cuda")
    grad_scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("  [Optimization] Automatic Mixed Precision (AMP) enabled.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print(f"\n[3/4] Training for {args.epochs} epochs ...")
    print("-" * 90)
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | "
          f"{'Train MAE':>10} | {'Val MAE':>9} | {'Val R²':>8} | "
          f"{'Tr Acc%':>7} | {'Val Acc%':>8} | LR")
    print("-" * 90)

    history = {k: [] for k in
               ["train_loss", "val_loss", "train_mse", "val_mse",
                "train_mae", "val_mae", "train_r2", "val_r2",
                "train_acc", "val_acc"]}

    best_val_loss    = float("inf")
    patience_counter = 0
    best_ckpt_path   = OUTPUT_DIR / "best_qemformer_model.pth"
    log_path         = OUTPUT_DIR / "training_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_mae", "val_mae",
                         "train_r2", "val_r2", "lr"])

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device,
                             scaler=grad_scaler, scheduler=scheduler)
        vl = evaluate(model, val_loader, criterion, device, use_amp=use_amp)

        current_lr = scheduler.get_last_lr()[0]

        for k in ["loss", "mse", "mae", "r2", "acc"]:
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(vl[k])

        if vl["loss"] < best_val_loss:
            best_val_loss = vl["loss"]
            torch.save({
                "epoch"             : epoch,
                "model_state"       : model.state_dict(),
                "optimizer_state"   : optimizer.state_dict(),
                "val_loss"          : best_val_loss,
                "args"              : vars(args),
                "feature_names"     : feature_names,
                "zne_col_indices"   : zne_col_indices,
                "baseline_col_index": baseline_col_index,
                "scaler_mean"       : scaler.mean_.tolist(),
                "scaler_scale"      : scaler.scale_.tolist(),
            }, best_ckpt_path)
            star = " *"
            patience_counter = 0
        else:
            star = ""
            patience_counter += 1

        elapsed = time.time() - t_start
        print(f"{epoch:>6} | {tr['loss']:>11.6f} | {vl['loss']:>10.6f} | "
              f"{tr['mae']:>10.6f} | {vl['mae']:>9.6f} | {vl['r2']:>8.4f} | "
              f"{tr['acc']:>6.1f}% | {vl['acc']:>7.1f}% | "
              f"{current_lr:.2e}{star}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr["loss"], vl["loss"],
                             tr["mae"], vl["mae"], tr["r2"], vl["r2"],
                             current_lr])

        if patience_counter >= 20:
            print(f"\n  [Early Stopping] No improvement for 20 epochs. "
                  f"Stopping at epoch {epoch}.")
            break

    total_time = time.time() - t_start
    print("-" * 90)
    print(f"  Training complete in {total_time:.1f}s "
          f"({total_time / epoch:.2f}s/epoch)")
    print(f"  Best val loss: {best_val_loss:.6f}  -->  {best_ckpt_path}")

    # Save final model
    final_path = OUTPUT_DIR / "final_qemformer_model.pth"
    torch.save({
        "epoch"             : epoch,
        "model_state"       : model.state_dict(),
        "args"              : vars(args),
        "feature_names"     : feature_names,
        "zne_col_indices"   : zne_col_indices,
        "baseline_col_index": baseline_col_index,
        "scaler_mean"       : scaler.mean_.tolist(),
        "scaler_scale"      : scaler.scale_.tolist(),
    }, final_path)
    print(f"  Final model  : {final_path}")

    # ------------------------------------------------------------------
    # Test set evaluation (best checkpoint)
    # ------------------------------------------------------------------
    print(f"\n[4/4] Evaluating on held-out test set ...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp=use_amp)

    print("\n" + "=" * 70)
    print("  QEMFormer — TEST SET RESULTS")
    print("=" * 70)
    print(f"  Total Loss     : {test_metrics['loss']:.6f}")
    print(f"  MSE Loss       : {test_metrics['mse']:.6f}")
    print(f"  MAE            : {test_metrics['mae']:.6f}")
    print(f"  R² Score       : {test_metrics['r2']:.6f}")
    print(f"  RMSE           : {np.sqrt(test_metrics['mse']):.6f}")
    print(f"  Accuracy (±.05): {test_metrics['acc']:.2f}%")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Collect test X for analysis plots
    # ------------------------------------------------------------------
    X_test_all = []
    for X_batch, _, __ in test_loader:
        X_test_all.append(X_batch.cpu().numpy())
    X_test_np = np.concatenate(X_test_all, axis=0)

    # ------------------------------------------------------------------
    # Visualization Suite
    # ------------------------------------------------------------------
    print("\n  Generating visualization suite ...")

    plot_training_curves(history, OUTPUT_DIR)
    plot_predictions(test_metrics["pred"], test_metrics["true"], OUTPUT_DIR)
    plot_residuals_kde(test_metrics["pred"], test_metrics["true"], OUTPUT_DIR)
    plot_error_vs_survival(test_metrics["pred"], test_metrics["true"],
                           X_test_np, feature_names, OUTPUT_DIR)
    plot_error_by_scale(test_metrics["pred"], test_metrics["true"],
                        X_test_np, feature_names, OUTPUT_DIR)
    compute_permutation_importance(model, test_loader, criterion, device,
                                   feature_names, OUTPUT_DIR)

    print(f"\n  Training log : {log_path}")
    print(f"  All outputs  : {OUTPUT_DIR}")
    print(f"  R² Score     : {test_metrics['r2']:.6f}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
