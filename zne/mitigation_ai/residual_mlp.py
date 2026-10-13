"""
mlp.py
======
Physics-Informed Multi-Layer Perceptron (PI-MLP) for Quantum Error Mitigation.

Architecture (funnel):
    Input (18)  --> Hidden-1 (128, LeakyReLU)
                --> Hidden-2 ( 64, LeakyReLU)
                --> Hidden-3 ( 32, LeakyReLU)
                --> Output   (  1, Linear)   -- scalar mitigated expectation value

Input features (22 total)
--------------------------
  Raw features from dataset (17):
  [0]  num_qubits       -- circuit metadata
  [1]  depth
  [2]  error_rate       -- base depolarizing error rate per gate
  [3]  T1_us            -- amplitude damping (T1 relaxation) time [us]
  [4]  T2_us            -- dephasing (T2 decoherence) time [us]
  [5]  gate_time_ns     -- single-qubit gate time [ns] (constant 50 ns)
  [6]  p_damp           -- amplitude damping prob per gate = 1-exp(-t/T1)
  [7]  p_deph           -- dephasing prob per gate        = 1-exp(-t/T2)
  [8]  p_leak           -- leakage to |2> prob per qubit per gate
  [9]  p_readout        -- symmetric readout confusion probability (SPAM)
  [10] p_xtalk          -- ZZ crosstalk dephasing per qubit-pair per depth layer
  [11] beta_nm          -- non-Markovian stretch exponent for depolarizing
  [12-16] exp@lam 1,3,5,7,9 -- ZNE noisy expectation values

  Derived physics features (5, engineered in load_and_split):
  [17] survival_combined  -- product of all channel gammas at lambda=1
  [18] log_survival       -- log of above; linear in noise-rate space
  [19] zne_extrap         -- Richardson 2-pt: (3*E1 - E3) / 2
  [20] zne_linear         -- linear OLS extrapolation to lambda=0 over all 5 scales
  [21] zne_quad           -- Richardson quadratic: (15*E1 - 10*E3 + 3*E5) / 8

Noise Model (6 channels)
------------------------
  1. Non-Markovian depolarizing: exp(-(eps*N)^beta_nm)  [stretched exponential]
  2. Amplitude damping:          exp(-t*N / T1)          [energy relaxation]
  3. Phase damping:              exp(-t*N / T2)          [dephasing]
  4. Leakage to |2>:             (1-p_leak)^N            [subspace escape]
  5. Crosstalk ZZ:               exp(-p_xtalk*pairs*d*lam) [quadratic in n]
  6. Readout/SPAM:               (1-2*p_readout)         [constant, breaks ZNE!]

  E_noisy(lambda) = (1-2*p_ro) * [E_ideal * prod(gammas) + bias_T1] + shot_noise

Dataset Filtering
-----------------
Samples where survival_combined < MIN_SURVIVAL are dropped before training.
These samples have E_noisy ≈ 0 regardless of E_ideal (information destroyed)
and are physically unlearnable. Retaining them only confuses the model.

Physics-Informed Loss
---------------------
Total loss = MSE(pred, ideal) + lambda_mono * monotonicity_penalty
                              + lambda_bndry * boundary_penalty

Monotonicity constraint: under the combined noise model the expectation value
magnitude must decrease as the noise scale factor increases:

    |E(lambda=1)| >= |E(lambda=3)| >= |E(lambda=5)| >= ... >= |E(lambda=9)|

Boundary penalty penalises predictions outside [-1, 1]:

    boundary_penalty = mean(ReLU(|pred| - 1))

Dataset
-------
    dataset/unmitigated_dataset.npz
    X : float32 (N, 17) -- [num_qubits, depth, error_rate, T1_us, T2_us,
                            gate_time_ns, p_damp, p_deph, p_leak, p_readout,
                            p_xtalk, beta_nm, exp@1..9]
    y : float32 (N,)    -- ideal expectation value (target)

Usage
-----
    python mitigation_ai/mlp.py

    # Or with custom hyper-parameters:
    python mitigation_ai/mlp.py --epochs 200 --lr 3e-4 --batch 256

Outputs (saved to mitigation_ai/outputs/)
-----------------------------------------
    best_mlp_model.pth     -- best checkpoint (lowest val loss)
    final_mlp_model.pth    -- model after all epochs
    training_curves.png    -- loss, MAE, and R² curves
    predictions_scatter.png-- predicted vs ideal scatter on test set
    training_log.csv       -- per-epoch metrics
"""


import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
from torch.optim.swa_utils import AveragedModel, SWALR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR    = Path(__file__).resolve().parent.parent          # project root
DATASET_NPZ = ROOT_DIR / "dataset" / "unmitigated_dataset.npz"
OUTPUT_DIR  = Path(__file__).resolve().parent / "outputs" / "residual_mlp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters (overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS      = 20
DEFAULT_LR          = 1e-3
DEFAULT_BATCH       = 1024
DEFAULT_SEED        = 42
DEFAULT_VAL_SPLIT   = 0.20    # 15 % validation
DEFAULT_TEST_SPLIT  = 0.10     # 10 % held-out test
DEFAULT_LAMBDA_MONO = 0.1   # weight of monotonicity penalty
DEFAULT_LAMBDA_BNDRY= 0.10     # weight of boundary [-1, 1] penalty
DEFAULT_LEAKY_NEG   = 0.1      # LeakyReLU negative slope
DEFAULT_DROPOUT     = 0.05     # light regularisation (10 %) on smaller dataset
DEFAULT_WEIGHT_DECAY= 1e-4     # L2 regularisation

# Minimum noise-survival factor for a sample to be included in training.
# With 6 noise channels (including quadratic crosstalk and non-Markovian effects),
# combined survival is lower than with depolarizing alone. Threshold lowered to 0.10
# to retain challenging-but-learnable samples at larger qubit/depth scales.
MIN_SURVIVAL = 0.10

# Raw feature column indices in the dataset (17 columns total)
IDX_NQUBITS      = 0   # num_qubits
IDX_DEPTH        = 1   # depth
IDX_ERROR        = 2   # error_rate
IDX_T1           = 3   # T1_us
IDX_T2           = 4   # T2_us
IDX_GATE_TIME    = 5   # gate_time_ns
IDX_P_DAMP       = 6   # p_damp
IDX_P_DEPH       = 7   # p_deph
IDX_P_LEAK       = 8   # p_leak     (leakage to |2>)
IDX_P_READOUT    = 9   # p_readout  (SPAM / readout confusion)
IDX_P_XTALK      = 10  # p_xtalk    (ZZ crosstalk per pair per layer)
IDX_BETA_NM      = 11  # beta_nm    (non-Markovian stretch exponent)

# ZNE exp-value columns start at index 12 (shifted by 4 new metadata cols)
NOISY_EXP_INDICES = [12, 13, 14, 15, 16]  # exp@lam1, lam3, lam5, lam7, lam9


# ===========================================================================
# Model Definition
# ===========================================================================

class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(self.fc(x))
        out = out + self.proj(x)
        out = self.act(out)
        return self.drop(out)


class QuantumMitigationMLP(nn.Module):
    def __init__(
        self,
        input_dim: int   = 40,
        hidden1: int     = 512,
        hidden2: int     = 256,
        hidden3: int     = 128,
        hidden4: int     = 64,
        dropout: float   = DEFAULT_DROPOUT,
    ):
        super().__init__()

        self.zne_conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.conv_act = nn.GELU()

        # Shared feature extractor (shallower)
        self.shared_net = nn.Sequential(
            ResBlock(input_dim, hidden1, dropout),
            ResBlock(hidden1, hidden2, dropout),
        )
        
        # Dedicated branch for the Ideal (Main) Prediction
        self.main_branch = nn.Sequential(
            ResBlock(hidden2, hidden3, dropout),
            ResBlock(hidden3, hidden4, dropout),
            nn.Linear(hidden4, 1)
        )
        
        # Dedicated branch for the Noisy (Auxiliary) Predictions
        self.aux_branch = nn.Sequential(
            ResBlock(hidden2, hidden3, dropout),
            ResBlock(hidden3, hidden4, dropout),
            nn.Linear(hidden4, 5)
        )

        self._init_weights()

    # ... keep _init_weights the same ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        static_features = torch.cat([x[:, :12], x[:, 17:]], dim=1)
        zne_seq = x[:, 12:17].unsqueeze(1) 
        zne_quad = x[:, 21]

        conv_out = self.conv_act(self.zne_conv(zne_seq)) 
        conv_out_flat = conv_out.flatten(start_dim=1)    

        combined = torch.cat([static_features, conv_out_flat], dim=1) 
        
        # Pass through shared trunk
        shared_features = self.shared_net(combined)
        
        # Split into independent branches
        main_out = self.main_branch(shared_features) + zne_quad.unsqueeze(1)
        aux_out = self.aux_branch(shared_features)
        
        return torch.cat([main_out, aux_out], dim=1)
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                nn.init.zeros_(m.bias)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ===========================================================================
# Physics-Informed Loss
# ===========================================================================

class PhysicsInformedLoss(nn.Module):
    """
    Combined loss for quantum error mitigation.

    L_total = MSE(y_pred, y_ideal)
            + lambda_mono  * monotonicity_penalty(X_noisy)
            + lambda_bndry * boundary_penalty(y_pred)

    Monotonicity Penalty (with shot-noise tolerance)
    ------------------------------------------------
    For the ZNE auxiliary predictions at increasing scale factors, we
    expect the *absolute* values to decrease with scale:

        |E(lam=1)| >= |E(lam=3)| >= ... >= |E(lam=9)|

    Violations are penalised only when they exceed a shot-noise-derived
    margin, preventing the network from being repelled away from E=0:

        diff_k  = |E(lam_{k+1})| - |E(lam_k)|
        margin  = mono_margin_scale * shot_noise_std   (per sample)
        penalty = mean(ReLU(diff_k - margin) * signal_envelope)

    where signal_envelope = tanh(|E(lam=1)| / signal_scale) fades the
    penalty smoothly to zero when the predicted signal is small, further
    preventing the model from learning to avoid E≈0.

    Boundary Penalty
    ----------------
    Quantum expectation values of Pauli observables must lie in [-1, 1].
    Any prediction outside this interval is penalised:
        boundary = mean(ReLU(|y_pred| - 1))

    Parameters
    ----------
    lambda_mono       : Weight for monotonicity penalty.
    lambda_bndry      : Weight for boundary penalty.
    noisy_indices     : Indices in X for noise-scaled exp. values.
    shot_noise_idx    : Column index of shot_noise_variance in X (default 23).
    mono_margin_scale : Multiplier on shot_noise_std used as tolerance margin
                        before a monotonicity violation is penalised (default 1.0).
    signal_scale      : Controls how quickly the signal envelope fades near E=0;
                        envelope = tanh(|E_pred| / signal_scale) (default 0.1).
    """

    def __init__(
        self,
        lambda_mono: float       = DEFAULT_LAMBDA_MONO,
        lambda_bndry: float      = DEFAULT_LAMBDA_BNDRY,
        noisy_indices: list      = None,
    ):
        super().__init__()
        self.lambda_mono       = lambda_mono
        self.lambda_bndry      = lambda_bndry
        self.noisy_indices     = noisy_indices if noisy_indices is not None else NOISY_EXP_INDICES
        self.huber             = nn.HuberLoss(delta=0.1)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        X:      torch.Tensor,
    ) -> tuple:
        # 1. Main Huber reconstruction loss
        main_loss = self.huber(y_pred[:, 0], y_true)

        # 2. Auxiliary Huber loss for ZNE outputs
        aux_loss = self.huber(y_pred[:, 1:], X[:, self.noisy_indices])

        # 3. Boundary Penalty
        boundary_penalty = torch.relu(y_pred.abs() - 1.0).mean()

        # 4. Monotonicity penalty eradicated
        mono_penalty = torch.tensor(0.0, device=y_pred.device)

        total_loss = main_loss + 0.2 * aux_loss + self.lambda_bndry * boundary_penalty

        return total_loss, main_loss, mono_penalty, aux_loss


# ===========================================================================
# Data Loading & Preprocessing
# ===========================================================================

def load_and_split(
    npz_path: Path,
    val_split: float    = DEFAULT_VAL_SPLIT,
    test_split: float   = DEFAULT_TEST_SPLIT,
    seed: int           = DEFAULT_SEED,
    min_survival: float = MIN_SURVIVAL,
    device: torch.device = None,
) -> tuple:
    """
    Load the dataset, engineer derived physics features, filter irrecoverable
    samples, apply StandardScaler to X (fitted on train only), and return
    TensorDatasets split into train / val / test.

    Raw dataset has 17 columns:
        [num_qubits, depth, error_rate, T1_us, T2_us, gate_time_ns,
         p_damp, p_deph, p_leak, p_readout, p_xtalk, beta_nm, exp@lam1..9]

    Derived features added (5 new columns -> 22 total)
    --------------------------------------------------
    survival_combined : product of all 6 channel attenuations at lambda=1
        gamma_dep_nm  = exp(-(eps*N)^beta_nm)
        gamma_T1      = exp(-t*N/T1)
        gamma_T2      = exp(-t*N/T2)
        gamma_leak    = (1-p_leak)^N
        gamma_xtalk   = exp(-p_xtalk * n_pairs * d)
        readout_factor = (1 - 2*p_readout)
        survival = gamma_dep_nm * gamma_T1 * gamma_T2 * gamma_leak
                 * gamma_xtalk * readout_factor

    log_survival    : log(survival_combined)  [linearises exponential decay]
    zne_extrap      : (3*E1 - E3) / 2         [Richardson 2-point]
    zne_linear      : OLS intercept at lambda=0 over all 5 scales
    zne_quad        : (15*E1 - 10*E3 + 3*E5) / 8  [Richardson quadratic]

    Returns
    -------
    train_ds, val_ds, test_ds, scaler, feature_names
    """
    data = np.load(npz_path, allow_pickle=True)
    X_raw = data["X"].astype(np.float32)         # (N, 17)
    y_raw = data["y"].astype(np.float32)         # (N,)
    feature_names = data["feature_names"].tolist()

    # ------------------------------------------------------------------
    # Derived features (computed from raw, unscaled values)
    # ------------------------------------------------------------------
    nq      = X_raw[:, IDX_NQUBITS  ].astype(np.float64)
    d       = X_raw[:, IDX_DEPTH    ].astype(np.float64)
    eps     = X_raw[:, IDX_ERROR    ].astype(np.float64)
    T1      = X_raw[:, IDX_T1       ].astype(np.float64)
    T2      = X_raw[:, IDX_T2       ].astype(np.float64)
    t_ns    = X_raw[:, IDX_GATE_TIME].astype(np.float64)
    t_us    = t_ns * 1e-3
    p_leak  = X_raw[:, IDX_P_LEAK   ].astype(np.float64)
    p_ro    = X_raw[:, IDX_P_READOUT].astype(np.float64)
    p_xt    = X_raw[:, IDX_P_XTALK  ].astype(np.float64)
    beta_nm = X_raw[:, IDX_BETA_NM  ].astype(np.float64)

    N = nq * d   # total qubit-gate operations at lambda=1

    # Non-Markovian depolarizing survival
    eps_N      = eps * N
    gamma_dep  = np.exp(-(np.where(eps_N > 0, eps_N, 0.0) ** beta_nm))

    # T1 and T2 survival
    gamma_T1   = np.exp(-t_us * N / np.maximum(T1, 1e-9))
    p_decay_T1 = (1.0 - gamma_T1).astype(np.float32)
    gamma_T2   = np.exp(-t_us * N / np.maximum(T2, 1e-9))

    # Leakage survival
    gamma_leak = (1.0 - p_leak) ** N

    # Crosstalk survival (scales as n*(n-1)/2 qubit pairs)
    n_pairs     = nq * (nq - 1) / 2.0
    gamma_xtalk = np.exp(-p_xt * n_pairs * d)

    # Readout/SPAM factor (constant, does not scale with lambda)
    readout_fac = 1.0 - 2.0 * p_ro

    # Combined survival
    survival_combined = (
        gamma_dep * gamma_T1 * gamma_T2 * gamma_leak * gamma_xtalk * readout_fac
    ).astype(np.float32)
    log_surv = np.log(np.clip(np.abs(survival_combined), 1e-10, 1.0)).astype(np.float32)

    # ZNE noisy expectation values (columns 12..16)
    e_noisy    = X_raw[:, NOISY_EXP_INDICES]   # (N, 5)
    e1, e3, e5 = e_noisy[:, 0], e_noisy[:, 1], e_noisy[:, 2]

    # 2-point Richardson
    zne_extrap = ((3.0 * e1 - e3) / 2.0).astype(np.float32)

    # Linear OLS extrapolation to lambda=0
    lam_dev    = np.array([-4., -2., 0., 2., 4.], dtype=np.float32)
    e_mean     = e_noisy.mean(axis=1)
    b_coef     = (e_noisy * lam_dev).sum(axis=1) / 40.0
    zne_linear = (e_mean - b_coef * 5.0).astype(np.float32)

    # Quadratic Richardson
    zne_quad = ((15.0 * e1 - 10.0 * e3 + 3.0 * e5) / 8.0).astype(np.float32)

    # Expected Shot Noise Feature
    shot_noise_variance = (np.maximum(0.0, 1.0 - zne_extrap**2) / 8192.0).astype(np.float32)

    # Signal-to-Noise Ratio (SNR)
    snr = (np.abs(zne_quad) / np.sqrt(shot_noise_variance + 1e-8)).astype(np.float32)

    # Append all derived features -- shape (N, 25)
    X_aug = np.concatenate(
        [X_raw,
         survival_combined[:, None], log_surv[:, None], zne_extrap[:, None],
         zne_linear[:, None], zne_quad[:, None], p_decay_T1[:, None],
         shot_noise_variance[:, None], snr[:, None]],
        axis=1,
    )
    aug_names = feature_names + [
        "survival_combined", "log_survival", "zne_extrap",
        "zne_linear", "zne_quad", "p_decay_T1", "shot_noise_variance", "SNR"
    ]

    # ------------------------------------------------------------------
    # Filter: drop samples where combined noise destroyed the signal
    # ------------------------------------------------------------------
    mask   = np.abs(survival_combined) >= min_survival
    X_aug  = X_aug[mask]
    y_raw  = y_raw[mask]
    N_kept = int(mask.sum())
    print(f"  Survival filter (>= {min_survival}): kept {N_kept:,} / {len(mask):,} samples "
          f"({100.0 * N_kept / len(mask):.1f}%)")

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------
    N_data  = len(y_raw)
    n_test  = int(N_data * test_split)
    n_val   = int(N_data * val_split)
    n_train = N_data - n_test - n_val

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N_data)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train : n_train + n_val]
    test_idx  = idx[n_train + n_val :]

    # Indices of bounded expectation values to exclude from scaling
    # (ZNE values at 12-16, derived extrapolations at 19-21, and shot noise at 23)
    zne_idx = [12, 13, 14, 15, 16, 19, 20, 21, 23]
    static_idx = [i for i in range(X_aug.shape[1]) if i not in zne_idx]

    scaler = StandardScaler()
    scaler.fit(X_aug[train_idx][:, static_idx])

    def scale_dataset(X):
        X_scaled = np.empty_like(X, dtype=np.float32)
        X_scaled[:, static_idx] = scaler.transform(X[:, static_idx]).astype(np.float32)
        X_scaled[:, zne_idx] = X[:, zne_idx].astype(np.float32)
        return X_scaled

    X_train = scale_dataset(X_aug[train_idx])
    X_val   = scale_dataset(X_aug[val_idx])
    X_test  = scale_dataset(X_aug[test_idx])

    y_train = y_raw[train_idx]
    y_val   = y_raw[val_idx  ]
    y_test  = y_raw[test_idx ]

    def to_tensor(arr):
        t = torch.from_numpy(arr)
        return t.to(device) if device else t

    train_ds = TensorDataset(to_tensor(X_train), to_tensor(y_train))
    val_ds   = TensorDataset(to_tensor(X_val),   to_tensor(y_val))
    test_ds  = TensorDataset(to_tensor(X_test),  to_tensor(y_test))

    print(f"  Train : {len(train_ds):,} samples")
    print(f"  Val   : {len(val_ds):,} samples")
    print(f"  Test  : {len(test_ds):,} samples")

    return train_ds, val_ds, test_ds, scaler, aug_names


# ===========================================================================
# Training & Evaluation
# ===========================================================================

def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Coefficient of determination R²."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1.0 - ss_res / ss_tot).item()


def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  PhysicsInformedLoss,
    device:     torch.device,
    scaler      = None,
    scheduler   = None,
) -> dict:
    model.train()
    total_loss = mse_sum = mono_sum = aux_sum = 0.0
    all_pred, all_true = [], []
    use_amp = (device.type == "cuda" and scaler is not None)

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device.type, enabled=use_amp):
            y_pred = model(X_batch)
            loss, mse, mono, aux = criterion(y_pred, y_batch, X_batch)

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

        bs = len(y_batch)
        total_loss += loss.item() * bs
        mse_sum    += mse.item()  * bs
        mono_sum   += mono.item() * bs
        aux_sum    += aux.item()  * bs

        all_pred.append(y_pred[:, 0].detach())
        all_true.append(y_batch.detach())

    n = len(loader.dataset)
    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    mae  = (all_pred - all_true).abs().mean().item()
    r2   = r2_score(all_pred, all_true)
    acc  = ((all_pred - all_true).abs() <= 0.05).float().mean().item() * 100.0

    return {
        "loss"  : total_loss / n,
        "mse"   : mse_sum    / n,
        "mono"  : mono_sum   / n,
        "aux"   : aux_sum    / n,
        "mae"   : mae,
        "r2"    : r2,
        "acc"   : acc,
    }


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: PhysicsInformedLoss,
    device:    torch.device,
    use_amp:   bool = False,
) -> dict:
    model.eval()
    total_loss = mse_sum = mono_sum = aux_sum = 0.0
    all_pred, all_true = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            y_pred = model(X_batch)
            loss, mse, mono, aux = criterion(y_pred, y_batch, X_batch)

        bs = len(y_batch)
        total_loss += loss.item() * bs
        mse_sum    += mse.item()  * bs
        mono_sum   += mono.item() * bs
        aux_sum    += aux.item()  * bs

        all_pred.append(y_pred[:, 0])
        all_true.append(y_batch)

    n = len(loader.dataset)
    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    mae = (all_pred - all_true).abs().mean().item()
    r2  = r2_score(all_pred, all_true)
    acc = ((all_pred - all_true).abs() <= 0.05).float().mean().item() * 100.0

    return {
        "loss"  : total_loss / n,
        "mse"   : mse_sum    / n,
        "mono"  : mono_sum   / n,
        "aux"   : aux_sum    / n,
        "mae"   : mae,
        "r2"    : r2,
        "acc"   : acc,
        "pred"  : all_pred.cpu().numpy(),
        "true"  : all_true.cpu().numpy(),
    }


# ===========================================================================
# Plotting
# ===========================================================================

def plot_training_curves(history: dict, out_dir: Path) -> None:
    """Save a 2x2 grid of training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    palette = {"train": "#7ec8e3", "val": "#f4a261"}

    axes_cfg = [
        (0, 0, "loss",  "Total PI-Loss",   True ),
        (0, 1, "mse",   "MSE Loss",        True ),
        (1, 0, "mae",   "MAE",             False),
        (1, 1, "r2",    "R² Score",        False),
    ]

    for row, col, key, title, logy in axes_cfg:
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        tr_vals  = history[f"train_{key}"]
        val_vals = history[f"val_{key}"]

        ax.plot(epochs, tr_vals,  color=palette["train"], lw=1.8,
                label="Train",      alpha=0.9)
        ax.plot(epochs, val_vals, color=palette["val"],   lw=1.8,
                label="Validation", alpha=0.9, linestyle="--")

        if logy:
            ax.set_yscale("log")

        ax.set_title(title, color="white", fontsize=12, pad=8)
        ax.set_xlabel("Epoch",  color="#aaaacc", fontsize=10)
        ax.tick_params(colors="#aaaacc")
        ax.legend(fontsize=9, framealpha=0.2, labelcolor="white")
        ax.grid(True, alpha=0.15, color="#ffffff")

    fig.suptitle("Physics-Informed MLP — Training Curves",
                 color="white", fontsize=15, y=1.01, fontweight="bold")

    path = out_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_predictions(pred: np.ndarray, true: np.ndarray, out_dir: Path) -> None:
    """Save a prediction scatter plot against the ideal diagonal."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    sc = ax.scatter(true, pred, c=np.abs(pred - true), cmap="plasma",
                    alpha=0.45, s=12, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("|Error|", color="#aaaacc")
    cbar.ax.yaxis.set_tick_params(color="#aaaacc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaacc")

    lims = [-1.05, 1.05]
    ax.plot(lims, lims, color="#7ec8e3", lw=1.5, linestyle="--",
            label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)

    r2  = r2_score(torch.tensor(pred), torch.tensor(true))
    mae = float(np.abs(pred - true).mean())
    ax.set_title(f"Test Set: Predicted vs Ideal\nR²={r2:.4f}   MAE={mae:.5f}",
                 color="white", fontsize=12)
    ax.set_xlabel("Ideal Expectation Value",   color="#aaaacc", fontsize=11)
    ax.set_ylabel("Mitigated (Predicted)",     color="#aaaacc", fontsize=11)
    ax.tick_params(colors="#aaaacc")
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="white")
    ax.grid(True, alpha=0.12, color="#ffffff")

    path = out_dir / "predictions_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


# ===========================================================================
# Main Training Pipeline
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Physics-Informed MLP for Quantum Error Mitigation"
    )
    p.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",            type=float, default=DEFAULT_LR)
    p.add_argument("--batch",         type=int,   default=DEFAULT_BATCH)
    p.add_argument("--seed",          type=int,   default=DEFAULT_SEED)
    p.add_argument("--val_split",     type=float, default=DEFAULT_VAL_SPLIT)
    p.add_argument("--test_split",    type=float, default=DEFAULT_TEST_SPLIT)
    p.add_argument("--lambda_mono",   type=float, default=DEFAULT_LAMBDA_MONO)
    p.add_argument("--lambda_bndry",  type=float, default=DEFAULT_LAMBDA_BNDRY)
    p.add_argument("--leaky_neg",     type=float, default=DEFAULT_LEAKY_NEG)
    p.add_argument("--dropout",       type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--weight_decay",  type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--dataset",       type=str,   default=str(DATASET_NPZ))
    p.add_argument("--min_survival",  type=float, default=MIN_SURVIVAL,
                   help="Min noise-survival factor for sample inclusion (default 0.20)")
    return p.parse_args()


def main():
    args   = parse_args()

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Enable fast mode optimizations by default
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  [Optimization] Fast mode enabled: TF32 and cuDNN benchmark ON.")

    # ------------------------------------------------------------------
    # Device selection: CUDA -> MPS -> CPU
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n  [Device] CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n  [Device] Apple MPS")
    else:
        device = torch.device("cpu")
        print(f"\n  [Device] CPU  (no GPU detected -- install CUDA-enabled PyTorch to use GPU)")

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Physics-Informed MLP — Quantum Error Mitigation")
    print("=" * 65)
    print(f"  Dataset      : {args.dataset}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size   : {args.batch}")
    print(f"  Min survival : {args.min_survival}")
    print(f"  lambda_mono  : {args.lambda_mono}")
    print(f"  lambda_bndry : {args.lambda_bndry}")
    print(f"  LeakyReLU a  : {args.leaky_neg}")
    print(f"  Dropout      : {args.dropout}")
    print(f"  Weight decay : {args.weight_decay}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading & splitting dataset ...")
    train_ds, val_ds, test_ds, scaler, feature_names = load_and_split(
        npz_path     = Path(args.dataset),
        val_split    = args.val_split,
        test_split   = args.test_split,
        seed         = args.seed,
        min_survival = args.min_survival,
        device       = device,
    )

    # Diagnostic Histogram Plot
    all_y_train = train_ds.tensors[1].cpu().numpy()
    all_y_val   = val_ds.tensors[1].cpu().numpy()
    all_y_test  = test_ds.tensors[1].cpu().numpy()
    all_y = np.concatenate([all_y_train, all_y_val, all_y_test])

    plt.figure(figsize=(8, 6))
    plt.hist(all_y, bins=100, color='#7ec8e3', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Ideal Expectation Values")
    plt.xlabel("Ideal Expectation Value (E)")
    plt.ylabel("Frequency")
    
    hist_path = OUTPUT_DIR / "dataset_distribution.png"
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved diagnostic histogram: {hist_path}")

    num_workers = 0   # 0 = main-process loading (safest on Windows)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=num_workers)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\n[2/4] Building model ...")
    model = QuantumMitigationMLP(
        input_dim = 40,
        hidden1   = 512,
        hidden2   = 256,
        hidden3   = 128,
        hidden4   = 64,
        dropout   = args.dropout,
    ).to(device)

    print(f"\n  Model architecture:")
    print(f"  {model}")
    print(f"\n  Trainable parameters: {model.count_parameters():,}")

    import os
    if hasattr(torch, "compile") and device.type == "cuda":
        if os.name != "nt":
            print("\n  [Optimization] Compiling model with torch.compile()...")
            model = torch.compile(model)
        else:
            print("\n  [Optimization] Skipping torch.compile() (Windows does not natively support Triton).")

    criterion = PhysicsInformedLoss(
        lambda_mono  = args.lambda_mono,
        lambda_bndry = args.lambda_bndry,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        fused        = True if device.type == "cuda" else False,
    )
    if device.type == "cuda":
        print("  [Optimization] Fused AdamW enabled.")

    # OneCycleLR schedule for optimal training dynamics
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15
    )

    # Stochastic Weight Averaging (SWA)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-4)
    swa_start = int(0.75 * args.epochs)

    use_amp = (device.type == "cuda")
    grad_scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("  [Optimization] Automatic Mixed Precision (AMP) enabled.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n[3/4] Training for {args.epochs} epochs ...")
    print("-" * 85)
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | "
          f"{'Train MAE':>10} | {'Val MAE':>9} | {'Val R²':>8} | "
          f"{'Tr Acc%':>7} | {'Val Acc%':>8} | LR")
    print("-" * 85)

    history = {k: [] for k in
               ["train_loss","val_loss","train_mse","val_mse",
                "train_mae","val_mae","train_r2","val_r2",
                "train_acc","val_acc"]}

    best_val_loss    = float("inf")
    patience_counter = 0
    best_ckpt_path   = OUTPUT_DIR / "best_mlp_model.pth"
    log_path       = OUTPUT_DIR / "training_log.csv"

    # CSV logger
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","val_loss","train_mae","val_mae",
                         "train_r2","val_r2","lr"])

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Standard scheduler only runs before swa_start
        current_scheduler = scheduler if epoch < swa_start else None
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=grad_scaler, scheduler=current_scheduler)
        
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            
        vl = evaluate(model, val_loader, criterion, device, use_amp=use_amp)

        current_lr = optimizer.param_groups[0]['lr']

        for k in ["loss","mse","mae","r2","acc"]:
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(vl[k])

        # Save best checkpoint
        if vl["loss"] < best_val_loss:
            best_val_loss = vl["loss"]
            torch.save({
                "epoch"         : epoch,
                "model_state"   : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss"      : best_val_loss,
                "args"          : vars(args),
                "feature_names" : feature_names,
                "scaler_mean"   : scaler.mean_.tolist(),
                "scaler_scale"  : scaler.scale_.tolist(),
            }, best_ckpt_path)
            star = " *"
            patience_counter = 0
        else:
            star = ""
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"{epoch:>6} | {tr['loss']:>11.6f} | {vl['loss']:>10.6f} | "
                  f"{tr['mae']:>10.6f} | {vl['mae']:>9.6f} | {vl['r2']:>8.4f} | "
                  f"{tr['acc']:>6.1f}% | {vl['acc']:>7.1f}% | "
                  f"{current_lr:.2e}{star}")

        # Append to CSV log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr["loss"], vl["loss"],
                             tr["mae"], vl["mae"], tr["r2"], vl["r2"],
                             current_lr])

        if patience_counter >= 15:
            print(f"\n  [Early Stopping] Validation loss hasn't improved for 15 epochs. Breaking at epoch {epoch}.")
            break

    total_time = time.time() - t_start
    print("-" * 85)
    print(f"  Training complete in {total_time:.1f}s "
          f"({total_time/args.epochs:.2f}s/epoch)")
    print(f"  Best val loss: {best_val_loss:.6f}  -->  {best_ckpt_path}")

    # Save final model
    final_path = OUTPUT_DIR / "final_mlp_model.pth"
    torch.save({
        "epoch"          : args.epochs,
        "model_state"    : model.state_dict(),
        "args"           : vars(args),
        "feature_names"  : feature_names,
        "scaler_mean"    : scaler.mean_.tolist(),
        "scaler_scale"   : scaler.scale_.tolist(),
    }, final_path)
    print(f"  Final model  : {final_path}")

    # ------------------------------------------------------------------
    # Test set evaluation (using SWA model)
    # ------------------------------------------------------------------
    print(f"\n[4/4] Updating SWA batch normalization and evaluating ...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    
    # Save SWA model
    swa_path = OUTPUT_DIR / "swa_mlp_model.pth"
    torch.save({
        "epoch"          : args.epochs,
        "model_state"    : swa_model.module.state_dict(),
        "args"           : vars(args),
        "feature_names"  : feature_names,
        "scaler_mean"    : scaler.mean_.tolist(),
        "scaler_scale"   : scaler.scale_.tolist(),
    }, swa_path)
    print(f"  SWA model    : {swa_path}")

    test_metrics = evaluate(swa_model, test_loader, criterion, device)

    print("\n" + "=" * 65)
    print("  TEST SET RESULTS")
    print("=" * 65)
    print(f"  Total PI-Loss  : {test_metrics['loss']:.6f}")
    print(f"  MSE Loss       : {test_metrics['mse']:.6f}")
    print(f"  MAE            : {test_metrics['mae']:.6f}")
    print(f"  R² Score       : {test_metrics['r2']:.6f}")
    print(f"  RMSE           : {np.sqrt(test_metrics['mse']):.6f}")
    print(f"  Accuracy (±.05): {test_metrics['acc']:.2f}%")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n  Generating plots ...")
    plot_training_curves(history, OUTPUT_DIR)
    plot_predictions(test_metrics["pred"], test_metrics["true"], OUTPUT_DIR)
    print(f"  Training log : {log_path}")
    print("  All outputs saved to:", OUTPUT_DIR)
    print(f"  Regression Prediction Accuracy (R-Squared Score) : {test_metrics['r2']:.6f}")
    print("\n" + "=" * 65)

    """
    print("=" * 65)
    print("\n  To load the best model for inference:")
    print("    import torch")
    print("    ckpt  = torch.load('mitigation_ai/outputs/best_mlp_model.pth')")
    print("    model = QuantumMitigationMLP(input_dim=13)")
    print("    model.load_state_dict(ckpt['model_state'])")
    print("    model.eval()")
    print("=" * 65)
    """

# Extract true values directly from the PyTorch TensorDatasets
    all_y_train = train_ds.tensors[1].cpu().numpy()
    all_y_val   = val_ds.tensors[1].cpu().numpy()
    all_y_test  = test_ds.tensors[1].cpu().numpy()
    
    # Combine them into the single array our plot expects
    all_y_true_values = np.concatenate([all_y_train, all_y_val, all_y_test])

    # Diagnostic Histogram of Ideal Expectation Values
    plt.figure(figsize=(8, 5))
    plt.hist(all_y_true_values, bins=100, color="#f4a261", edgecolor="black")
    plt.title("Distribution of Ideal Expectation Values")
    plt.xlabel("Ideal Expectation Value (E)")
    plt.ylabel("Frequency")
    plt.savefig(OUTPUT_DIR / "dataset_distribution.png")
    plt.close()
    
    print(f"  Saved diagnostic histogram to {OUTPUT_DIR / 'dataset_distribution.png'}")

if __name__ == "__main__":
    main()
