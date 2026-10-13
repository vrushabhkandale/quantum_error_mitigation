"""
mlp.py
======
Physics-Informed Multi-Layer Perceptron (PI-MLP) for Quantum Error Mitigation.

Architecture (funnel):
    Input (dynamic)  --> Hidden-1 (2048, GELU ResBlock)
                     --> Hidden-2 (2048, GELU ResBlock)
                     --> Hidden-3 (1024, GELU ResBlock)
                     --> [Main branch]  Output (1, Linear)
                     --> [Aux  branch]  Output (5, Linear) -- noisy ZNE predictions

Input features (25 total after pruning)
-----------------------------------------
  Raw features from dataset (21, after pruning 4 zero-var/collinear cols):
  [0]  num_qubits
  [1]  depth
  [2]  error_1q          -- per-gate 1Q depolarizing rate
  [3]  error_2q          -- per-gate 2Q depolarizing rate
  [4]  p_damp_1q         -- amp. damping prob per 1Q gate
  [5]  p_deph_1q         -- dephasing prob per 1Q gate
  [6]  p_damp_2q         -- amp. damping prob per 2Q gate
  [7]  p_deph_2q         -- dephasing prob per 2Q gate
  [8]  p_leak            -- leakage to |2> prob per qubit per gate
  [9]  p_01              -- SPAM: P(measure 0 | state 1)
  [10] p_10              -- SPAM: P(measure 1 | state 0)
  [11] p_xtalk           -- ZZ crosstalk dephasing rate
  [12] beta_nm           -- non-Markovian stretch exponent
  [13] theta_error       -- coherent over-rotation per gate (rad)
  [14] observable_alignment -- alignment with Z-basis (+1/-1)
  [15] survival_factor   -- pre-computed combined survival at lambda=1
  [16-20] exp@lam 1,3,5,7,9 -- ZNE noisy expectation values (Conv1D input)

  Derived physics features (4, engineered in load_and_split):
  [21] survival_combined  -- re-computed product of all channel gammas at lambda=1
  [22] log_survival       -- log of above; linear in noise-rate space
  [23] p_decay_T1         -- (1 - gamma_T1) at lambda=1
  [24] shot_noise_variance -- (1 - E1^2) / 8192

  NOTE: zne_extrap, zne_linear, zne_quad are intentionally REMOVED.
  The model must learn to extrapolate from raw ZNE curves and physics
  features, not from pre-computed analytic extrapolation shortcuts.

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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR    = Path(__file__).resolve().parent.parent          # project root
DATASET_NPZ = ROOT_DIR / "dataset" / "vector_dataset.npz"
OUTPUT_DIR  = Path(__file__).resolve().parent / "outputs" / "mlp_mitiq"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters (overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS      = 100

DEFAULT_LR          = 1e-3
DEFAULT_TRAIN_BATCH = 4096
DEFAULT_EVAL_BATCH  = 16384
DEFAULT_SEED        = 42
DEFAULT_VAL_SPLIT   = 0.20    # 15 % validation
DEFAULT_TEST_SPLIT  = 0.10     # 10 % held-out test
DEFAULT_LAMBDA_MONO = 0.1  # weight of monotonicity penalty (restored with residual learning)
DEFAULT_LAMBDA_BNDRY= 0.10     # weight of boundary [-1, 1] penalty
DEFAULT_LEAKY_NEG   = 0.1      # LeakyReLU negative slope
DEFAULT_DROPOUT     = 0.05     # light regularisation (10 %) on smaller dataset
DEFAULT_WEIGHT_DECAY= 1e-4     # L2 regularisation

# Minimum noise-survival factor for a sample to be included in training.
# With 6 noise channels (including quadratic crosstalk and non-Markovian effects),
# combined survival is lower than with depolarizing alone. Threshold lowered to 0.10
# to retain challenging-but-learnable samples at larger qubit/depth scales.
MIN_SURVIVAL = 0.1

# Raw feature column indices — vector_dataset.npz (25 columns)
# 0:num_qubits  1:depth  2:error_1q  3:error_2q
# 4:T1_us  5:T2_us  6:time_1q_ns  7:time_2q_ns
# 8:p_damp_1q  9:p_deph_1q  10:p_damp_2q  11:p_deph_2q
# 12:p_leak  13:p_01  14:p_10  15:p_xtalk  16:beta_nm
# 17:theta_error  18:observable_alignment  19:survival_factor
# 20:exp_val_scale_1  21:exp_val_scale_3  22:exp_val_scale_5
# 23:exp_val_scale_7  24:exp_val_scale_9
IDX_NQUBITS      = 0   # num_qubits
IDX_DEPTH        = 1   # depth
IDX_ERROR_1Q     = 2   # error_1q
IDX_ERROR_2Q     = 3   # error_2q
IDX_T1           = 4   # T1_us
IDX_T2           = 5   # T2_us
IDX_TIME_1Q      = 6   # time_1q_ns
IDX_TIME_2Q      = 7   # time_2q_ns
IDX_P_DAMP_1Q    = 8   # p_damp_1q
IDX_P_DEPH_1Q    = 9   # p_deph_1q
IDX_P_DAMP_2Q    = 10  # p_damp_2q
IDX_P_DEPH_2Q    = 11  # p_deph_2q
IDX_P_LEAK       = 12  # p_leak
IDX_P_01         = 13  # p_01  (SPAM: P(0|1))
IDX_P_10         = 14  # p_10  (SPAM: P(1|0))
IDX_P_XTALK      = 15  # p_xtalk
IDX_BETA_NM      = 16  # beta_nm
IDX_THETA_ERROR  = 17  # theta_error (coherent over-rotation)
IDX_OBS_ALIGN    = 18  # observable_alignment
# IDX 19 = survival_factor (pre-computed in dataset)

# ZNE exp-value columns: indices 20-24 (survival_factor occupies index 19)
NOISY_EXP_INDICES = [20, 21, 22, 23, 24]  # exp@lam1, lam3, lam5, lam7, lam9


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
    """
    Physics-informed MLP for quantum error mitigation with residual learning.

    Architecture changes vs previous version:
    - Conv1D (k=3) replaces the symmetric ZNEEncoder to restore sequential
      awareness of the ZNE decay curve (lam=1→3→5→7→9 is an ordered sequence
      on a decay manifold; mean/max pooling destroyed this inductive bias).
      Positional bias is now controlled via weight_decay (L2) rather than
      destroying the sequence order.
    - Model predicts the ERROR RESIDUAL (ideal - exp_val_scale_1), not the
      absolute ideal expectation value. This forces the network to explain
      WHY the noisy measurement differs from truth using physical features.
      At inference, exp_val_scale_1 is added back to recover the mitigated value.
    """
    def __init__(
        self,
        input_dim: int   = 46,  # 20 static + 4ch*5 conv output = 40; + static = 46
        hidden1: int     = 2048,
        hidden2: int     = 2048,
        hidden3: int     = 1024,
        hidden4: int     = 1024,
        hidden5: int     = 512,
        hidden6: int     = 256,
        dropout: float   = DEFAULT_DROPOUT,
    ):
        super().__init__()

        # Conv1D over the 5 ZNE scale factors — treats them as an ordered
        # time-series/decay curve with local neighbourhood structure.
        self.zne_conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.conv_act = nn.GELU()

        # Shared feature extractor
        self.shared_net = nn.Sequential(
            ResBlock(input_dim, hidden1, dropout),
            ResBlock(hidden1, hidden2, dropout),
            ResBlock(hidden2, hidden3, dropout),
        )
        
        # Main branch: predicts the ERROR RESIDUAL (ideal - exp_val_scale_1)
        self.main_branch = nn.Sequential(
            ResBlock(hidden3, hidden4, dropout),
            ResBlock(hidden4, hidden5, dropout),
            ResBlock(hidden5, hidden6, dropout),
            nn.Linear(hidden6, 1)
        )
        
        # Auxiliary branch: reconstructs all 5 noisy ZNE exp values
        self.aux_branch = nn.Sequential(
            ResBlock(hidden3, hidden4, dropout),
            ResBlock(hidden4, hidden5, dropout),
            ResBlock(hidden5, hidden6, dropout),
            nn.Linear(hidden6, 5)
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pruned feature layout (23 cols after removing T1_us, T2_us,
        # time_1q_ns, time_2q_ns, survival_factor):
        #   0-15  : static physics features
        #   16-20 : exp_val_scale_{1,3,5,7,9}  (ordered decay curve → Conv1D)
        #   21-22 : survival_combined, log_survival, p_decay_T1
        #
        # Conv1D captures the local slope/curvature of the ZNE decay.
        # Positional bias of exp_val_scale_1 at slot 0 is regulated by
        # weight_decay (L2) on the conv weights, not by destroying ordering.
        static_features = torch.cat([x[:, :16], x[:, 21:]], dim=1)  # 20 static cols
        zne_seq  = x[:, 16:21].unsqueeze(1)              # (batch, 1, 5)
        conv_out = self.conv_act(self.zne_conv(zne_seq))  # (batch, 4, 5)
        conv_flat = conv_out.flatten(start_dim=1)          # (batch, 20)

        combined = torch.cat([static_features, conv_flat], dim=1)
        
        shared_features = self.shared_net(combined)
        
        main_out = self.main_branch(shared_features)   # predicts residual
        aux_out  = self.aux_branch(shared_features)    # reconstructs ZNE curve
        
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
    Combined loss for quantum error mitigation (Residual Learning edition).

    Since the model now predicts the ERROR RESIDUAL (ideal - exp_val_scale_1)
    rather than the absolute ideal value, the target y_true passed here is
    also the residual. The final mitigated value is reconstructed externally
    by adding exp_val_scale_1 back to the prediction.

    L_total = HuberLoss(residual_pred, residual_true)          [main]
            + 0.2  * MSE(aux_pred, ZNE_exp_values)            [aux branch]
            + lambda_mono  * monotonicity_penalty(aux_pred)   [restored]
            + lambda_bndry * boundary_penalty(residual_pred)  [boundary]

    Huber loss (delta=0.1)
    ----------------------
    Behaves like MSE near zero but falls back to MAE for large residuals.
    This prevents extreme shot-noise spikes from dominating the gradient.

    Monotonicity Penalty (restored)
    --------------------------------
    The auxiliary branch predicts the 5 ZNE exp values. We penalise violations
    of |E(lam=1)| >= |E(lam=3)| >= ... >= |E(lam=9)| with a soft margin:
        diff_k  = |aux_pred[:,k+1]| - |aux_pred[:,k]|
        penalty = mean(ReLU(diff_k))    (no margin — clean constraint)

    Boundary Penalty
    ----------------
    Residual predictions can be large positive/negative; we only penalise
    the RECONSTRUCTED value (residual + exp_val_scale_1) if it exits [-1,1].
    Here we apply a soft penalty on the raw residual prediction instead.
    """

    def __init__(
        self,
        lambda_mono: float   = DEFAULT_LAMBDA_MONO,
        lambda_bndry: float  = DEFAULT_LAMBDA_BNDRY,
        noisy_indices: list  = None,
        huber_delta: float   = 0.1,
    ):
        super().__init__()
        self.lambda_mono   = lambda_mono
        self.lambda_bndry  = lambda_bndry
        self.noisy_indices = noisy_indices if noisy_indices is not None else NOISY_EXP_INDICES
        self.huber         = nn.HuberLoss(delta=huber_delta)
        self.mse           = nn.MSELoss()

    def forward(
        self,
        y_pred: torch.Tensor,   # (batch, 6): col0=residual, col1-5=ZNE aux
        y_true: torch.Tensor,   # (batch,): error residual (ideal - exp_val_scale_1)
        X:      torch.Tensor,   # (batch, F): feature matrix (unmodified)
    ) -> tuple:
        # 1. Main Huber loss on the error residual
        huber_loss = self.huber(y_pred[:, 0], y_true)

        # 2. Auxiliary MSE: aux branch must reconstruct the ZNE exp-value curve
        aux_mse = self.mse(y_pred[:, 1:], X[:, self.noisy_indices])

        # 3. Boundary penalty on residual prediction (soft)
        boundary_penalty = torch.relu(y_pred[:, 0].abs() - 1.0).mean()

        # 4. Monotonicity penalty (RESTORED):
        # The absolute magnitude of ZNE noisy exp values should decrease
        # monotonically with increasing scale factor (|E1| >= |E3| >= ... >= |E9|).
        aux_abs = y_pred[:, 1:].abs()  # (batch, 5)
        mono_violations = torch.relu(aux_abs[:, 1:] - aux_abs[:, :-1])  # 4 pairs
        mono_penalty = mono_violations.mean()

        total_loss = (huber_loss
                      + 0.2 * aux_mse
                      + self.lambda_mono  * mono_penalty
                      + self.lambda_bndry * boundary_penalty)

        return total_loss, huber_loss, mono_penalty, aux_mse


# ===========================================================================
# Survival Feature Dropout Dataset
# ===========================================================================

class TabularFeatureDropout(torch.utils.data.Dataset):
    """
    Generalised tabular feature dropout dataset.

    Accepts a dropout_schedule: a dict mapping a tuple of column indices to a
    per-group dropout probability.  During training, each group is independently
    zeroed out with its assigned probability, forcing the network to learn
    redundant representations of each information channel from the remaining
    physical features.

    Dropout schedule used in this project
    --------------------------------------
    Group                      | Cols (in aug matrix)  | prob
    exp_val_scale_1 (oracle)   | [19]                  | 0.50
    survival_combined /        |                       |
      log_survival (God Feats) | [24, 25]              | 0.40
    ZNE scales 3/5/7/9         | [20, 21, 22, 23]      | 0.25
      (collinear with scale_1) |                       |

    Validation and test splits always receive the full, unmasked feature set.

    Parameters
    ----------
    X               : float32 GPU tensor (N, F)
    y               : float32 GPU tensor (N,)
    is_train        : bool
    dropout_schedule: dict[tuple[int,...], float]  -- {col_indices: prob}
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        is_train: bool,
        dropout_schedule: dict = None,
        y_baseline: torch.Tensor = None,  # exp_val_scale_1 for residual reconstruction
    ):
        self.X                = X
        self.y                = y
        self.y_baseline       = y_baseline  # None in old code paths; exp1 for residual mode
        self.is_train         = is_train
        # Store as list of (col_indices_tensor, prob) for fast GPU iteration
        self.dropout_schedule = []
        if dropout_schedule and is_train:
            for cols, prob in dropout_schedule.items():
                col_t = torch.tensor(list(cols), dtype=torch.long, device=X.device)
                self.dropout_schedule.append((col_t, prob))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x    = self.X[idx]
        y    = self.y[idx]
        yb   = self.y_baseline[idx] if self.y_baseline is not None else torch.tensor(0.0)
        if self.dropout_schedule:
            cloned = False
            for col_indices, prob in self.dropout_schedule:
                if torch.rand(1).item() < prob:
                    if not cloned:
                        x = x.clone()  # clone once, lazily
                        cloned = True
                    x[col_indices] = 0.0
        return x, y, yb


# Backward-compat alias so GPUDataLoader type hint still resolves
SurvivalDropoutDataset = TabularFeatureDropout


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
    X_raw = data["X"].astype(np.float32)         # (N, 24)
    y_raw = data["y"].astype(np.float32)         # (N,)
    feature_names = data["feature_names"].tolist()

    # ------------------------------------------------------------------
    # Derived features (computed from raw, unscaled values)
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
    obs_align = X_raw[:, IDX_OBS_ALIGN  ].astype(np.float64)

    # Gate-set fractions (must match vector_data.py)
    FRAC_1Q = 2.0 / 3.0
    FRAC_2Q = 1.0 / 3.0

    t1q_us = t1q_ns * 1e-3
    t2q_us = t2q_ns * 1e-3

    N_total  = nq * d   # total qubit-gate operations at lambda=1
    t_eff_us = FRAC_1Q * N_total * t1q_us + FRAC_2Q * N_total * t2q_us

    # Blended 1Q/2Q effective depolarizing rate
    eps_eff = FRAC_1Q * e1q + FRAC_2Q * e2q

    # Non-Markovian depolarizing survival
    eps_N      = eps_eff * N_total
    gamma_dep  = np.exp(-(np.where(eps_N > 0, eps_N, 0.0) ** beta_nm))

    # T1 and T2 survival
    gamma_T1   = np.exp(-t_eff_us / np.maximum(T1, 1e-9))
    p_decay_T1 = (1.0 - gamma_T1).astype(np.float32)
    gamma_T2   = np.exp(-t_eff_us / np.maximum(T2, 1e-9))

    # Leakage survival
    gamma_leak = (1.0 - p_leak) ** N_total

    # Crosstalk survival (scales as n*(n-1)/2 qubit pairs)
    n_pairs     = nq * (nq - 1) / 2.0
    gamma_xtalk = np.exp(-p_xt * n_pairs * d)

    # Asymmetric SPAM factor — attenuation component only
    # (the bias (p_01 - p_10) is constant and embedded in ZNE exp values already)
    readout_fac = 1.0 - p_01 - p_10

    # Coherent over-rotation factor at lambda=1
    gamma_coh = np.cos(theta_err * N_total)

    # Combined survival
    survival_combined = (
        gamma_dep * gamma_T1 * gamma_T2 * gamma_leak
        * gamma_xtalk * gamma_coh * readout_fac
    ).astype(np.float32)
    log_surv = np.log(np.clip(np.abs(survival_combined), 1e-10, 1.0)).astype(np.float32)

    # ZNE noisy expectation values (columns 20-24 in the raw 25-col dataset)
    e_noisy = X_raw[:, NOISY_EXP_INDICES]   # (N, 5)

    # Expected Shot Noise Feature (based on scale-1 exp value)
    e1 = e_noisy[:, 0]  # exp_val_scale_1
    shot_noise_variance = ((1.0 - e1**2) / 8192.0).astype(np.float32)

    # Append derived features -- shape (N, 27)
    # Raw 25 cols + 2 derived = 27 total.
    # REMOVED: shot_noise_variance  -- derived purely from exp_val_scale_1^2,
    #           so it contains ZERO independent information and gives exp_val_scale_1
    #           a second proxy channel that inflates its permutation importance.
    # REMOVED: survival_factor is already in the raw dataset at col 19; we
    #           re-compute survival_combined from scratch for the loss/filter,
    #           but we prune the raw survival_factor from the feature matrix
    #           to avoid redundancy with the re-computed version.
    X_aug = np.concatenate(
        [X_raw,
         survival_combined[:, None], log_surv[:, None],
         p_decay_T1[:, None]],
        axis=1,
    )
    aug_names = feature_names + [
        "survival_combined", "log_survival", "p_decay_T1"
    ]

    # ------------------------------------------------------------------
    # Column pruning: drop zero-variance constants and collinear pairs.
    # Resolved by NAME so reordering the dataset never breaks the logic.
    #   time_1q_ns / time_2q_ns : constant gate times (low variance)
    #   T1_us / T2_us           : perfectly collinear with p_damp/p_deph
    #   survival_factor         : pre-computed in dataset but redundant with
    #                             survival_combined (re-computed here from raw
    #                             params with asymmetric SPAM + coherent term);
    #                             keeping both gives the model a shortcut that
    #                             distorts feature importance for all other cols
    # ------------------------------------------------------------------
    PRUNE_COLS = {"time_1q_ns", "time_2q_ns", "T1_us", "T2_us", "survival_factor"}
    prune_mask = [i for i, n in enumerate(aug_names) if n in PRUNE_COLS]
    if prune_mask:
        keep = [i for i in range(len(aug_names)) if i not in prune_mask]
        dropped = [aug_names[i] for i in prune_mask]
        print(f"  [Pruning] Dropping zero-variance / collinear columns: {dropped}")
        X_aug     = X_aug[:, keep]
        aug_names = [aug_names[i] for i in keep]
    print(f"  [Pruning] Active feature count: {len(aug_names)} columns.")

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
    # Train / val / test split (Accelerated with PyTorch)
    # ------------------------------------------------------------------
    import torch

    print("  Moving data to device for fast split and scaling...")
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

    # Indices of ZNE and derived columns NOT to StandardScale (bounded [-1,1]).
    # Resolved by NAME so column pruning above doesn't break the index math.
    # shot_noise_variance removed (no longer generated).
    no_scale_names = {
        "exp_val_scale_1", "exp_val_scale_3", "exp_val_scale_5",
        "exp_val_scale_7", "exp_val_scale_9",
    }
    zne_idx    = [i for i, n in enumerate(aug_names) if n in no_scale_names]
    static_idx = [i for i in range(X_aug_pt.shape[1]) if i not in zne_idx]

    # StandardScaler fitted ONLY on training split (zero data leakage)
    X_train_static = X_aug_pt[train_idx][:, static_idx]
    mean = X_train_static.mean(dim=0)
    std  = X_train_static.std(dim=0, unbiased=False)
    std[std == 0] = 1.0  # guard against exact-zero std

    def scale_dataset(X):
        X_scaled = torch.empty_like(X, dtype=torch.float32, device=device)
        X_scaled[:, static_idx] = (X[:, static_idx] - mean) / std
        X_scaled[:, zne_idx]    = X[:, zne_idx]
        return X_scaled

    X_train = scale_dataset(X_aug_pt[train_idx])
    X_val   = scale_dataset(X_aug_pt[val_idx])
    X_test  = scale_dataset(X_aug_pt[test_idx])

    # ------------------------------------------------------------------
    # Residual learning: y = ideal_exp_val - exp_val_scale_1
    # The model predicts HOW MUCH the noisy value needs to be corrected,
    # not what the absolute ideal value is. At inference, add exp_val_scale_1
    # back to recover the mitigated value.
    # We store exp_val_scale_1 (unscaled) alongside y for reconstruction.
    # ------------------------------------------------------------------
    exp1_col_idx = aug_names.index("exp_val_scale_1")
    exp1_train   = X_aug_pt[train_idx, exp1_col_idx]   # unscaled (ZNE cols not scaled)
    exp1_val     = X_aug_pt[val_idx,   exp1_col_idx]
    exp1_test    = X_aug_pt[test_idx,  exp1_col_idx]

    y_train_abs = y_raw_pt[train_idx]
    y_val_abs   = y_raw_pt[val_idx  ]
    y_test_abs  = y_raw_pt[test_idx ]

    # Residual targets
    y_train = y_train_abs - exp1_train
    y_val   = y_val_abs   - exp1_val
    y_test  = y_test_abs  - exp1_test
    print(f"  [Residual] Target = ideal - exp_val_scale_1  "
          f"(mean residual = {y_train.mean():.4f}, std = {y_train.std():.4f})")

    # ------------------------------------------------------------------
    # Build per-group dropout schedule (resolved by name, post-pruning)
    # ------------------------------------------------------------------
    def _cols(*names):
        """Return tuple of column indices for a set of feature names."""
        return tuple(i for i, n in enumerate(aug_names) if n in set(names))

    dropout_schedule = {
        # exp_val_scale_1 dropout REDUCED from 70% -> 15%.
        # With residual learning, exp_val_scale_1 IS the baseline; hiding it
        # 70% of the time forced the model to guess the baseline from scratch,
        # artificially capping accuracy. A light 15% dropout still prevents
        # complete reliance while keeping the signal available.
        _cols("exp_val_scale_1")                                      : 0.15,
        # survival_combined / log_survival are analytical "God Features" that
        # summarise all noise channels perfectly. 40% dropout forces the model
        # to build redundant pathways through the raw component features.
        _cols("survival_combined", "log_survival", "p_decay_T1")      : 0.40,
        # ZNE scales 3/5/7/9 carry the decay-curve shape: 25% dropout
        _cols("exp_val_scale_3", "exp_val_scale_5",
              "exp_val_scale_7", "exp_val_scale_9")                   : 0.25,
    }
    dropout_schedule = {k: v for k, v in dropout_schedule.items() if k}

    print(f"  [Dropout] Tabular feature dropout schedule:")
    for cols, prob in dropout_schedule.items():
        names_in_group = [aug_names[c] for c in cols]
        print(f"    prob={prob:.0%}  cols={names_in_group}")

    # Pass exp_val_scale_1 (unscaled) as y_baseline for residual reconstruction
    train_ds = TabularFeatureDropout(X_train, y_train, is_train=True,
                                     dropout_schedule=dropout_schedule,
                                     y_baseline=exp1_train)
    val_ds   = TabularFeatureDropout(X_val,   y_val,   is_train=False,
                                     y_baseline=exp1_val)
    test_ds  = TabularFeatureDropout(X_test,  y_test,  is_train=False,
                                     y_baseline=exp1_test)

    print(f"  Train : {len(train_ds):,} samples")
    print(f"  Val   : {len(val_ds):,} samples")
    print(f"  Test  : {len(test_ds):,} samples")

    class DummyScaler:
        def __init__(self, m, s):
            self.mean_ = m.cpu().numpy()
            self.scale_ = s.cpu().numpy()

    scaler = DummyScaler(mean, std)

    # Resolve ZNE exp-val column indices IN THE PRUNED FEATURE SPACE.
    zne_exp_names = {
        "exp_val_scale_1", "exp_val_scale_3", "exp_val_scale_5",
        "exp_val_scale_7", "exp_val_scale_9",
    }
    pruned_noisy_indices = [i for i, n in enumerate(aug_names) if n in zne_exp_names]
    print(f"  [Loss] Aux ZNE indices in pruned feature space: {pruned_noisy_indices}")

    return train_ds, val_ds, test_ds, scaler, aug_names, pruned_noisy_indices


# ===========================================================================
# GPU-Native Batch Iterator
# ===========================================================================

class GPUDataLoader:
    """
    Zero-copy GPU-native batch iterator.

    The tensors from TabularFeatureDropout are already resident on the GPU
    (placed there by load_and_split). This iterator slices them directly,
    eliminating all DataLoader worker processes, inter-process shared-memory
    transfers, and the "Sharing CUDA tensors" warnings from num_workers > 0.

    Feature dropout is applied here at batch granularity using vectorised GPU
    masks for each group in the dropout_schedule — far more efficient than the
    per-sample Python loop in __getitem__.

    Parameters
    ----------
    dataset    : TabularFeatureDropout -- holds .X, .y, .is_train, .dropout_schedule
    batch_size : int
    shuffle    : bool
    """

    def __init__(self, dataset: TabularFeatureDropout, batch_size: int, shuffle: bool):
        self.X                = dataset.X
        self.y                = dataset.y
        self.y_baseline       = dataset.y_baseline  # exp_val_scale_1 for residual mode
        self.is_train         = dataset.is_train
        self.dropout_schedule = dataset.dropout_schedule  # list of (col_tensor, prob)
        self.batch_size       = batch_size
        self.shuffle          = shuffle
        self.n                = len(dataset)
        self.dataset          = dataset   # expose for len(loader.dataset)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        device = self.X.device
        idx = (torch.randperm(self.n, device=device) if self.shuffle
               else torch.arange(self.n, device=device))

        for start in range(0, self.n, self.batch_size):
            batch_idx  = idx[start : start + self.batch_size]
            X_batch    = self.X[batch_idx]   # strided GPU view — zero copy
            y_batch    = self.y[batch_idx]
            yb_batch   = (self.y_baseline[batch_idx]
                          if self.y_baseline is not None
                          else torch.zeros_like(y_batch))

            if self.is_train and self.dropout_schedule:
                cloned = False
                for col_indices, prob in self.dropout_schedule:
                    drop_mask = torch.rand(len(batch_idx), device=device) < prob
                    if drop_mask.any():
                        if not cloned:
                            X_batch = X_batch.clone()  # clone once, lazily
                            cloned = True
                        for c in col_indices:
                            X_batch[drop_mask, c] = 0.0

            yield X_batch, y_batch, yb_batch


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
    all_pred, all_true = [], []   # reconstructed absolute values for metrics
    use_amp = (device.type == "cuda" and scaler is not None)

    for X_batch, y_batch, yb_batch in loader:
        X_batch  = X_batch.to(device)
        y_batch  = y_batch.to(device)   # residual target
        yb_batch = yb_batch.to(device)  # exp_val_scale_1 baseline

        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device.type, enabled=use_amp):
            y_pred = model(X_batch)   # col0 = predicted residual
            loss, huber, mono, aux = criterion(y_pred, y_batch, X_batch)

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
        total_loss += loss.item()  * bs
        mse_sum    += huber.item() * bs
        mono_sum   += mono.item()  * bs
        aux_sum    += aux.item()   * bs

        # Reconstruct absolute mitigated value for MAE/R² reporting
        pred_abs = (y_pred[:, 0].detach() + yb_batch).clamp(-1.0, 1.0)
        true_abs = y_batch.detach() + yb_batch
        all_pred.append(pred_abs)
        all_true.append(true_abs)

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
    all_pred, all_true = [], []   # reconstructed absolute values

    for X_batch, y_batch, yb_batch in loader:
        X_batch  = X_batch.to(device)
        y_batch  = y_batch.to(device)   # residual
        yb_batch = yb_batch.to(device)  # exp_val_scale_1 baseline

        with torch.autocast(device_type=device.type, enabled=use_amp):
            y_pred = model(X_batch)
            loss, huber, mono, aux = criterion(y_pred, y_batch, X_batch)

        bs = len(y_batch)
        total_loss += loss.item()  * bs
        mse_sum    += huber.item() * bs
        mono_sum   += mono.item()  * bs
        aux_sum    += aux.item()   * bs

        # Reconstruct absolute mitigated value: residual + exp_val_scale_1
        pred_abs = (y_pred[:, 0] + yb_batch).clamp(-1.0, 1.0)
        true_abs = y_batch + yb_batch
        all_pred.append(pred_abs)
        all_true.append(true_abs)

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

def compute_permutation_importance(
    model: nn.Module,
    loader: DataLoader,
    criterion: PhysicsInformedLoss,
    device: torch.device,
    feature_names: list,
    out_dir: Path
) -> None:
    """Computes and plots permutation feature importance on the test set."""
    print("\n  Computing Permutation Feature Importance ...")
    model.eval()
    
    # Collect the whole test set (loader yields X, y_residual, yb=exp_val_scale_1)
    X_all, y_all, yb_all = [], [], []
    for X, y, yb in loader:
        X_all.append(X)
        y_all.append(y)
        yb_all.append(yb)
    X_all  = torch.cat(X_all,  dim=0).to(device)
    y_all  = torch.cat(y_all,  dim=0).to(device)   # residual targets
    yb_all = torch.cat(yb_all, dim=0).to(device)   # exp_val_scale_1 baselines

    # Reconstruct absolute ideal values for importance scoring
    y_abs_all = y_all + yb_all  # ideal_exp_val in absolute units
    
    # Baseline: MSE of reconstructed absolute predictions vs absolute ideal
    with torch.no_grad():
        baseline_pred_res = model(X_all)[:, 0]              # predicted residual
        baseline_pred_abs = (baseline_pred_res + yb_all).clamp(-1.0, 1.0)
        baseline_mse = ((baseline_pred_abs - y_abs_all) ** 2).mean().item()
        
        importances = []
        for i in range(X_all.shape[1]):
            X_shuffled = X_all.clone()
            perm = torch.randperm(X_all.shape[0], device=device)
            X_shuffled[:, i] = X_all[perm, i]
            
            pred_res = model(X_shuffled)[:, 0]
            pred_abs = (pred_res + yb_all).clamp(-1.0, 1.0)
            mse = ((pred_abs - y_abs_all) ** 2).mean().item()
            importances.append(mse - baseline_mse)
            
    # Plotting
    importances = np.array(importances)
    idx = np.argsort(importances) # ascending for barh
    
    sorted_names = [feature_names[j] for j in idx]
    sorted_imps = importances[idx]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
        
    ax.barh(range(len(sorted_imps)), sorted_imps, color="#f4a261", edgecolor="#333355")
    ax.set_yticks(range(len(sorted_imps)))
    ax.set_yticklabels(sorted_names, color="#aaaacc")
    ax.tick_params(colors="#aaaacc")
    
    ax.set_xlabel("Increase in MSE Loss (Permutation Importance)", color="#aaaacc", fontsize=11)
    ax.set_title("Feature Importance (Test Set)", color="white", fontsize=14)
    ax.grid(axis='x', alpha=0.15, color="#ffffff")
    
    out_path = out_dir / "feature_importance.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved Feature Importance Histogram: {out_path}")


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


def plot_residuals_kde(pred: np.ndarray, true: np.ndarray, out_dir: Path) -> None:
    """Save a continuous KDE plot of the error residuals."""
    from scipy.stats import gaussian_kde
    
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    residuals = pred - true
    
    # Calculate KDE
    kde = gaussian_kde(residuals)
    x_range = np.linspace(min(residuals), max(residuals), 500)
    y_kde = kde(x_range)
    
    ax.plot(x_range, y_kde, color='#7ec8e3', lw=2)
    ax.fill_between(x_range, y_kde, color='#7ec8e3', alpha=0.3)
    
    ax.axvline(0, color='#ff6b6b', linestyle='--', linewidth=2, label="Zero Error")
    
    mae = float(np.abs(residuals).mean())
    std = float(np.std(residuals))
    
    ax.set_title(f"Test Set: Error Residuals Distribution\nMean: {np.mean(residuals):.5f} | Std: {std:.5f} | MAE: {mae:.5f}",
                 color="white", fontsize=12)
    ax.set_xlabel("Error (Predicted - Ideal)", color="#aaaacc", fontsize=11)
    ax.set_ylabel("Density", color="#aaaacc", fontsize=11)
    ax.tick_params(colors="#aaaacc")
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="white")
    ax.grid(True, alpha=0.12, color="#ffffff")

    path = out_dir / "predictions_residuals_kde.png"
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
    p.add_argument("--train_batch",   type=int,   default=DEFAULT_TRAIN_BATCH)
    p.add_argument("--eval_batch",    type=int,   default=DEFAULT_EVAL_BATCH)
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
    print(f"  Train Batch  : {args.train_batch}")
    print(f"  Eval Batch   : {args.eval_batch}")
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
    train_ds, val_ds, test_ds, scaler, feature_names, pruned_noisy_indices = load_and_split(
        npz_path     = Path(args.dataset),
        val_split    = args.val_split,
        test_split   = args.test_split,
        seed         = args.seed,
        min_survival = args.min_survival,
        device       = device,
    )

    # Diagnostic Histogram Plot
    all_y_train = train_ds.y.cpu().numpy()
    all_y_val   = val_ds.y.cpu().numpy()
    all_y_test  = test_ds.y.cpu().numpy()
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

    # Data is already on the GPU — use the zero-copy GPU-native iterator.
    train_loader = GPUDataLoader(train_ds, batch_size=args.train_batch, shuffle=True)
    val_loader   = GPUDataLoader(val_ds,   batch_size=args.eval_batch, shuffle=False)
    test_loader  = GPUDataLoader(test_ds,  batch_size=args.eval_batch, shuffle=False)

    # ------------------------------------------------------------------
    # Model — input_dim computed dynamically from pruned feature count
    # ------------------------------------------------------------------
    print("\n[2/4] Building model ...")
    # Conv1D processes 5 raw ZNE exp-value columns (ordered decay curve).
    # Conv1D output: 4 channels × 5 width = 20.
    # Static cols = total cols - 5 ZNE cols.
    n_total   = train_ds.X.shape[1]   # actual column count after pruning
    n_zne_raw = 5                      # ZNE cols fed to Conv1D
    n_static  = n_total - n_zne_raw
    input_dim = n_static + 4 * n_zne_raw   # static + conv output
    print(f"  Feature matrix: {n_total} cols "
          f"({n_zne_raw} ZNE -> Conv1D(4ch,k=3), {n_static} static) "
          f"=> input_dim={input_dim}")

    model = QuantumMitigationMLP(
        input_dim = input_dim,
        hidden1   = 2048,
        hidden2   = 2048,
        hidden3   = 1024,
        hidden4   = 1024,
        hidden5   = 512,
        hidden6   = 256,
        dropout   = args.dropout,
    ).to(device)

    # Sanity-check: catch shape mismatch before the first forward pass
    expected = model.shared_net[0].fc.in_features
    assert expected == input_dim, (
        f"[FATAL] Model expects {expected} inputs but dataset provides {input_dim}. "
        f"Check column pruning logic."
    )

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
        lambda_mono   = args.lambda_mono,
        lambda_bndry  = args.lambda_bndry,
        noisy_indices = pruned_noisy_indices,  # post-pruning ZNE col positions
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
        steps_per_epoch=len(train_loader)
    )

    use_amp = (device.type == "cuda")
    grad_scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("  [Optimization] Automatic Mixed Precision (AMP) enabled.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n[3/4] Training for {args.epochs} epochs ...")
    print("-" * 109)
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | "
          f"{'Train MAE':>10} | {'Val MAE':>9} | {'Train R²':>8} | {'Val R²':>8} | {'Test R²':>8} | "
          f"{'Tr Acc%':>7} | {'Val Acc%':>8} | LR")
    print("-" * 109)

    history = {k: [] for k in
               ["train_loss","val_loss","train_mse","val_mse",
                "train_mae","val_mae","train_r2","val_r2",
                "train_acc","val_acc","test_r2"]}

    best_val_loss    = float("inf")
    patience_counter = 0
    best_ckpt_path   = OUTPUT_DIR / "best_mlp_model.pth"
    log_path       = OUTPUT_DIR / "training_log.csv"

    # CSV logger
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","val_loss","train_mae","val_mae",
                         "train_r2","val_r2","test_r2","lr"])

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=grad_scaler, scheduler=scheduler)
        vl = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
        ts = evaluate(model, test_loader, criterion, device, use_amp=use_amp)

        current_lr = scheduler.get_last_lr()[0]

        for k in ["loss","mse","mae","r2","acc"]:
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(vl[k])
        history["test_r2"].append(ts["r2"])

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

        if epoch % 1 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"{epoch:>6} | {tr['loss']:>11.6f} | {vl['loss']:>10.6f} | "
                  f"{tr['mae']:>10.6f} | {vl['mae']:>9.6f} | {tr['r2']:>8.4f} | {vl['r2']:>8.4f} | {ts['r2']:>8.4f} | "
                  f"{tr['acc']:>6.1f}% | {vl['acc']:>7.1f}% | "
                  f"{current_lr:.2e}{star}")

        # Append to CSV log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr["loss"], vl["loss"],
                             tr["mae"], vl["mae"], tr["r2"], vl["r2"], ts["r2"],
                             current_lr])

        if patience_counter >= 15:
            print(f"\n  [Early Stopping] Validation loss hasn't improved for 15 epochs. Breaking at epoch {epoch}.")
            break

    total_time = time.time() - t_start
    print("-" * 109)
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
    # Test set evaluation (using best checkpoint)
    # ------------------------------------------------------------------
    print(f"\n[4/4] Evaluating on held-out test set ...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)

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
    plot_residuals_kde(test_metrics["pred"], test_metrics["true"], OUTPUT_DIR)
    compute_permutation_importance(model, test_loader, criterion, device, feature_names, OUTPUT_DIR)
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

if __name__ == "__main__":
    main()
