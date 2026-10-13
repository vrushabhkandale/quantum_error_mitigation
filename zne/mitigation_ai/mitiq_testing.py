import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Import model architecture
from mlp_mitiq import QuantumMitigationMLP, r2_score

def preprocess_dataset(npz_path, mean, scale):
    data = np.load(npz_path, allow_pickle=True)
    X_raw = data["X"].astype(np.float32)
    
    if X_raw.shape[1] == 25:
        is_biased = X_raw[:, 19].astype(bool)
        # Reconstruct X_raw to match 24-col model training format
        X_raw = np.concatenate([X_raw[:, :19], X_raw[:, 20:25]], axis=1)
    else:
        is_biased = np.zeros(X_raw.shape[0], dtype=bool)
    y_raw = data["y"].astype(np.float32)         # (N,)

    # Extract columns
    nq        = X_raw[:, 0].astype(np.float64)
    d         = X_raw[:, 1].astype(np.float64)
    e1q       = X_raw[:, 2].astype(np.float64)
    e2q       = X_raw[:, 3].astype(np.float64)
    T1        = X_raw[:, 4].astype(np.float64)
    T2        = X_raw[:, 5].astype(np.float64)
    t1q_ns    = X_raw[:, 6].astype(np.float64)
    t2q_ns    = X_raw[:, 7].astype(np.float64)
    p_leak    = X_raw[:, 12].astype(np.float64)
    p_01      = X_raw[:, 13].astype(np.float64)
    p_10      = X_raw[:, 14].astype(np.float64)
    p_xt      = X_raw[:, 15].astype(np.float64)
    beta_nm   = X_raw[:, 16].astype(np.float64)
    theta_err = X_raw[:, 17].astype(np.float64)
    obs_align = X_raw[:, 18].astype(np.float64)

    # Gate-set fractions (must match vector_data.py)
    FRAC_1Q = 2.0 / 3.0
    FRAC_2Q = 1.0 / 3.0

    t1q_us = t1q_ns * 1e-3
    t2q_us = t2q_ns * 1e-3

    N_total  = nq * d   # total qubit-gate operations at lambda=1
    t_eff_us = FRAC_1Q * N_total * t1q_us + FRAC_2Q * N_total * t2q_us

    eps_eff = FRAC_1Q * e1q + FRAC_2Q * e2q
    eps_N      = eps_eff * N_total
    gamma_dep  = np.exp(-(np.where(eps_N > 0, eps_N, 0.0) ** beta_nm))
    gamma_T1   = np.exp(-t_eff_us / np.maximum(T1, 1e-9))
    p_decay_T1 = (1.0 - gamma_T1).astype(np.float32)
    gamma_T2   = np.exp(-t_eff_us / np.maximum(T2, 1e-9))
    gamma_leak = (1.0 - p_leak) ** N_total
    n_pairs     = nq * (nq - 1) / 2.0
    gamma_xtalk = np.exp(-p_xt * n_pairs * d)
    readout_fac = 1.0 - p_01 - p_10
    gamma_coh = np.cos(theta_err * N_total)

    survival_combined = (
        gamma_dep * gamma_T1 * gamma_T2 * gamma_leak
        * gamma_xtalk * gamma_coh * readout_fac
    ).astype(np.float32)
    log_surv = np.log(np.clip(np.abs(survival_combined), 1e-10, 1.0)).astype(np.float32)

    e_noisy = X_raw[:, 19:24]
    e1 = e_noisy[:, 0]

    # X_raw is 25 cols. Add survival_combined, log_surv, p_decay_T1.
    X_aug = np.concatenate(
        [X_raw,
         survival_combined[:, None], log_surv[:, None], p_decay_T1[:, None]],
        axis=1,
    )

    # Mask low survival
    mask   = np.abs(survival_combined) >= 0.3
    X_aug  = X_aug[mask]
    y_raw  = y_raw[mask]
    is_biased = is_biased[mask]

    feature_names = [
        "num_qubits", "depth", "error_1q", "error_2q",
        "T1_us", "T2_us", "time_1q_ns", "time_2q_ns",
        "p_damp_1q", "p_damp_2q", "p_deph_1q", "p_deph_2q",
        "p_leak", "p_01", "p_10", "p_xtalk", "beta_nm", "theta_error",
        "observable_alignment",
        "exp_val_scale_1", "exp_val_scale_3", "exp_val_scale_5",
        "exp_val_scale_7", "exp_val_scale_9"
    ]
    aug_names = feature_names + [
        "survival_combined", "log_survival", "p_decay_T1"
    ]

    PRUNE_COLS = {"time_1q_ns", "time_2q_ns", "T1_us", "T2_us", "survival_factor"}
    prune_mask = [i for i, n in enumerate(aug_names) if n in PRUNE_COLS]
    if prune_mask:
        keep = [i for i in range(len(aug_names)) if i not in prune_mask]
        X_aug = X_aug[:, keep]
        aug_names = [aug_names[i] for i in keep]

    no_scale_names = {
        "exp_val_scale_1", "exp_val_scale_3", "exp_val_scale_5",
        "exp_val_scale_7", "exp_val_scale_9",
    }
    zne_idx    = [i for i, n in enumerate(aug_names) if n in no_scale_names]
    static_idx = [i for i in range(X_aug.shape[1]) if i not in zne_idx]

    X_scaled = X_aug.copy()
    X_scaled[:, static_idx] = (X_aug[:, static_idx] - mean) / scale

    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_raw, dtype=torch.float32), is_biased, aug_names


def evaluate_mitiq_model():
    print("=" * 80)
    print("  Testing best_mlp_model.pth on Vector Dataset")
    print("=" * 80)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(BASE_DIR, "outputs", "mlp_mitiq", "best_mlp_model.pth")
    dataset_path = os.path.join(BASE_DIR, "..", "dataset", "gaussian_dataset_small.npz")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    scaler_mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
    
    print(f"Preprocessing {dataset_path} ...")
    X_test, y_test_abs, is_biased, aug_names = preprocess_dataset(dataset_path, scaler_mean, scaler_scale)
    print(f"Test samples after filtering: {len(X_test)}")
    
    if len(X_test) == 0:
        print("No samples survived the survival_combined filter (>=0.2). Exiting.")
        return
        
    # Model
    n_total = X_test.shape[1]
    input_dim = (n_total - 5) + 20
    model = QuantumMitigationMLP(input_dim=input_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    with torch.no_grad():
        preds_res = model(X_test)[:, 0]
        exp1_idx = aug_names.index("exp_val_scale_1")
        yb = X_test[:, exp1_idx]
        preds = (preds_res + yb).clamp(-1.0, 1.0)
        
    mae = torch.abs(preds - y_test_abs).mean().item()
    r2 = r2_score(preds, y_test_abs)
    
    acc_threshold = 0.05
    acc = (torch.abs(preds - y_test_abs) <= acc_threshold).float().mean().item() * 100.0
    
    print("-" * 80)
    print(f"Results on Qiskit Aer Dataset (N={len(X_test)}):")
    print(f"  MAE       : {mae:.6f}")
    print(f"  R²        : {r2:.4f}")
    print(f"  Acc (5%)  : {acc:.2f}%")
    
    if is_biased.any():
        print("-" * 80)
        mask_b = torch.tensor(is_biased, dtype=torch.bool)
        mask_u = ~mask_b
        
        preds_b, y_b = preds[mask_b], y_test_abs[mask_b]
        mae_b = torch.abs(preds_b - y_b).mean().item()
        acc_b = (torch.abs(preds_b - y_b) <= acc_threshold).float().mean().item() * 100.0
        r2_b = r2_score(preds_b, y_b) if len(y_b) > 1 else float('nan')
        
        preds_u, y_u = preds[mask_u], y_test_abs[mask_u]
        mae_u = torch.abs(preds_u - y_u).mean().item()
        acc_u = (torch.abs(preds_u - y_u) <= acc_threshold).float().mean().item() * 100.0
        r2_u = r2_score(preds_u, y_u) if len(y_u) > 1 else float('nan')

        print(f"  Biased Init (N={mask_b.sum()}):")
        print(f"    MAE       : {mae_b:.6f}")
        print(f"    R²        : {r2_b:.4f}")
        print(f"    Acc (5%)  : {acc_b:.2f}%")
        print(f"  Uniform Init (N={mask_u.sum()}):")
        print(f"    MAE       : {mae_u:.6f}")
        print(f"    R²        : {r2_u:.4f}")
        print(f"    Acc (5%)  : {acc_u:.2f}%")
    print("-" * 80)
    
    # Plot Scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_abs.numpy(), preds.numpy(), alpha=0.5, s=10)
    
    min_val = min(y_test_abs.min().item(), preds.min().item())
    max_val = max(y_test_abs.max().item(), preds.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel("True Expectation Value (Ideal)")
    plt.ylabel("Predicted Mitigated Value")
    plt.title("Mitigated vs Ideal (Vector Dataset)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    out_plot = os.path.join(BASE_DIR, "outputs", "mitiq_testing_qiskit_scatter.png")
    plt.savefig(out_plot, dpi=150)
    print(f"Saved scatter plot to {out_plot}")
    
if __name__ == "__main__":
    evaluate_mitiq_model()
