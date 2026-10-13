"""
finetune.py
===========
Finetune the best Physics-Informed MLP checkpoint on the real Qiskit Aer dataset.

Strategy
--------
- Loads  : outputs/mlp_mitiq/best_mlp_model.pth  (pretrained on synthetic data)
- Data   : ../dataset/qiskit_vector_dataset.npz   (~948 real Qiskit Aer samples)
- Saves  : outputs/mlp_mitiq/best_finetuned_model.pth

Finetuning details
------------------
* The scaler (mean/scale) from the pretrained checkpoint is re-used so features
  are normalised identically to pretraining.
* The shared trunk (shared_net + zne_conv) is initially FROZEN for the first
  FREEZE_EPOCHS epochs so only the output branches adapt.
* After FREEZE_EPOCHS, the entire network is unfrozen and trained with a much
  smaller learning rate (FULL_LR) to gently shift the shared representations.
* Early stopping on validation loss (patience = PATIENCE epochs).
* Checkpoint format is identical to mlp_mitiq.py so mitiq_testing.py works
  without any changes.

Usage
-----
    cd zne/mitigation_ai
    python finetune.py [--epochs 60] [--freeze_epochs 15] [--batch 32]
                       [--lr 5e-5] [--full_lr 1e-5] [--patience 20]
"""

import os
import sys
import argparse
import time
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
ROOT_DIR    = BASE_DIR.parent
QISKIT_NPZ  = ROOT_DIR / "dataset" / "test_data.npz"
OUTPUT_DIR  = BASE_DIR / "outputs" / "mlp_mitiq"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRETRAIN_CKPT   = OUTPUT_DIR / "best_mlp_model.pth"
FINETUNE_CKPT   = OUTPUT_DIR / "best_finetuned_model.pth"

# ---------------------------------------------------------------------------
# Import model & loss from main training script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(BASE_DIR))
from mlp_mitiq import (
    QuantumMitigationMLP,
    PhysicsInformedLoss,
    r2_score,
    train_one_epoch,
    evaluate,
    compute_permutation_importance,
    # Column index constants
    IDX_NQUBITS, IDX_DEPTH, IDX_ERROR_1Q, IDX_ERROR_2Q,
    IDX_T1, IDX_T2, IDX_TIME_1Q, IDX_TIME_2Q,
    IDX_P_LEAK, IDX_P_01, IDX_P_10, IDX_P_XTALK,
    IDX_BETA_NM, IDX_THETA_ERROR, IDX_OBS_ALIGN,
    NOISY_EXP_INDICES,
    MIN_SURVIVAL,
)

# ---------------------------------------------------------------------------
# Hyper-parameters (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS        = 150      # increased for large datasets
DEFAULT_FREEZE_EPOCHS = 20       # epochs to keep shared trunk frozen
DEFAULT_LR            = 2e-5    # LR during frozen-trunk phase (lowered)
DEFAULT_FULL_LR       = 5e-6    # LR after full-network unfreeze (lowered)
DEFAULT_BATCH         = 16      # small dataset — tiny batches
DEFAULT_PATIENCE      = 30      # increased patience for large datasets
DEFAULT_MIN_SURVIVAL  = 0.1     # Qiskit Aer data is cleaner; use stricter filter
DEFAULT_VAL_FRAC      = 0.15
DEFAULT_SEED          = 42


# ---------------------------------------------------------------------------
# Data preprocessing  (mirrors mitiq_testing.py + load_and_split)
# ---------------------------------------------------------------------------

def build_qiskit_dataset(
    npz_path: Path,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    expected_feature_names: list,
    min_survival: float = DEFAULT_MIN_SURVIVAL,
    val_frac: float     = DEFAULT_VAL_FRAC,
    seed: int           = DEFAULT_SEED,
    device: torch.device = None,
):
    """
    Load qiskit_vector_dataset.npz (25 cols), strip the is_biased flag (col 19),
    engineer the same 7 derived physics features as load_and_split(), apply the
    *pretrained* scaler, filter dead circuits, then split train/val.

    Returns  train_ds, val_ds, is_biased_full (bool array over all kept samples)
    """
    data = np.load(npz_path, allow_pickle=True)
    X_raw_25 = data["X"].astype(np.float32)   # (N, 25)
    y_raw    = data["y"].astype(np.float32)

    # Strip is_biased column (index 19 in the 25-col layout) and record it
    is_biased_flag = X_raw_25[:, 19].astype(bool)

    # Reconstruct 24-col X identical to the synthetic dataset layout
    X_raw = np.concatenate([X_raw_25[:, :19], X_raw_25[:, 20:25]], axis=1)  # (N, 24)

    # ------------------------------------------------------------------
    # Derived physics features  (identical to mlp_mitiq.load_and_split)
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

    t1q_us    = t1q_ns * 1e-3
    t2q_us    = t2q_ns * 1e-3
    N_total   = nq * d
    t_eff_us  = FRAC_1Q * N_total * t1q_us + FRAC_2Q * N_total * t2q_us

    eps_eff   = FRAC_1Q * e1q + FRAC_2Q * e2q
    eps_N     = eps_eff * N_total

    gamma_dep   = np.exp(-(np.where(eps_N > 0, eps_N, 0.0) ** beta_nm))
    gamma_T1    = np.exp(-t_eff_us / np.maximum(T1, 1e-9))
    p_decay_T1  = (1.0 - gamma_T1).astype(np.float32)
    gamma_T2    = np.exp(-t_eff_us / np.maximum(T2, 1e-9))
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

    e_noisy = X_raw[:, 19:24]          # ZNE exp vals at scales 1,3,5,7,9
    e1, e3, e5 = e_noisy[:, 0], e_noisy[:, 1], e_noisy[:, 2]
    zne_extrap  = ((3.0 * e1 - e3) / 2.0).astype(np.float32)
    lam_dev     = np.array([-4., -2., 0., 2., 4.], dtype=np.float32)
    e_mean      = e_noisy.mean(axis=1)
    b_coef      = (e_noisy * lam_dev).sum(axis=1) / 40.0
    zne_linear  = (e_mean - b_coef * 5.0).astype(np.float32)
    zne_quad    = ((15.0 * e1 - 10.0 * e3 + 3.0 * e5) / 8.0).astype(np.float32)
    shot_noise_variance = ((1.0 - zne_extrap ** 2) / 8192.0).astype(np.float32)

    X_aug = np.concatenate(
        [X_raw,
         survival_combined[:, None], log_surv[:, None], zne_extrap[:, None],
         zne_linear[:, None], zne_quad[:, None], p_decay_T1[:, None],
         shot_noise_variance[:, None]],
        axis=1,
    )   # (N, 31)

    # ------------------------------------------------------------------
    # Column pruning
    # ------------------------------------------------------------------
    base_feature_names = [
        "num_qubits", "depth", "error_1q", "error_2q",
        "T1_us", "T2_us", "time_1q_ns", "time_2q_ns",
        "p_damp_1q", "p_deph_1q", "p_damp_2q", "p_deph_2q",
        "p_leak", "p_01", "p_10", "p_xtalk", "beta_nm", "theta_error",
        "observable_alignment",
        "exp_val_scale_1", "exp_val_scale_3", "exp_val_scale_5",
        "exp_val_scale_7", "exp_val_scale_9"
    ]
    
    aug_names = base_feature_names + [
        "survival_combined", "log_survival", "zne_extrap",
        "zne_linear", "zne_quad", "p_decay_T1", "shot_noise_variance"
    ]
    
    PRUNE_COLS = {"time_1q_ns", "time_2q_ns", "T1_us", "T2_us"}
    prune_mask = [i for i, n in enumerate(aug_names) if n in PRUNE_COLS]
    if prune_mask:
        keep = [i for i in range(len(aug_names)) if i not in prune_mask]
        X_aug = X_aug[:, keep]
        aug_names = [aug_names[i] for i in keep]
        
    assert aug_names == expected_feature_names, "Feature names after pruning do not match the checkpoint's expected features!"

    # ------------------------------------------------------------------
    # Survival filter
    # ------------------------------------------------------------------
    mask          = np.abs(survival_combined) >= min_survival
    X_aug         = X_aug[mask]
    y_raw         = y_raw[mask]
    is_biased_flag = is_biased_flag[mask]
    print(f"  Survival filter (>= {min_survival}): kept {mask.sum():,} / {len(mask):,} samples")

    # ------------------------------------------------------------------
    # Apply PRETRAINED scaler (do not refit -- that would break alignment)
    # ------------------------------------------------------------------
    no_scale_names = {
        "exp_val_scale_1", "exp_val_scale_3", "exp_val_scale_5",
        "exp_val_scale_7", "exp_val_scale_9",
        "zne_extrap", "zne_linear", "zne_quad", "shot_noise_variance"
    }
    zne_idx    = [i for i, n in enumerate(aug_names) if n in no_scale_names]
    static_idx = [i for i in range(X_aug.shape[1]) if i not in zne_idx]

    X_scaled = X_aug.copy()
    X_scaled[:, static_idx] = (X_aug[:, static_idx] - scaler_mean) / scaler_scale

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    rng    = np.random.default_rng(seed)
    perm   = rng.permutation(len(y_raw))
    n_val  = max(1, int(len(y_raw) * val_frac))
    n_train = len(y_raw) - n_val

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    X_t = torch.tensor(X_scaled[train_idx], dtype=torch.float32, device=device)
    y_t = torch.tensor(y_raw[train_idx],    dtype=torch.float32, device=device)
    X_v = torch.tensor(X_scaled[val_idx],   dtype=torch.float32, device=device)
    y_v = torch.tensor(y_raw[val_idx],      dtype=torch.float32, device=device)

    train_ds = TensorDataset(X_t, y_t)
    val_ds   = TensorDataset(X_v, y_v)

    print(f"  Train : {len(train_ds)} samples")
    print(f"  Val   : {len(val_ds)} samples")

    return train_ds, val_ds, is_biased_flag


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _dark_ax(fig, ax):
    """Apply the standard dark-mode styling."""
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.tick_params(colors="#aaaacc")
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.title.set_color("white")
    ax.grid(True, alpha=0.12, color="#ffffff")


def plot_finetune_curves(history: dict, out_dir: Path):
    metrics = [
        ("loss",  "PI-Loss",      False),
        ("mse",   "MSE",          False),
        ("mae",   "MAE",          False),
        ("r2",    "R2 Score",     False),
        ("acc",   "Acc (+-0.05) %", False),
    ]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.patch.set_facecolor("#0f0f1a")

    for ax, (key, title, logy) in zip(axes, metrics):
        _dark_ax(fig, ax)
        ax.plot(history[f"train_{key}"], label="Train", color="#7ec8e3")
        ax.plot(history[f"val_{key}"],   label="Val",   color="#ff6b6b")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(framealpha=0.2, labelcolor="white")
        if logy:
            ax.set_yscale("log")

    fig.suptitle("Finetuning Curves - Qiskit Aer Dataset",
                 color="white", fontsize=15, y=1.01, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "finetune_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_scatter(pred: np.ndarray, true: np.ndarray, r2: float, mae: float, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    _dark_ax(fig, ax)

    sc = ax.scatter(true, pred, c=np.abs(pred - true), cmap="plasma",
                    alpha=0.55, s=18, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("|Error|", color="#aaaacc")
    cbar.ax.yaxis.set_tick_params(color="#aaaacc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaacc")

    lims = [-1.05, 1.05]
    ax.plot(lims, lims, color="#7ec8e3", lw=1.5, linestyle="--", label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(f"Finetuned - Test Set: Predicted vs Ideal\nR2={r2:.4f}   MAE={mae:.5f}",
                 color="white", fontsize=12)
    ax.set_xlabel("Ideal Expectation Value",  color="#aaaacc", fontsize=11)
    ax.set_ylabel("Mitigated (Predicted)",    color="#aaaacc", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="white")

    path = out_dir / "finetune_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_residuals(pred: np.ndarray, true: np.ndarray, out_dir: Path):
    residuals = pred - true
    fig, ax = plt.subplots(figsize=(7, 6))
    _dark_ax(fig, ax)

    kde    = gaussian_kde(residuals)
    xr     = np.linspace(residuals.min(), residuals.max(), 500)
    ykde   = kde(xr)
    ax.plot(xr, ykde, color="#7ec8e3", lw=2)
    ax.fill_between(xr, ykde, color="#7ec8e3", alpha=0.3)
    ax.axvline(0, color="#ff6b6b", linestyle="--", lw=2, label="Zero Error")

    mae = float(np.abs(residuals).mean())
    std = float(np.std(residuals))
    ax.set_title(f"Finetuned - Error Residuals Distribution\n"
                 f"Mean: {residuals.mean():.5f} | Std: {std:.5f} | MAE: {mae:.5f}",
                 color="white", fontsize=12)
    ax.set_xlabel("Error (Predicted - Ideal)", color="#aaaacc", fontsize=11)
    ax.set_ylabel("Density",                   color="#aaaacc", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="white")

    path = out_dir / "finetune_residuals_kde.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Finetune PI-MLP on Qiskit Aer dataset")
    p.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--freeze_epochs", type=int,   default=DEFAULT_FREEZE_EPOCHS,
                   help="Epochs to keep shared trunk frozen (head-only warmup)")
    p.add_argument("--lr",            type=float, default=DEFAULT_LR,
                   help="LR during frozen-trunk phase")
    p.add_argument("--full_lr",       type=float, default=DEFAULT_FULL_LR,
                   help="LR after full-network unfreeze")
    p.add_argument("--batch",         type=int,   default=DEFAULT_BATCH)
    p.add_argument("--patience",      type=int,   default=DEFAULT_PATIENCE)
    p.add_argument("--min_survival",  type=float, default=DEFAULT_MIN_SURVIVAL)
    p.add_argument("--val_frac",      type=float, default=DEFAULT_VAL_FRAC)
    p.add_argument("--seed",          type=int,   default=DEFAULT_SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device : CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Device : Apple MPS")
    else:
        device = torch.device("cpu")
        print("  Device : CPU")

    # Load pretrained checkpoint
    print(f"\n[1/4] Loading pretrained checkpoint: {PRETRAIN_CKPT}")
    if not PRETRAIN_CKPT.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found at {PRETRAIN_CKPT}")

    ckpt         = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
    scaler_mean  = np.array(ckpt["scaler_mean"],  dtype=np.float32)
    scaler_scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
    feature_names = ckpt.get("feature_names", [])

    input_dim = ckpt["model_state"]["shared_net.0.fc.weight"].shape[1]
    model = QuantumMitigationMLP(input_dim=input_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} | val_loss={ckpt.get('val_loss', float('nan')):.6f}")
    print(f"  Parameters: {model.count_parameters():,}")

    # Load & preprocess Qiskit dataset
    print(f"\n[2/4] Loading Qiskit Aer dataset: {QISKIT_NPZ}")
    if not QISKIT_NPZ.exists():
        raise FileNotFoundError(f"Qiskit dataset not found at {QISKIT_NPZ}")

    train_ds, val_ds, is_biased_all = build_qiskit_dataset(
        QISKIT_NPZ, scaler_mean, scaler_scale, feature_names,
        min_survival=args.min_survival,
        val_frac=args.val_frac,
        seed=args.seed,
        device=device,
    )

    # ------------------------------------------------------------------
    # Auto-Scaling Logic for 50k Dataset
    # ------------------------------------------------------------------
    num_train = len(train_ds)
    if num_train > 10000:
        # For large 50k datasets, batch=16 is too slow and gradients are too noisy.
        # We scale up batch size to 256 and use a proportionately higher LR (Linear Scaling Rule)
        # to ensure it can reach R2 > 0.98 quickly without getting stuck.
        actual_batch = 256
        actual_lr = 1e-4
        actual_full_lr = 2e-5
        print(f"\n  [Auto-Scale] Large dataset detected ({num_train:,} samples).")
        print(f"  [Auto-Scale] Scaling Batch Size -> {actual_batch}")
        print(f"  [Auto-Scale] Scaling LRs -> {actual_lr} (phase 1) / {actual_full_lr} (phase 2)")
    elif num_train > 2000:
        actual_batch = 64
        actual_lr = 5e-5
        actual_full_lr = 1e-5
        print(f"\n  [Auto-Scale] Medium dataset detected ({num_train:,} samples).")
        print(f"  [Auto-Scale] Scaling Batch Size -> {actual_batch}, LRs -> {actual_lr}, {actual_full_lr}")
    else:
        actual_batch = args.batch
        actual_lr = args.lr
        actual_full_lr = args.full_lr

    train_loader = DataLoader(train_ds, batch_size=actual_batch, shuffle=True,  pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=actual_batch, shuffle=False, pin_memory=False)

    # Criterion
    criterion = PhysicsInformedLoss()

    # Phase 1: Freeze shared trunk, train output branches only
    print(f"\n[3/4] Finetuning ({args.freeze_epochs} frozen + "
          f"{max(0, args.epochs - args.freeze_epochs)} full epochs) ...")

    def set_trunk_frozen(frozen: bool):
        for name, param in model.named_parameters():
            if name.startswith("shared_net") or name.startswith("zne_conv"):
                param.requires_grad = not frozen

    set_trunk_frozen(True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=actual_lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.freeze_epochs), eta_min=actual_lr * 0.1,
    )

    history = {k: [] for k in
               ["train_loss", "train_mse", "train_mae", "train_r2", "train_acc",
                "val_loss",   "val_mse",   "val_mae",   "val_r2",   "val_acc"]}

    best_val_loss    = float("inf")
    patience_counter = 0
    t_start          = time.time()

    for epoch in range(1, args.epochs + 1):

        # Phase transition: unfreeze trunk
        if epoch == args.freeze_epochs + 1:
            print(f"\n  Epoch {epoch}: Unfreezing full network (LR -> {actual_full_lr})")
            set_trunk_frozen(False)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=actual_full_lr, weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - args.freeze_epochs),
                eta_min=actual_full_lr * 0.1,
            )

        tr = train_one_epoch(model, train_loader, optimizer, criterion, device,
                             scaler=None, scheduler=scheduler)
        vl = evaluate(model, val_loader, criterion, device)

        for k in ["loss", "mse", "mae", "r2", "acc"]:
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(vl[k])

        phase = "frozen" if epoch <= args.freeze_epochs else "full"
        print(f"  Ep {epoch:>4d}/{args.epochs} [{phase:>6s}] "
              f"| train loss={tr['loss']:.5f}  mae={tr['mae']:.5f}  r2={tr['r2']:.4f}"
              f"  | val loss={vl['loss']:.5f}  mae={vl['mae']:.5f}  r2={vl['r2']:.4f}", end="")

        # Save best checkpoint (same format as mlp_mitiq.py)
        if vl["loss"] < best_val_loss:
            best_val_loss = vl["loss"]
            torch.save({
                "epoch"           : epoch,
                "model_state"     : model.state_dict(),
                "optimizer_state" : optimizer.state_dict(),
                "val_loss"        : best_val_loss,
                "args"            : vars(args),
                "feature_names"   : feature_names,
                "scaler_mean"     : scaler_mean.tolist(),
                "scaler_scale"    : scaler_scale.tolist(),
            }, FINETUNE_CKPT)
            print(" *", end="")
            patience_counter = 0
        else:
            patience_counter += 1

        print()

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    elapsed = time.time() - t_start
    print(f"\n  Finetuning complete in {elapsed:.1f}s")
    print(f"  Best val loss : {best_val_loss:.6f}")
    print(f"  Checkpoint    : {FINETUNE_CKPT}")

    # [4/4] Evaluate finetuned model on val set
    print(f"\n[4/4] Final evaluation on validation set ...")
    best_ckpt = torch.load(FINETUNE_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    final_metrics = evaluate(model, val_loader, criterion, device)

    print("\n" + "=" * 65)
    print("  FINETUNED MODEL - VALIDATION SET RESULTS")
    print("=" * 65)
    print(f"  PI-Loss  : {final_metrics['loss']:.6f}")
    print(f"  MSE      : {final_metrics['mse']:.6f}")
    print(f"  MAE      : {final_metrics['mae']:.6f}")
    print(f"  R2       : {final_metrics['r2']:.6f}")
    print(f"  Acc(+-0.05): {final_metrics['acc']:.2f}%")
    print("=" * 65)

    # Plots
    print("\n  Generating plots ...")
    plot_finetune_curves(history, OUTPUT_DIR)

    pred = final_metrics["pred"]
    true = final_metrics["true"]
    plot_scatter(pred, true, final_metrics["r2"], final_metrics["mae"], OUTPUT_DIR)
    plot_residuals(pred, true, OUTPUT_DIR)
    
    compute_permutation_importance(model, val_loader, criterion, device, feature_names, OUTPUT_DIR)
    old_feat_path = OUTPUT_DIR / "feature_importance.png"
    new_feat_path = OUTPUT_DIR / "finetuned_feature_importance.png"
    if old_feat_path.exists():
        old_feat_path.replace(new_feat_path)
        print(f"  Saved Feature Importance to: {new_feat_path}")

    # CSV log
    log_path = OUTPUT_DIR / "finetune_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_mae",
                                               "train_r2", "val_loss", "val_mae", "val_r2"])
        writer.writeheader()
        for i in range(len(history["train_loss"])):
            writer.writerow({
                "epoch"      : i + 1,
                "train_loss" : history["train_loss"][i],
                "train_mae"  : history["train_mae"][i],
                "train_r2"   : history["train_r2"][i],
                "val_loss"   : history["val_loss"][i],
                "val_mae"    : history["val_mae"][i],
                "val_r2"     : history["val_r2"][i],
            })
    print(f"  Training log : {log_path}")
    print(f"  All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
