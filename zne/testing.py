from asyncio import base_futures
from asyncio import base_futures
import os
import sys
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from decision_trees.lightgbm_ml import engineer_lgbm_features
from mitigation_ai.mlp_mitiq import QuantumMitigationMLP

def main():
    dataset_path = os.path.join(BASE_DIR, 'dataset', 'test_data.npz')
    print(f"Loading dataset from {dataset_path}")
    data = np.load(dataset_path)
    feature_names = list(data['feature_names'])
    X_raw = data['X']
    y_raw = data['y']
    
    # --- LightGBM Evaluation ---
    print("\n========================================")
    print(" Evaluating LightGBM Model ")
    print("========================================")
    
    lgbm_model_path = os.path.join(BASE_DIR, 'decision_trees', 'lgbm_model.txt')
    lgbm_model = lgb.Booster(model_file=lgbm_model_path)
    
    # Prepare LightGBM features (drop is_biased if exists)
    X_lgbm_raw = X_raw.copy()
    fn_lgbm = feature_names.copy()
    if 'is_biased' in fn_lgbm:
        idx = fn_lgbm.index('is_biased')
        fn_lgbm.pop(idx)
        X_lgbm_raw = np.delete(X_lgbm_raw, idx, axis=1)
        
    df = pd.DataFrame(X_lgbm_raw, columns=fn_lgbm)
    df['ideal_exp_val'] = y_raw
    df = engineer_lgbm_features(df)
    
    SURVIAL_FIL = 0.05
    sur_filter = SURVIAL_FIL
    # Apply survival mask >= 0.3 to match mitiq_testing.py exactly
    mask = np.abs(df['survival_combined']) >= sur_filter
    df_filtered = df[mask]
    
    target_col = 'ideal_exp_val'
    feature_cols = [c for c in df_filtered.columns if c != target_col]
    X_lgbm = df_filtered[feature_cols].values
    y_lgbm = df_filtered[target_col].values
    
    y_pred_lgbm = lgbm_model.predict(X_lgbm)
    
    mae_lgbm = mean_absolute_error(y_lgbm, y_pred_lgbm)
    r2_lgbm = r2_score(y_lgbm, y_pred_lgbm)
    print(f"Tested on {len(X_lgbm)} samples (filtered >= {sur_filter}).")
    print(f"LightGBM Test MAE: {mae_lgbm:.6f}")
    print(f"LightGBM Test R²:  {r2_lgbm:.6f}")
    
    # --- PI-MLP Evaluation ---
    print("\n========================================")
    print(" Evaluating PI-MLP Model ")
    print("========================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp_model_path = os.path.join(BASE_DIR, 'mitigation_ai', 'outputs', 'mlp_mitiq', 'best_finetuned_model.pth')
    
    ckpt = torch.load(mlp_model_path, map_location=device, weights_only=False)
    
    # Apply EXACT SAME scaling from training
    scaler_mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
    
    # We can reuse the unscaled features from LightGBM DataFrame, because they are exactly the same
    # But PyTorch expects specific column order (augmented columns). Let's build them to be safe.
    nq        = X_lgbm_raw[:, 0].astype(np.float64)
    d         = X_lgbm_raw[:, 1].astype(np.float64)
    e1q       = X_lgbm_raw[:, 2].astype(np.float64)
    e2q       = X_lgbm_raw[:, 3].astype(np.float64)
    T1        = X_lgbm_raw[:, 4].astype(np.float64)
    T2        = X_lgbm_raw[:, 5].astype(np.float64)
    t1q_ns    = X_lgbm_raw[:, 6].astype(np.float64)
    t2q_ns    = X_lgbm_raw[:, 7].astype(np.float64)
    p_leak    = X_lgbm_raw[:, 12].astype(np.float64)
    p_01      = X_lgbm_raw[:, 13].astype(np.float64)
    p_10      = X_lgbm_raw[:, 14].astype(np.float64)
    p_xt      = X_lgbm_raw[:, 15].astype(np.float64)
    beta_nm   = X_lgbm_raw[:, 16].astype(np.float64)
    theta_err = X_lgbm_raw[:, 17].astype(np.float64)
    
    FRAC_1Q = 2.0 / 3.0
    FRAC_2Q = 1.0 / 3.0
    t1q_us = t1q_ns * 1e-3
    t2q_us = t2q_ns * 1e-3
    N_total  = nq * d
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
    
    survival_combined = (gamma_dep * gamma_T1 * gamma_T2 * gamma_leak * gamma_xtalk * gamma_coh * readout_fac).astype(np.float32)
    log_surv = np.log(np.clip(np.abs(survival_combined), 1e-10, 1.0)).astype(np.float32)
    
    e_noisy = X_lgbm_raw[:, 19:24]
    e1, e3, e5 = e_noisy[:, 0], e_noisy[:, 1], e_noisy[:, 2]
    zne_extrap = ((3.0 * e1 - e3) / 2.0).astype(np.float32)
    lam_dev    = np.array([-4., -2., 0., 2., 4.], dtype=np.float32)
    e_mean     = e_noisy.mean(axis=1)
    b_coef     = (e_noisy * lam_dev).sum(axis=1) / 40.0
    zne_linear = (e_mean - b_coef * 5.0).astype(np.float32)
    zne_quad = ((15.0 * e1 - 10.0 * e3 + 3.0 * e5) / 8.0).astype(np.float32)
    shot_noise_variance = ((1.0 - zne_extrap**2) / 8192.0).astype(np.float32)
    
    X_aug = np.concatenate([
        X_lgbm_raw.astype(np.float32), 
        survival_combined[:, None], log_surv[:, None], zne_extrap[:, None],
        zne_linear[:, None], zne_quad[:, None], p_decay_T1[:, None],
        shot_noise_variance[:, None]
    ], axis=1)
    
    aug_names = fn_lgbm + [
        "survival_combined", "log_survival", "zne_extrap",
        "zne_linear", "zne_quad", "p_decay_T1", "shot_noise_variance"
    ]
    
    PRUNE_COLS = {"time_1q_ns", "time_2q_ns", "T1_us", "T2_us"}
    prune_mask = [i for i, n in enumerate(aug_names) if n in PRUNE_COLS]
    if prune_mask:
        keep = [i for i in range(len(aug_names)) if i not in prune_mask]
        X_aug = X_aug[:, keep]
        aug_names = [aug_names[i] for i in keep]
    
    # Filter with mask >= 0.3
    X_aug_filtered = X_aug[mask]
    y_raw_filtered = y_raw[mask]
    
    # Scale using loaded scaler
    no_scale_names = {
        "exp_val_scale_1", "exp_val_scale_3", "exp_val_scale_5",
        "exp_val_scale_7", "exp_val_scale_9",
        "zne_extrap", "zne_linear", "zne_quad", "shot_noise_variance"
    }
    zne_idx = [i for i, n in enumerate(aug_names) if n in no_scale_names]
    static_idx = [i for i in range(X_aug_filtered.shape[1]) if i not in zne_idx]
    
    X_scaled = X_aug_filtered.copy()
    X_scaled[:, static_idx] = (X_aug_filtered[:, static_idx] - scaler_mean) / scaler_scale
    
    input_dim = (X_scaled.shape[1] - 5) + 20
    mlp_model = QuantumMitigationMLP(input_dim=input_dim).to(device)
    if 'model_state' in ckpt:
        mlp_model.load_state_dict(ckpt['model_state'])
    else:
        mlp_model.load_state_dict(ckpt)
    mlp_model.eval()
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_raw_filtered, dtype=torch.float32).to(device)
    
    test_ds = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1024, shuffle=False)
    
    y_pred_mlp = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds = mlp_model(X_batch)[:, 0]
            y_pred_mlp.extend(preds.cpu().numpy())
            
    y_pred_mlp = np.array(y_pred_mlp)
    mae_mlp = mean_absolute_error(y_raw_filtered, y_pred_mlp)
    r2_mlp = r2_score(y_raw_filtered, y_pred_mlp)
    
    print(f"Tested on {len(X_tensor)} samples (filtered >= {sur_filter}).")
    print(f"PI-MLP Test MAE: {mae_mlp:.6f}")
    print(f"PI-MLP Test R²:  {r2_mlp:.6f}")
    
    # Generate scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.tick_params(colors="#aaaacc")
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.title.set_color("white")
        ax.grid(True, alpha=0.12, color="#ffffff")

    # LightGBM Scatter
    sc1 = ax1.scatter(y_lgbm, y_pred_lgbm, c=np.abs(y_pred_lgbm - y_lgbm), 
                      cmap="plasma", alpha=0.45, s=10, rasterized=True)
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label("|Error|", color="#aaaacc")
    cbar1.ax.yaxis.set_tick_params(color="#aaaacc")
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color="#aaaacc")
    ax1.plot([-1.1, 1.1], [-1.1, 1.1], color="#7ec8e3", lw=1.5, linestyle="--")
    ax1.set_xlim([-1.1, 1.1]); ax1.set_ylim([-1.1, 1.1])
    ax1.set_title(f"LightGBM\nR²={r2_lgbm:.4f} | MAE={mae_lgbm:.4f}")
    ax1.set_xlabel("Ideal Expectation Value")
    ax1.set_ylabel("Predicted Expectation Value")

    # MLP Scatter
    sc2 = ax2.scatter(y_raw_filtered, y_pred_mlp, c=np.abs(y_pred_mlp - y_raw_filtered), 
                      cmap="plasma", alpha=0.45, s=10, rasterized=True)
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label("|Error|", color="#aaaacc")
    cbar2.ax.yaxis.set_tick_params(color="#aaaacc")
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="#aaaacc")
    ax2.plot([-1.1, 1.1], [-1.1, 1.1], color="#7ec8e3", lw=1.5, linestyle="--")
    ax2.set_xlim([-1.1, 1.1]); ax2.set_ylim([-1.1, 1.1])
    ax2.set_title(f"PI-MLP\nR²={r2_mlp:.4f} | MAE={mae_mlp:.4f}")
    ax2.set_xlabel("Ideal Expectation Value")
    ax2.set_ylabel("Predicted Expectation Value")

    plt.tight_layout()
    scatter_path = os.path.join(BASE_DIR, 'model_comparison_scatter.png')
    plt.savefig(scatter_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    
if __name__ == '__main__':
    main()
