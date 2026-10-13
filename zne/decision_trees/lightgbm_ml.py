import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def engineer_lgbm_features(df):
    df = df.copy()

    # 1. Extrapolations
    e1 = df['exp_val_scale_1']
    e3 = df['exp_val_scale_3']
    e5 = df['exp_val_scale_5']
    e7 = df['exp_val_scale_7']
    e9 = df['exp_val_scale_9']
    
    df['zne_extrap'] = (3.0 * e1 - e3) / 2.0
    df['zne_quad'] = (15.0 * e1 - 10.0 * e3 + 3.0 * e5) / 8.0
    
    # Linear OLS
    e_mean = (e1 + e3 + e5 + e7 + e9) / 5.0
    b_coef = (e1 * -4.0 + e3 * -2.0 + e5 * 0.0 + e7 * 2.0 + e9 * 4.0) / 40.0
    df['zne_linear'] = e_mean - b_coef * 5.0

    # 2. Survival Probabilities
    FRAC_1Q = 2.0 / 3.0
    FRAC_2Q = 1.0 / 3.0
    
    t1q_us = df['time_1q_ns'] * 1e-3
    t2q_us = df['time_2q_ns'] * 1e-3
    N_total = df['num_qubits'] * df['depth']
    t_eff_us = FRAC_1Q * N_total * t1q_us + FRAC_2Q * N_total * t2q_us
    
    eps_eff = FRAC_1Q * df['error_1q'] + FRAC_2Q * df['error_2q']
    eps_N = eps_eff * N_total
    
    T1 = np.maximum(df['T1_us'], 1e-9)
    T2 = np.maximum(df['T2_us'], 1e-9)
    
    gamma_dep = np.exp(-(np.where(eps_N > 0, eps_N, 0.0) ** df['beta_nm']))
    gamma_T1 = np.exp(-t_eff_us / T1)
    gamma_T2 = np.exp(-t_eff_us / T2)
    gamma_leak = (1.0 - df['p_leak']) ** N_total
    
    n_pairs = df['num_qubits'] * (df['num_qubits'] - 1) / 2.0
    gamma_xtalk = np.exp(-df['p_xtalk'] * n_pairs * df['depth'])
    
    readout_fac = 1.0 - df['p_01'] - df['p_10']
    gamma_coh = np.cos(df['theta_error'] * N_total)
    
    df['survival_combined'] = (
        gamma_dep * gamma_T1 * gamma_T2 * gamma_leak
        * gamma_xtalk * gamma_coh * readout_fac
    )
    df['log_survival'] = np.log(np.clip(np.abs(df['survival_combined']), 1e-10, 1.0))
    
    # 3. Original features
    df['delta_1_3'] = e1 - e3
    shot_noise_var = 1e-4
    df['snr_1'] = np.abs(e1) / np.sqrt(shot_noise_var + 1e-8)
    df['volume'] = N_total
    df['total_gate_area'] = df['error_2q'] * N_total
    df['estimated_duration_ns'] = df['depth'] * ((df['time_1q_ns'] + df['time_2q_ns']) / 2.0)
    df['t1_exhaustion'] = df['estimated_duration_ns'] / (df['T1_us'] * 1000.0)
    df['t2_exhaustion'] = df['estimated_duration_ns'] / (df['T2_us'] * 1000.0)
    df['zne_curvature'] = (e1 - e3) - (e3 - e5)
    df['extrap_divergence'] = np.abs(df['zne_linear'] - df['zne_quad'])
    df['readout_bias'] = df['p_01'] - df['p_10']
    df['alignment_interaction'] = df['observable_alignment'] * e1
    df['shot_noise_variance'] = (1.0 - df['zne_extrap']**2) / 8192.0

    return df


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
    
    print("Loading dataset...")
    dataset_path = os.path.join(ROOT_DIR, 'zne', 'dataset', 'vector_dataset.npz')
    data = np.load(dataset_path)
    feature_names = list(data['feature_names'])
    X = data['X']
    y = data['y']

    df = pd.DataFrame(X, columns=feature_names)
    df['ideal_exp_val'] = y

    print(f"Dataset shape: {df.shape}")
    print(f"Features: {len(feature_names)}")

    print("Engineering features...")
    df = engineer_lgbm_features(df)
    
    min_survival = 0.1
    mask = np.abs(df['survival_combined']) >= min_survival
    kept = mask.sum()
    print(f"Survival filter (>= {min_survival}): kept {kept:,} / {len(mask):,} samples")
    df = df[mask]

    target_col = 'ideal_exp_val'
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values
    y = df[target_col].values

    print("Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_cols, reference=train_data)

    survival_idx = feature_cols.index('survival_combined') if 'survival_combined' in feature_cols else -1
    monotone_constraints = [0] * len(feature_cols)
    if survival_idx >= 0:
        monotone_constraints[survival_idx] = 1
    monotone_constraints = tuple(monotone_constraints)

    params = {
        'objective': 'regression',
        'metric': ['rmse', 'l1'],
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_gain_to_split': 0.01,
        'verbose': -1,
        'monotone_constraints': monotone_constraints,
        'seed': 42,
        'deterministic': True,
        'force_col_wise': True,
    }

    print("Training LightGBM model...")
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(evals_result)
        ],
    )

    print(f"Best iteration: {model.best_iteration}")

    print("Evaluating on test set...")
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MAE: {mae:.6f}")
    print(f"Test R²: {r2:.6f}")

    print("Plotting learning curves (RMSE & MAE)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
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

    ax1.plot(evals_result['train']['rmse'], label='Train RMSE', color='#7ec8e3')
    ax1.plot(evals_result['valid']['rmse'], label='Valid RMSE', color='#ff6b6b')
    ax1.set_xlabel('Boosting Iterations')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Root Mean Squared Error (Log Scale)')
    ax1.legend(framealpha=0.2, labelcolor="white")
    ax1.set_yscale('log')

    ax2.plot(evals_result['train']['l1'], label='Train MAE', color='#7ec8e3')
    ax2.plot(evals_result['valid']['l1'], label='Valid MAE', color='#ff6b6b')
    ax2.set_xlabel('Boosting Iterations')
    ax2.set_ylabel('MAE')
    ax2.set_title('Mean Absolute Error (Log Scale)')
    ax2.legend(framealpha=0.2, labelcolor="white")
    ax2.set_yscale('log')

    plt.tight_layout()
    curve_path = os.path.join(BASE_DIR, 'lgbm_learning_curve.png')
    plt.savefig(curve_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved learning curves to {curve_path}")

    print("Plotting residuals distribution...")
    residuals = y_pred - y_test
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(residuals)
    x_range = np.linspace(min(residuals), max(residuals), 500)
    y_kde = kde(x_range)
    
    ax.plot(x_range, y_kde, color='#7ec8e3', lw=2)
    ax.fill_between(x_range, y_kde, color='#7ec8e3', alpha=0.3)
    
    ax.axvline(0, color='#ff6b6b', linestyle='--', linewidth=2, label="Zero Error")
    
    res_mae = float(np.abs(residuals).mean())
    res_std = float(np.std(residuals))
    
    ax.set_title(f'Test Set: Error Residuals Distribution\nMean: {np.mean(residuals):.5f} | Std: {res_std:.5f} | MAE: {res_mae:.5f}', color="white")
    ax.set_xlabel("Error (Predicted - Ideal)", color="#aaaacc")
    ax.set_ylabel("Density", color="#aaaacc")
    ax.tick_params(colors="#aaaacc")
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="white")
    ax.grid(True, alpha=0.12, color="#ffffff")

    plt.tight_layout()
    resid_path = os.path.join(BASE_DIR, 'lgbm_residuals.png')
    plt.savefig(resid_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved residuals plot to {resid_path}")

    print("Plotting scatter plot...")
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    sc = ax.scatter(y_test, y_pred, c=np.abs(y_pred - y_test), cmap="plasma",
                    alpha=0.45, s=12, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("|Error|", color="#aaaacc")
    cbar.ax.yaxis.set_tick_params(color="#aaaacc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaacc")

    lims = [-1.05, 1.05]
    ax.plot(lims, lims, color="#7ec8e3", lw=1.5, linestyle="--",
            label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)

    ax.set_title(f"Test Set: Predicted vs Ideal\nR²={r2:.4f}   MAE={mae:.5f}",
                 color="white", fontsize=12)
    ax.set_xlabel("Ideal Expectation Value",   color="#aaaacc", fontsize=11)
    ax.set_ylabel("Mitigated (Predicted)",     color="#aaaacc", fontsize=11)
    ax.tick_params(colors="#aaaacc")
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="white")
    ax.grid(True, alpha=0.12, color="#ffffff")

    scatter_path = os.path.join(BASE_DIR, 'lgbm_scatter.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved scatter plot to {scatter_path}")

    print("Plotting feature importance...")
    fig, ax = plt.subplots(figsize=(10, 12))
    lgb.plot_importance(model, ax=ax, importance_type='gain', max_num_features=30)
    plt.title('LightGBM Feature Importance (Gain)')
    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, 'lgbm_feature_importance.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to {plot_path}")

    model_path = os.path.join(BASE_DIR, 'lgbm_model.txt')
    model.save_model(model_path)
    print(f"Saved model to {model_path}")


if __name__ == '__main__':
    main()