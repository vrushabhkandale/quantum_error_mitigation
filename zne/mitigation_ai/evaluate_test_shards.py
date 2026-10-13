import sys
import os
import argparse
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

# Add project root to PYTHONPATH so we can import modules
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR.parent))

from zne.mitigation_ai.gnn import (
    QEMFormerGNN,
    PhysicsInformedLoss,
    evaluate,
    plot_predictions_scatter,
    plot_residual_histogram,
    plot_gamma_histogram,
    plot_attention_heatmap,
    plot_error_vs_depth,
    plot_zne_trajectories,
    plot_feature_importance,
    DEFAULT_LAMBDA_BNDRY,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    GLOBAL_FEAT_DIM,
)
from zne.dataset.dag_data import ShardedDAGIterableDataset

def main():
    parser = argparse.ArgumentParser(description="Evaluate QEMFormerGNN on test shards")
    parser.add_argument("--batch", type=int, default=8192)
    args = parser.parse_args()

    # Paths
    test_shard_dir = ROOT_DIR / "dataset" / "test_shards"
    model_path = ROOT_DIR / "mitigation_ai" / "outputs" / "gnn" / "best_gnn_model.pth"
    output_dir = ROOT_DIR / "mitigation_ai" / "outputs" / "gnn_test_shards"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Windows cp1252 console can't encode Unicode math symbols
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Loading test shards from: {test_shard_dir}")
    if not os.path.exists(test_shard_dir) or not os.path.exists(test_shard_dir / "index.json"):
        print(f"Error: test_shard_dir {test_shard_dir} or index.json does not exist.")
        sys.exit(1)

    test_ds = ShardedDAGIterableDataset(str(test_shard_dir), shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    
    _loader_kwargs = dict(
        batch_size=args.batch,
        num_workers=max(4, min(8, (os.cpu_count() or 4) // 2)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=8,
    )
    test_loader = PyGDataLoader(test_ds, **_loader_kwargs)
    
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: model file {model_path} does not exist.")
        sys.exit(1)

    model = QEMFormerGNN(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        global_feat_dim=GLOBAL_FEAT_DIM,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        global_dim=64,
        fusion_dim=256,
        dropout=0.10,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Fix 4: huber_delta=1.0 is the current setting
    criterion = PhysicsInformedLoss(lambda_bndry=DEFAULT_LAMBDA_BNDRY, huber_delta=1.0)
    
    print("\nRunning evaluation on test shards...")
    test_results = evaluate(
        model, test_loader, criterion, device,
        use_amp=use_amp, collect_details=True,
    )
    
    print("\n" + "=" * 75)
    print("  Test Shards Evaluation Results")
    print("=" * 75)
    print(f"  Test Loss : {test_results['loss']:.6f}")
    print(f"  Test MAE  : {test_results['mae']:.6f}")
    print(f"  Test RMSE : {test_results['rmse']:.6f}")
    print(f"  Test R2   : {test_results['r2']:.6f}")
    
    details = test_results["details"]
    
    print("\n" + "=" * 75)
    print("  Generating Diagnostic Plots for Test Shards")
    print("=" * 75)
    
    try:
        plot_predictions_scatter(details["pred"], details["ideal"], output_dir)
    except Exception as e:
        print(f"    [Error] Predictions scatter: {e}")

    try:
        plot_residual_histogram(details["y_res"], details["delta_hat"], output_dir)
    except Exception as e:
        print(f"    [Error] Residual histogram: {e}")

    try:
        plot_gamma_histogram(details["gamma"], output_dir)
    except Exception as e:
        print(f"    [Error] Gamma histogram: {e}")
    
    try:
        print("    Extracting attention weights...")
        sample_batch = None
        for batch in test_loader:
            sample_batch = batch
            break
        if sample_batch is not None:
            plot_attention_heatmap(model, sample_batch, device, output_dir, num_heads=4)
    except Exception as e:
        print(f"    [Error] Attention heatmap: {e}")
        
    try:
        error = details["pred"] - details["ideal"]
        plot_error_vs_depth(error, details["depth"], output_dir)
    except Exception as e:
        print(f"    [Error] Error vs depth: {e}")

    try:
        plot_zne_trajectories(test_loader, model, device, output_dir, n_samples=6)
    except Exception as e:
        print(f"    [Error] ZNE trajectories: {e}")
    
    try:
        print("    Computing feature importance (permutation method)...")
        plot_feature_importance(model, test_loader, device, output_dir, use_amp=use_amp)
    except Exception as e:
        print(f"    [Error] Feature importance: {e}")
    
    print(f"\n  Outputs saved to : {output_dir}")
    print("=" * 75)

if __name__ == "__main__":
    main()
