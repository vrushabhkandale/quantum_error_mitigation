"""
gnn.py
======
QEMFormer Graph Neural Network — Dual-Branch Graph Transformer for
Quantum Error Mitigation via TransformerConv.

Architecture (Dual-Branch Fusion):
    ┌─────────────────────────────────────────────────────────────┐
    │               Gate-DAG Input (PyG Data)                     │
    ├──────────────────────────┬──────────────────────────────────┤
    │  Branch A: Graph         │  Branch B: Global Context        │
    │  Transformer             │  MLP (ZNE Dropout=50%)           │
    │                          │                                  │
    │  NodeEncoder(15→128)     │  ZNE Dropout(p=0.50, cols 4–8) │
    │  EdgeEncoder(4→128)      │  Linear(9→128)→GELU              │
    │  4× TransformerConv      │  Linear(128→128)→GELU            │
    │     (edge-conditioned    │  Linear(128→64)                  │
    │      multi-head attn)    │                                  │
    │  ObsCentric Readout      │        │                         │
    │       │                  │        │                         │
    │       └──────────┬───────┘        │                         │
    │                  │                │                         │
    │          Fusion (Concat)                                    │
    │          Linear(192→256)→GELU                               │
    │          Linear(256→128)→GELU                               │
    │          Linear(128→1)  → predicted error residual Δ̂        │
    └─────────────────────────────────────────────────────────────┘

    Inference: E_mitigated = Δ̂ + baseline(λ=1)

Branch A — Topological Graph Transformer:
    Processes the gate-level Directed Acyclic Graph (DAG) from dag_data.py.
    Each gate is a node with 15 features (one-hot gate type, T1/T2, error
    rate, SPAM rates, normalised layer depth). Directed edges represent
    causal qubit-wire flow with 4 features (wire index, layer gap, idle
    time, CX directionality).

    TransformerConv injects edge features directly into the attention score:
    α_ij is conditioned on the physical coupling error on the connecting wire.
    Multi-head attention (4 heads) disentangles orthogonal noise channels:
    Head 1 traces coherent phase errors, Head 2 routes ZZ-crosstalk, etc.

    Fix 3 — Positional Encodings:
    Normalised layer depth (col 14 of node features) breaks the temporal
    symmetry of attention, preventing collapse to a uniform mean-aggregator.

    Fix 1 — Observable-Centric Readout:
    Instead of global_mean_pool (which averages ALL gate embeddings), only
    the terminal observable-gate embeddings (data.obs_mask == True) are
    pooled. This anchors the readout to the physical measurement resonators
    and preserves causal locality. Falls back to global_mean_pool when
    obs_mask is absent (old shards).

Branch B — Global Context MLP:
    Processes macroscopic circuit metadata from data.u (9 features):
    depth, beta_nm, idle_frac, log_shots, + 5 ZNE expectation values.
    Captures the baseline decay trajectory before spatial defects are factored in.

    Fix 2 — ZNE Feature Dropout (50%):
    During training, the 5 ZNE features (indices 4–8) are randomly zeroed
    with p=0.50, forcing gradient signal into the graph branch rather than
    allowing the MLP to shortcut via curve-fitting ZNE values.

Residual Prediction:
    The model predicts Δ̂ = E_ideal − E_noisy(λ=1), NOT E_ideal directly.
    This forces the graph branch to explain WHERE and HOW error accumulated
    through the topological structure, preventing linear shortcut exploitation.

Physics-Informed Loss:
    Fix 4 — Huber delta raised from 0.5 → 1.0 (larger gradient signal for
    recoverable circuits); boundary penalty weight lowered from 0.10 → 0.05.
    L_total = HuberLoss(Δ̂, Δ_true; δ=1.0) + λ_bndry × boundary_penalty
    boundary_penalty = mean(ReLU(|E_mitigated| − 1))

Data Pipeline:
    Consumes ShardedDAGIterableDataset from dag_data.py (1M samples, 20 shards).
    Train/Val/Test split by shard index: shards 0-15 train, 16-17 val, 18-19 test.

Diagnostic Plots (8):
    1. 2×2 Training Curves (Loss, MAE, RMSE, R²)
    2. Predicted vs Ideal Scatter (color-coded by |error|)
    3. Error Residual Histogram (Δ_true vs Δ̂)
    4. Survival Probability (γ) Histogram
    5. Attention Weight Heatmap (multi-head α_ij)
    6. Error vs Circuit Depth Box Plot
    7. ZNE Trajectory Samples (raw ZNE + mitigated point)
    8. Feature Importance via Permutation (ΔMAE per feature group)

Usage:
    python zne/mitigation_ai/gnn.py
    python zne/mitigation_ai/gnn.py --epochs 200 --lr 3e-4 --batch 256
    python zne/mitigation_ai/gnn.py --epochs 2 --max-shards 1  # smoke test

Outputs (saved to zne/mitigation_ai/outputs/gnn/):
    best_gnn_model.pth          — best checkpoint (lowest val loss)
    final_gnn_model.pth         — model after all epochs
    training_log.csv            — per-epoch metrics
    training_curves.png         — 2×2 metric grid
    predictions_scatter.png     — predicted vs ideal with error colormap
    residual_histogram.png      — error residual distribution
    gamma_histogram.png         — survival probability distribution
    attention_heatmap.png       — TransformerConv attention weights
    error_vs_depth.png          — error by circuit depth
    zne_trajectories.png        — sample ZNE curves + mitigated values
    feature_importance.png      — permutation-based feature group importance
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import threading

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import scatter as pyg_scatter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR   = Path(__file__).resolve().parent.parent              # zne/
SHARD_DIR  = ROOT_DIR / "dataset" / "1m_shards"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "gnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters (overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS       = 200
DEFAULT_LR           = 3e-4
DEFAULT_BATCH        = 8192
DEFAULT_SEED         = 42
DEFAULT_PATIENCE     = 5      # early stopping patience
DEFAULT_LAMBDA_BNDRY = 0.05     # boundary penalty weight (Fix 4: was 0.10)
DEFAULT_DROPOUT      = 0.10
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP    = 1.0

# Architecture
DEFAULT_HIDDEN_DIM   = 128      # TransformerConv hidden dimension
DEFAULT_NUM_HEADS    = 4        # multi-head attention heads
DEFAULT_NUM_LAYERS   = 8        # TransformerConv layers
DEFAULT_GLOBAL_DIM   = 64       # global MLP output dimension
DEFAULT_FUSION_DIM   = 256      # fusion MLP hidden dimension

# Data
DEFAULT_MAX_SHARDS   = None     # None = use all shards

# Feature dimensions from dag_data.py
NODE_FEAT_DIM   = 15   # Fix 3: +1 for normalized layer depth (was 14)
EDGE_FEAT_DIM   = 4
GLOBAL_FEAT_DIM = 9
SCALE_FACTORS   = [1, 3, 5, 7, 9]


# ===========================================================================
# Model Architecture
# ===========================================================================

class NodeEncoder(nn.Module):
    """Projects 15-dim node features into the hidden dimension.

    Fix 3: Input dimension updated from 14 to 15 to accommodate the new
    normalized layer-depth positional encoding (col 14).
    Old shards (14 features) are zero-padded in GraphTransformerBranch.forward().
    """
    def __init__(self, in_dim: int = NODE_FEAT_DIM, hidden_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


class EdgeEncoder(nn.Module):
    """Projects 4-dim edge features into the hidden dimension."""
    def __init__(self, in_dim: int = EDGE_FEAT_DIM, hidden_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(edge_attr))


class TransformerConvBlock(nn.Module):
    """
    Single TransformerConv layer with residual connection, LayerNorm, and dropout.

    TransformerConv injects edge features directly into the attention score
    calculation. The attention weight α_ij determining how much noise flows
    from gate j to gate i is actively scaled by the physical coupling error
    on the edge connecting them.

    With 4 heads, the layer can simultaneously track:
      - Head 1: coherent phase errors along wires
      - Head 2: ZZ-crosstalk across CX junctions
      - Head 3: T1/T2 decoherence cascades
      - Head 4: gate error accumulation
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        edge_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,        # concatenate head outputs → hidden_dim
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward sub-layer (matches standard Transformer architecture)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm residual connection (more stable training)
        h = self.norm1(x)
        h = self.conv(h, edge_index, edge_attr)
        x = x + h

        # Feed-forward with pre-norm residual
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        return x


class GraphTransformerBranch(nn.Module):
    """
    Branch A: Topological Graph Transformer.

    Processes the gate-DAG through stacked TransformerConv layers,
    then condenses the topology into a single dense vector via an
    observable-centric masked readout (Fix 1) or global mean pooling
    as a fallback when obs_mask is absent (old shards).

    Fix 3 — Backward compatibility:
        If data.x has 14 features (old shards), a zero-padded 15th column is
        appended at runtime so the NodeEncoder(15→128) still runs correctly.
        The missing layer-depth information degrades Fix 3, but the model
        remains fully functional on old data.
    """
    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.node_encoder = NodeEncoder(node_feat_dim, hidden_dim)
        self.edge_encoder = EdgeEncoder(edge_feat_dim, hidden_dim)

        self.conv_layers = nn.ModuleList([
            TransformerConvBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                edge_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Fix 3: Backward-compat zero-pad if old shard has 14-feature nodes
        if x.shape[1] == NODE_FEAT_DIM - 1:
            x = torch.cat([x, torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=1)

        # Encode node and edge features into hidden dimension
        h = self.node_encoder(x)
        edge_h = self.edge_encoder(edge_attr)

        # Stacked TransformerConv message passing
        for conv_block in self.conv_layers:
            h = conv_block(h, edge_index, edge_h)

        h = self.final_norm(h)

        # Fix 1: Observable-centric readout — pool ONLY terminal observable gates.
        # Falls back to global_mean_pool when obs_mask is absent (old shards).
        if obs_mask is not None and obs_mask.any():
            h_masked = h.clone()
            h_masked[~obs_mask] = 0.0
            # Sum observable embeddings per graph, divide by observable gate count
            # pyg_scatter: (src, index, dim, dim_size, reduce)
            n_graphs  = int(batch.max().item()) + 1
            obs_count = pyg_scatter(obs_mask.float(), batch, dim=0,
                                    dim_size=n_graphs, reduce="sum").clamp(min=1.0)  # [B]
            obs_sum   = pyg_scatter(h_masked, batch, dim=0,
                                    dim_size=n_graphs, reduce="sum")                 # [B, D]
            graph_embed = obs_sum / obs_count.unsqueeze(-1)
        else:
            # Fallback: global mean pool (old shards lacking obs_mask)
            graph_embed = global_mean_pool(h, batch)  # [batch_size, hidden_dim]

        return graph_embed


class GlobalContextBranch(nn.Module):
    """
    Branch B: Global Context MLP.

    Processes macroscopic circuit metadata (depth, beta_nm, idle_frac,
    log_shots) and multi-scale ZNE expectation values. Captures the
    baseline decay trajectory before spatial defects are factored in.

    Fix 2 — ZNE Feature Dropout:
        During training, the 5 ZNE features (indices 4–8 of u) are randomly
        zeroed with probability zne_dropout=0.50. This prevents the MLP from
        shortcutting via ZNE curve-fitting, forcing gradient updates into the
        GraphTransformerBranch to recover error rates from topology.
        The first 4 global features (depth, beta_nm, idle_frac, log_shots)
        are never dropped.
    """
    def __init__(
        self,
        global_feat_dim: int = GLOBAL_FEAT_DIM,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.10,
        zne_dropout: float = 0.50,     # Fix 2: heavy dropout on ZNE features only
    ):
        super().__init__()
        self.zne_dropout = zne_dropout
        self.net = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [batch_size, 1, GLOBAL_FEAT_DIM] → squeeze
        if u.dim() == 3:
            u = u.squeeze(1)

        # Fix 2: Apply heavy dropout ONLY to ZNE features (indices 4:9) during training.
        # The first 4 global features (depth, beta_nm, idle_frac, log_shots) are kept.
        if self.training and self.zne_dropout > 0.0:
            mask = torch.ones_like(u)
            zne_mask = torch.bernoulli(
                torch.full((u.size(0), 5), 1.0 - self.zne_dropout, device=u.device)
            )
            mask[:, 4:9] = zne_mask
            u = u * mask

        return self.net(u)  # [batch_size, output_dim]


class QEMFormerGNN(nn.Module):
    """
    QEMFormer Graph Neural Network — Dual-Branch Architecture.

    Fuses topological graph features (via TransformerConv message passing)
    with global circuit metadata (via MLP) to predict the error residual
    Δ̂ = E_ideal − E_noisy(λ=1).

    During inference: E_mitigated = Δ̂ + baseline(λ=1)
    """
    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        global_dim: int = 64,
        fusion_dim: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()

        # Branch A: Graph Transformer
        self.graph_branch = GraphTransformerBranch(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Branch B: Global Context MLP
        self.global_branch = GlobalContextBranch(
            global_feat_dim=global_feat_dim,
            hidden_dim=128,
            output_dim=global_dim,
            dropout=dropout,
        )

        # Fusion + Residual Readout Head
        fusion_input_dim = hidden_dim + global_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming normal for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass on a batched PyG Data object.

        Returns:
            predicted residual Δ̂ of shape [batch_size, 1]
        """
        # Fix 1: Pass obs_mask (may be None for old shards — fallback handled in branch)
        obs_mask = getattr(data, "obs_mask", None)

        # Branch A: Graph Transformer on gate-DAG topology
        graph_embed = self.graph_branch(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
            obs_mask=obs_mask,
        )  # [batch_size, hidden_dim]

        # Branch B: Global Context MLP on circuit metadata + ZNE
        global_embed = self.global_branch(data.u)  # [batch_size, global_dim]

        # Fusion: concatenate and regress to residual
        fused = torch.cat([graph_embed, global_embed], dim=1)
        delta_hat = self.fusion(fused)  # [batch_size, 1]

        return delta_hat

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Physics-Informed Loss
# ===========================================================================

class PhysicsInformedLoss(nn.Module):
    """
    Fix 4 — Updated loss hyperparameters:
        L_total = HuberLoss(Δ̂, Δ_true; δ=1.0) + λ_bndry × boundary_penalty

    Huber delta raised from 0.5 → 1.0: larger gradient signal for circuits
    with recoverable intermediate residuals; tighter clipping near outliers
    was prematurely suppressing gradient updates.

    lambda_bndry lowered from 0.10 → 0.05 via DEFAULT_LAMBDA_BNDRY:
    softer boundary pull prevents the loss from over-constraining predictions
    toward the centre of [-1, 1].

    Boundary penalty enforces that the reconstructed mitigated value stays
    within the physical range [−1, 1].
    """
    def __init__(self, lambda_bndry: float = 0.05, huber_delta: float = 1.0):
        super().__init__()
        self.lambda_bndry = lambda_bndry
        self.huber = nn.HuberLoss(reduction="mean", delta=huber_delta)

    def forward(
        self,
        delta_hat: torch.Tensor,   # [B, 1] predicted residual
        delta_true: torch.Tensor,  # [B]    true residual
        baseline: torch.Tensor,    # [B]    λ=1 baseline
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_hat_flat = delta_hat.squeeze(-1)

        # Huber loss on residual
        huber_loss = self.huber(delta_hat_flat, delta_true)

        # Boundary penalty: E_mitigated = Δ̂ + baseline must be in [−1, 1]
        e_mitigated = delta_hat_flat + baseline
        boundary_penalty = torch.relu(e_mitigated.abs() - 1.0).mean()

        total_loss = huber_loss + self.lambda_bndry * boundary_penalty

        return total_loss, huber_loss, boundary_penalty


# ===========================================================================
# Metrics
# ===========================================================================

def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """R² coefficient of determination."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return 0.0
    return (1.0 - ss_res / ss_tot).item()


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Root Mean Square Error."""
    return torch.sqrt(((y_pred - y_true) ** 2).mean()).item()


# ===========================================================================
# Data Pipeline — Shard-Based Train/Val/Test Split
# ===========================================================================

class SplitShardDataset(torch.utils.data.IterableDataset):
    """
    Streaming IterableDataset that loads a subset of shards for a given split.

    Shard assignment:
        train: shards 0–(n_shards - n_val - n_test - 1)
        val:   next n_val shards
        test:  last n_test shards

    Within each shard, graphs are shuffled in-memory (zero I/O cost).
    Shards are shuffled at epoch start for training.
    """
    def __init__(
        self,
        shard_dir: str,
        split: str = "train",
        n_val_shards: int = 2,
        n_test_shards: int = 2,
        max_shards: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        index_path = os.path.join(shard_dir, "index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        all_shards = index["shards"]
        n_total = len(all_shards)

        # Apply max_shards limit (for smoke testing)
        if max_shards is not None and max_shards < n_total:
            all_shards = all_shards[:max_shards]
            n_total = len(all_shards)
            # Adjust val/test shard counts for small datasets
            if n_total <= 4:
                n_val_shards = 1
                n_test_shards = 1
            if n_total <= 2:
                n_val_shards = 0
                n_test_shards = 0

        n_train = n_total - n_val_shards - n_test_shards
        assert n_train > 0, f"Not enough shards: {n_total} total, need {n_val_shards + n_test_shards + 1}"

        if split == "train":
            self._shard_info = all_shards[:n_train]
        elif split == "val":
            self._shard_info = all_shards[n_train:n_train + n_val_shards]
        elif split == "test":
            self._shard_info = all_shards[n_train + n_val_shards:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self._shard_dir = shard_dir
        self._split = split
        self._shuffle = shuffle and (split == "train")
        self._seed = seed
        self._epoch = 0
        self._total = sum(s["size"] for s in self._shard_info)

    def __len__(self) -> int:
        return self._total

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _load_shard(self, shard_idx: int) -> list:
        """Load a single shard from disk. Called from background threads."""
        path = os.path.join(self._shard_dir, self._shard_info[shard_idx]["path"])
        return torch.load(path, weights_only=False)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shard_indices = list(range(len(self._shard_info)))

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shard_indices = [s for s in shard_indices if s % num_workers == worker_id]

        worker_id_safe = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(
            self._seed + self._epoch * 1_000_003 + worker_id_safe
        )

        if self._shuffle:
            rng.shuffle(shard_indices)

        if not shard_indices:
            return

        # Load the very first shard synchronously (nothing else to do yet).
        graphs = self._load_shard(shard_indices[0])

        for i, shard_idx in enumerate(shard_indices):
            # `graphs` already holds the current shard's data (pre-loaded
            # either synchronously above for i==0, or by the background
            # thread from the previous iteration for i>0).

            # Kick off disk I/O for the NEXT shard in a background thread
            # so it overlaps with yielding the current shard's 50K graphs.
            next_result = [None]
            loader_thread = None
            if i + 1 < len(shard_indices):
                next_idx = shard_indices[i + 1]

                def _bg_load(idx=next_idx):
                    next_result[0] = self._load_shard(idx)

                loader_thread = threading.Thread(target=_bg_load, daemon=True)
                loader_thread.start()

            # Yield individual graphs — PyGDataLoader handles fast C++ collation.
            if self._shuffle:
                perm = rng.permutation(len(graphs)).tolist()
                for p in perm:
                    yield graphs[p]
            else:
                yield from graphs

            # Release current shard memory before picking up the next one.
            del graphs

            # Wait for the background loader (should already be done —
            # yielding 50K graphs takes longer than loading one shard).
            if loader_thread is not None:
                loader_thread.join()
                graphs = next_result[0]

    def __repr__(self) -> str:
        return (
            f"SplitShardDataset(split={self._split}, "
            f"n={self._total:,}, shards={len(self._shard_info)})"
        )


# ===========================================================================
# Async GPU Prefetcher — Dual-Stream Pipelining
# ===========================================================================

class CUDAPrefetcher:
    """
    Overlaps host→device PCIe transfer with GPU compute using a secondary
    CUDA stream.

    While the default stream executes forward/backward for batch N, the
    prefetch stream transfers batch N+1 across the PCIe bus in the background.
    When the model finishes batch N, batch N+1 is already resident in VRAM.

    On CPU (no CUDA), falls back transparently to a standard iterator.

    Usage
    -----
        for batch in CUDAPrefetcher(loader, device):
            # batch is already on `device`, no .to(device) needed
            ...
    """

    def __init__(self, loader: PyGDataLoader, device: torch.device) -> None:
        self._loader = loader
        self._device = device
        self._use_cuda = (device.type == "cuda")
        if self._use_cuda:
            self._stream = torch.cuda.Stream(device=device)
        self._next_batch = None
        self._iter: Optional[object] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preload(self) -> None:
        """Asynchronously load the *next* batch onto the device."""
        try:
            raw = next(self._iter)  # type: ignore[arg-type]
        except StopIteration:
            self._next_batch = None
            return

        if self._use_cuda:
            with torch.cuda.stream(self._stream):
                self._next_batch = raw.to(
                    self._device,
                    non_blocking=True,   # DMA — no CPU mediation
                )
        else:
            self._next_batch = raw.to(self._device)

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> "CUDAPrefetcher":
        if self._use_cuda:
            self._stream.synchronize()
        self._iter = iter(self._loader)
        self._preload()
        return self

    def __next__(self):
        if self._use_cuda:
            # Synchronise: wait until the prefetch stream has finished the
            # transfer before returning the batch to the caller.
            torch.cuda.current_stream(self._device).wait_stream(self._stream)

        batch = self._next_batch
        if batch is None:
            raise StopIteration

        if self._use_cuda:
            # Record a CUDA event on the batch tensors so that the prefetch
            # stream can safely start writing the *next* batch.
            batch.record_stream(torch.cuda.current_stream(self._device))

        # Kick off the asynchronous transfer of the *next* batch while the
        # GPU compute cores process the *current* batch.
        self._preload()
        return batch


# ===========================================================================
# Training & Evaluation
# ===========================================================================

def train_one_epoch(
    model: nn.Module,
    loader: PyGDataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PhysicsInformedLoss,
    device: torch.device,
    amp_scaler: Optional[torch.amp.GradScaler] = None,
    scheduler: Optional[OneCycleLR] = None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch. Returns dict of metrics."""
    model.train()
    total_loss = huber_sum = bndry_sum = 0.0
    abs_err_sum = sq_err_sum = y_sum = y_sq_sum = 0.0
    n_samples = 0
    use_amp = (device.type == "cuda" and amp_scaler is not None)

    # CUDAPrefetcher overlaps host→device transfer with GPU compute.
    # Each batch is already on `device` when the loop body starts.
    for batch in CUDAPrefetcher(loader, device):
        y_residual = batch.y         # [B] residual target
        baseline   = batch.baseline  # [B] λ=1 unmitigated

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            delta_hat = model(batch)  # [B, 1]
            loss, huber, bndry = criterion(delta_hat, y_residual, baseline)

        step_executed = True
        if use_amp:
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            # Record scale to check if the optimizer step is skipped due to inf/nan
            scale_before = amp_scaler.get_scale()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            scale_after = amp_scaler.get_scale()
            
            # If scale decreased, it means the optimizer step was skipped
            if scale_after < scale_before:
                step_executed = False
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        if scheduler is not None and step_executed:
            try:
                scheduler.step()
            except ValueError:
                pass  # OneCycleLR exhausted — ignore extra steps

        bs = int(y_residual.shape[0])
        n_samples  += bs
        total_loss += loss.detach() * bs
        huber_sum  += huber.detach() * bs
        bndry_sum  += bndry.detach() * bs

        # Reconstruct mitigated predictions for metrics
        with torch.no_grad():
            e_mitigated = delta_hat.squeeze(-1) + baseline
            e_ideal     = y_residual + baseline
            abs_err_sum += (e_mitigated - e_ideal).abs().sum()
            sq_err_sum += (e_mitigated - e_ideal).square().sum()
            y_sum += e_ideal.sum()
            y_sq_sum += e_ideal.square().sum()

    if n_samples == 0:
        return {"loss": 0, "huber": 0, "bndry": 0, "mae": 0, "rmse": 0, "r2": 0}

    mae_val = (abs_err_sum / n_samples).item()
    rmse_val = torch.sqrt(sq_err_sum / n_samples).item()
    ss_tot = y_sq_sum - ((y_sum ** 2) / n_samples)
    r2_val = (1.0 - (sq_err_sum / torch.clamp(ss_tot, min=1e-12))).item()

    return {
        "loss"  : (total_loss / n_samples).item(),
        "huber" : (huber_sum / n_samples).item(),
        "bndry" : (bndry_sum / n_samples).item(),
        "mae"   : mae_val,
        "rmse"  : rmse_val,
        "r2"    : r2_val,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: PyGDataLoader,
    criterion: PhysicsInformedLoss,
    device: torch.device,
    use_amp: bool = False,
    collect_details: bool = False,
) -> Dict:
    """
    Evaluate on a dataset. If collect_details=True, also returns per-sample
    predictions, ideals, baselines, gammas, and depths for diagnostic plots.
    """
    model.eval()
    total_loss = huber_sum = bndry_sum = 0.0
    abs_err_sum = sq_err_sum = y_sum = y_sq_sum = 0.0
    n_samples = 0

    # Detail collectors for diagnostic plots
    all_pred_detail   = [] if collect_details else None
    all_ideal_detail  = [] if collect_details else None
    all_baseline_detail = [] if collect_details else None
    all_gamma_detail  = [] if collect_details else None
    all_depth_detail  = [] if collect_details else None
    all_y_res_detail  = [] if collect_details else None
    all_delta_hat_detail = [] if collect_details else None

    for batch in CUDAPrefetcher(loader, device):
        # batch is pre-transferred to `device` by the prefetcher.
        y_residual = batch.y
        baseline   = batch.baseline

        with torch.autocast(device_type=device.type, enabled=use_amp):
            delta_hat = model(batch)
            loss, huber, bndry = criterion(delta_hat, y_residual, baseline)

        bs = int(y_residual.shape[0])
        n_samples  += bs
        total_loss += loss.detach() * bs
        huber_sum  += huber.detach() * bs
        bndry_sum  += bndry.detach() * bs

        e_mitigated = delta_hat.squeeze(-1) + baseline
        e_ideal     = y_residual + baseline
        
        abs_err_sum += (e_mitigated - e_ideal).abs().sum()
        sq_err_sum += (e_mitigated - e_ideal).square().sum()
        y_sum += e_ideal.sum()
        y_sq_sum += e_ideal.square().sum()

        if collect_details:
            all_pred_detail.append(e_mitigated.cpu())
            all_ideal_detail.append(batch.y_ideal.cpu())
            all_baseline_detail.append(baseline.cpu())
            all_gamma_detail.append(batch.gamma.cpu())
            # Depth is u[:, 0] — but u is [B, 1, 9], need per-graph depth
            # In batched data, data.u is [total_graphs_in_batch, 1, 9]
            depths = batch.u[:, 0, 0] if batch.u.dim() == 3 else batch.u[:, 0]
            all_depth_detail.append(depths.cpu())
            all_y_res_detail.append(y_residual.cpu())
            all_delta_hat_detail.append(delta_hat.squeeze(-1).cpu())

    if n_samples == 0:
        result = {"loss": 0, "huber": 0, "bndry": 0, "mae": 0, "rmse": 0, "r2": 0}
        if collect_details:
            result["details"] = {}
        return result

    mae_val = (abs_err_sum / n_samples).item()
    rmse_val = torch.sqrt(sq_err_sum / n_samples).item()
    ss_tot = y_sq_sum - ((y_sum ** 2) / n_samples)
    r2_val = (1.0 - (sq_err_sum / torch.clamp(ss_tot, min=1e-12))).item()

    result = {
        "loss"  : (total_loss / n_samples).item(),
        "huber" : (huber_sum / n_samples).item(),
        "bndry" : (bndry_sum / n_samples).item(),
        "mae"   : mae_val,
        "rmse"  : rmse_val,
        "r2"    : r2_val,
    }

    if collect_details:
        result["details"] = {
            "pred"     : torch.cat(all_pred_detail).numpy(),
            "ideal"    : torch.cat(all_ideal_detail).numpy(),
            "baseline" : torch.cat(all_baseline_detail).numpy(),
            "gamma"    : torch.cat(all_gamma_detail).numpy(),
            "depth"    : torch.cat(all_depth_detail).numpy(),
            "y_res"    : torch.cat(all_y_res_detail).numpy(),
            "delta_hat": torch.cat(all_delta_hat_detail).numpy(),
        }

    return result


# ===========================================================================
# Attention Weight Extraction
# ===========================================================================

def extract_attention_weights(
    model: QEMFormerGNN,
    sample_batch: Data,
    device: torch.device,
    layer_idx: int = 0,
) -> Optional[torch.Tensor]:
    """
    Extract multi-head attention weights from a specified TransformerConv layer
    for a single graph in the batch.

    Returns attention coefficients α_ij of shape [num_edges, num_heads],
    or None if extraction fails.
    """
    model.eval()
    sample_batch = sample_batch.to(device)

    # Get the specified TransformerConv block
    conv_blocks = model.graph_branch.conv_layers
    if layer_idx >= len(conv_blocks):
        return None

    # Forward pass through encoders
    with torch.no_grad():
        h = model.graph_branch.node_encoder(sample_batch.x)
        edge_h = model.graph_branch.edge_encoder(sample_batch.edge_attr)

        # Pass through layers up to and including the target layer
        for i, conv_block in enumerate(conv_blocks):
            if i == layer_idx:
                # Use return_attention_weights to get α_ij
                h_normed = conv_block.norm1(h)
                try:
                    _, (edge_index_out, alpha) = conv_block.conv(
                        h_normed,
                        sample_batch.edge_index,
                        edge_h,
                        return_attention_weights=True,
                    )
                    return alpha.cpu()  # [num_edges, num_heads]
                except Exception:
                    return None
            else:
                h = conv_block(h, sample_batch.edge_index, edge_h)

    return None


# ===========================================================================
# Diagnostic Plots
# ===========================================================================

def plot_training_curves(
    history: Dict[str, list],
    output_dir: Path,
) -> None:
    """Plot 1: 2×2 grid of training & validation curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("QEMFormer GNN — Training & Validation Curves", fontsize=14, fontweight="bold")

    metrics = [
        ("train_loss", "val_loss", "Total Loss (Huber + Boundary)", True),
        ("train_mae",  "val_mae",  "Mean Absolute Error (MAE)", False),
        ("train_rmse", "val_rmse", "Root Mean Square Error (RMSE)", False),
        ("train_r2",   "val_r2",   "R² Score", False),
    ]

    for ax, (train_key, val_key, title, use_log) in zip(axes.flat, metrics):
        ax.plot(epochs, history[train_key], label="Train", linewidth=1.5, color="#2196F3")
        ax.plot(epochs, history[val_key],   label="Val",   linewidth=1.5, color="#FF5722",
                linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_predictions_scatter(
    pred: np.ndarray,
    ideal: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot 2: Predicted vs Ideal scatter, color-coded by |error|."""
    error = np.abs(pred - ideal)

    fig, ax = plt.subplots(figsize=(8, 8))

    sc = ax.scatter(
        ideal, pred,
        c=error,
        cmap="inferno_r",
        s=2, alpha=0.4,
        norm=Normalize(vmin=0, vmax=np.percentile(error, 95)),
        rasterized=True,
    )
    ax.plot([-1.2, 1.2], [-1.2, 1.2], "k--", linewidth=1, alpha=0.5, label="y = x")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Ideal Expectation Value (E_ideal)", fontsize=11)
    ax.set_ylabel("Mitigated Prediction (Ê_mitigated)", fontsize=11)
    ax.set_title("Mitigated vs Ideal Expectation Values", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("|Prediction Error|", fontsize=10)

    mae = np.mean(error)
    r2 = 1.0 - np.sum((ideal - pred) ** 2) / np.sum((ideal - ideal.mean()) ** 2)
    ax.text(0.02, 0.95, f"MAE = {mae:.4f}\nR² = {r2:.4f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = output_dir / "predictions_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_residual_histogram(
    y_res_true: np.ndarray,
    delta_hat: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot 3: Error residual distribution histogram."""
    prediction_error = delta_hat - y_res_true

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Error Residual Distributions", fontsize=13, fontweight="bold")

    # Left: true vs predicted residuals
    axes[0].hist(y_res_true, bins=100, alpha=0.6, label="Δ_true = E_ideal − E_noisy(λ=1)",
                 color="#2196F3", density=True)
    axes[0].hist(delta_hat, bins=100, alpha=0.6, label="Δ̂ (predicted)",
                 color="#FF5722", density=True)
    axes[0].set_xlabel("Residual Value")
    axes[0].set_ylabel("Density")
    axes[0].set_title("True vs Predicted Residuals")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # Right: prediction error (should be bell curve at 0)
    axes[1].hist(prediction_error, bins=100, alpha=0.7, color="#4CAF50", density=True)
    axes[1].axvline(0, color="red", linewidth=1, linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Prediction Error (Δ̂ − Δ_true)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Prediction Error Distribution")
    mean_err = np.mean(prediction_error)
    std_err  = np.std(prediction_error)
    axes[1].text(0.02, 0.95, f"μ = {mean_err:.5f}\nσ = {std_err:.5f}",
                 transform=axes[1].transAxes, fontsize=10,
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = output_dir / "residual_histogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_gamma_histogram(
    gamma: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot 4: Physical survival probability (γ) distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(gamma, bins=100, alpha=0.7, color="#9C27B0", density=True, edgecolor="white",
            linewidth=0.3)
    ax.set_xlabel("Analytical Survival Probability γ ∈ [0, 1]", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Physical Survival Probability Distribution (γ at λ=1)",
                 fontsize=13, fontweight="bold")

    mean_g = np.mean(gamma)
    median_g = np.median(gamma)
    ax.axvline(mean_g, color="red", linewidth=1.5, linestyle="--", label=f"Mean = {mean_g:.3f}")
    ax.axvline(median_g, color="orange", linewidth=1.5, linestyle=":", label=f"Median = {median_g:.3f}")

    # Annotate recoverable vs noise-floor fractions
    recoverable = np.mean(gamma > 0.1)
    ax.text(0.02, 0.95, f"Recoverable (γ > 0.1): {recoverable:.1%}\n"
            f"Noise floor (γ ≤ 0.1): {1 - recoverable:.1%}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = output_dir / "gamma_histogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_attention_heatmap(
    model: QEMFormerGNN,
    sample_batch: Data,
    device: torch.device,
    output_dir: Path,
    num_heads: int = 4,
) -> None:
    """Plot 5: Multi-head attention weight heatmap from TransformerConv."""
    alpha = extract_attention_weights(model, sample_batch, device, layer_idx=0)
    if alpha is None:
        print("    [Attention Heatmap] Could not extract attention weights, skipping.")
        return

    alpha_np = alpha.numpy()  # [num_edges, num_heads]
    n_edges = alpha_np.shape[0]
    actual_heads = alpha_np.shape[1] if alpha_np.ndim > 1 else 1

    if actual_heads == 1:
        alpha_np = alpha_np.reshape(-1, 1)

    # Limit display to first 50 edges for readability
    max_edges = min(50, n_edges)
    alpha_display = alpha_np[:max_edges, :min(actual_heads, num_heads)]

    fig, ax = plt.subplots(figsize=(8, max(6, max_edges * 0.15)))
    im = ax.imshow(alpha_display, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xlabel("Attention Head", fontsize=11)
    ax.set_ylabel("Edge Index (gate_src → gate_dst)", fontsize=11)
    ax.set_title("TransformerConv Multi-Head Attention Weights (Layer 0)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(range(alpha_display.shape[1]))
    ax.set_xticklabels([f"Head {i}" for i in range(alpha_display.shape[1])])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Attention Weight α_ij", fontsize=10)

    # Annotate entropy per head (measures specialization)
    for h in range(alpha_display.shape[1]):
        col = alpha_display[:, h]
        col_norm = col / (col.sum() + 1e-12)
        entropy = -np.sum(col_norm * np.log(col_norm + 1e-12))
        max_entropy = np.log(max_edges + 1e-12)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        ax.text(h, -1.5, f"H={normalized_entropy:.2f}",
                ha="center", fontsize=8, color="blue")

    plt.tight_layout()
    path = output_dir / "attention_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_error_vs_depth(
    error: np.ndarray,
    depth: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot 6: |error| box plot grouped by circuit depth."""
    abs_error = np.abs(error)
    unique_depths = sorted(np.unique(depth.astype(int)))

    fig, ax = plt.subplots(figsize=(12, 6))

    data_by_depth = [abs_error[depth.astype(int) == d] for d in unique_depths]
    bp = ax.boxplot(data_by_depth, positions=range(len(unique_depths)),
                    showfliers=False, patch_artist=True, widths=0.6)

    # Color boxes by median error
    medians = [np.median(d) for d in data_by_depth]
    max_med = max(medians) if medians else 1
    cmap = plt.cm.RdYlGn_r
    for patch, med in zip(bp["boxes"], medians):
        patch.set_facecolor(cmap(med / max(max_med, 1e-6)))
        patch.set_alpha(0.7)

    ax.set_xticks(range(len(unique_depths)))
    ax.set_xticklabels([str(d) for d in unique_depths])
    ax.set_xlabel("Circuit Depth", fontsize=11)
    ax.set_ylabel("|Prediction Error|", fontsize=11)
    ax.set_title("Prediction Error vs Circuit Depth", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    path = output_dir / "error_vs_depth.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_zne_trajectories(
    loader: PyGDataLoader,
    model: QEMFormerGNN,
    device: torch.device,
    output_dir: Path,
    n_samples: int = 6,
) -> None:
    """Plot 7: Sample ZNE curves with raw noisy, mitigated, and ideal values."""
    model.eval()

    samples = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            delta_hat = model(batch)

        # Iterate over individual graphs in the batch
        batch_size = int(batch.y.shape[0])
        for i in range(min(batch_size, n_samples - len(samples))):
            u_i = batch.u[i]  # [1, 9] or [9]
            if u_i.dim() == 2:
                u_i = u_i.squeeze(0)
            zne_vals = u_i[4:9].cpu().numpy()  # ZNE values at λ=1,3,5,7,9
            ideal_val = batch.y_ideal[i].cpu().item()
            baseline_val = batch.baseline[i].cpu().item()
            mitigated_val = (delta_hat[i].squeeze() + batch.baseline[i]).cpu().item()
            depth_val = u_i[0].cpu().item()

            samples.append({
                "zne": zne_vals,
                "ideal": ideal_val,
                "baseline": baseline_val,
                "mitigated": mitigated_val,
                "depth": depth_val,
            })
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break

    if not samples:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("ZNE Trajectory Samples — Raw Noisy → Mitigated vs Ideal",
                 fontsize=14, fontweight="bold")

    lam_values = SCALE_FACTORS

    for idx, (ax, s) in enumerate(zip(axes.flat, samples)):
        # Plot ZNE noisy curve
        ax.plot(lam_values, s["zne"], "o-", color="#2196F3", linewidth=1.5,
                markersize=6, label="ZNE Noisy", zorder=3)

        # Ideal value (horizontal line)
        ax.axhline(s["ideal"], color="#4CAF50", linewidth=1.5, linestyle="--",
                   label=f"Ideal = {s['ideal']:.3f}", zorder=2)

        # Mitigated prediction (star at λ=0)
        ax.plot(0, s["mitigated"], "*", color="#FF5722", markersize=14, zorder=5,
                label=f"Mitigated = {s['mitigated']:.3f}")

        # Baseline (circle at λ=1)
        ax.plot(1, s["baseline"], "D", color="#9C27B0", markersize=8, zorder=4,
                label=f"Baseline = {s['baseline']:.3f}")

        ax.set_xlabel("Scale Factor λ")
        ax.set_ylabel("Expectation Value")
        ax.set_title(f"Sample {idx + 1} (depth={int(s['depth'])})", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-0.5, 10)
        ax.set_ylim(-1.15, 1.15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "zne_trajectories.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ===========================================================================
# Plot 8: Feature Importance via Permutation
# ===========================================================================

@torch.no_grad()
def plot_feature_importance(
    model: "QEMFormerGNN",
    loader: PyGDataLoader,
    device: torch.device,
    output_dir: Path,
    use_amp: bool = False,
) -> None:
    """Plot 8: Feature Importance via Permutation.

    For each feature group, the group's values are randomly shuffled across
    the batch dimension (breaking the correlation with the label) and MAE is
    re-evaluated. Importance = DELTA_MAE = MAE_permuted − MAE_baseline.
    A larger DELTA_MAE means the model relied heavily on that group.

    Feature groups
    --------------
    Node features (data.x columns):
      - Gate type one-hot  cols 0–3
      - RZ angle theta     col  4
      - T1/T2 coherence    cols 5–6
      - Gate error rate    col  7
      - SPAM rates         cols 8–9
      - Target qubit params cols 10–13
      - Layer depth        col  14  (new, Fix 3)
    Edge features (data.edge_attr): all columns
    Global features (data.u):
      - Circuit metadata   cols 0–3
      - ZNE values         cols 4–8
    """
    model.eval()

    # ----------------------------------------------------------------
    # 1. Compute baseline MAE
    # ----------------------------------------------------------------
    abs_err_sum = 0.0
    n_samples   = 0
    all_batches: List[Data] = []

    for batch in CUDAPrefetcher(loader, device):
        delta_hat   = model(batch)
        e_mitigated = delta_hat.squeeze(-1) + batch.baseline
        e_ideal     = batch.y + batch.baseline
        abs_err_sum += (e_mitigated - e_ideal).abs().sum().item()
        n_samples   += int(batch.y.shape[0])
        all_batches.append(batch.cpu())  # store CPU copy for permutation

    if n_samples == 0:
        print("    [Feature Importance] No samples found, skipping.")
        return

    baseline_mae = abs_err_sum / n_samples
    print(f"    Baseline MAE = {baseline_mae:.5f}")

    # ----------------------------------------------------------------
    # 2. Define feature groups as (label, attribute, col_slice_or_None)
    #    col_slice=None means permute the entire attribute tensor
    # ----------------------------------------------------------------
    FEAT_GROUPS = [
        ("Gate type\n(one-hot)",    "x",      slice(0, 4)),
        ("RZ angle\nθ",             "x",      slice(4, 5)),
        ("T1/T2\ncoherence",        "x",      slice(5, 7)),
        ("Gate error\nrate",        "x",      slice(7, 8)),
        ("SPAM rates",              "x",      slice(8, 10)),
        ("Target qubit\nparams",    "x",      slice(10, 14)),
        ("Layer depth\n(pos. enc)", "x",      slice(14, 15)),
        ("Wire topology\n(edges)",  "edge_attr", None),
        ("Circuit\nmetadata",       "u",      slice(0, 4)),
        ("ZNE\nvalues",             "u",      slice(4, 9)),
    ]

    delta_maes: List[float] = []

    for label, attr_name, col_slice in FEAT_GROUPS:
        perm_abs_err = 0.0
        perm_n       = 0

        for batch in all_batches:
            # Deep copy the relevant tensor only
            batch_p = batch.clone()
            attr    = getattr(batch_p, attr_name).clone()

            # Generate a random row permutation for this batch
            n_rows = attr.shape[0]
            perm   = torch.randperm(n_rows)

            if col_slice is None:
                # Permute the entire tensor (edge features)
                setattr(batch_p, attr_name, attr[perm])
            else:
                # Permute only the selected columns, keep others intact
                attr[:, col_slice] = attr[perm][:, col_slice]
                setattr(batch_p, attr_name, attr)

            batch_p = batch_p.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                delta_hat   = model(batch_p)
            e_mitigated = delta_hat.squeeze(-1) + batch_p.baseline
            e_ideal     = batch_p.y + batch_p.baseline
            perm_abs_err += (e_mitigated - e_ideal).abs().sum().item()
            perm_n       += int(batch_p.y.shape[0])

        perm_mae    = perm_abs_err / max(perm_n, 1)
        delta_mae   = perm_mae - baseline_mae
        delta_maes.append(delta_mae)
        print(f"    {label.replace(chr(10), ' '):30s}  DELTA_MAE = {delta_mae:+.5f}")

    # ----------------------------------------------------------------
    # 3. Plot horizontal bar chart, sorted by importance
    # ----------------------------------------------------------------
    labels    = [g[0] for g in FEAT_GROUPS]
    order     = sorted(range(len(delta_maes)), key=lambda i: delta_maes[i], reverse=True)
    sorted_labels = [labels[i] for i in order]
    sorted_vals   = [delta_maes[i] for i in order]

    colours = [
        "#E53935" if v > 0.01 else ("#FB8C00" if v > 0.002 else "#43A047")
        for v in sorted_vals
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(sorted_labels)), sorted_vals, color=colours,
                   edgecolor="white", linewidth=0.5, height=0.65)

    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(r"$\Delta$MAE = MAE$_{\,\mathrm{permuted}}$ − MAE$_{\,\mathrm{baseline}}$",
                  fontsize=11)
    ax.set_title("Feature Importance via Permutation\n"
                 f"(baseline MAE = {baseline_mae:.4f})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)

    # Value annotations
    for bar, val in zip(bars, sorted_vals):
        ax.text(
            bar.get_width() + max(abs(v) for v in sorted_vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center", fontsize=8,
        )

    plt.tight_layout()
    path = output_dir / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ===========================================================================
# Main Training Script
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QEMFormer GNN — Dual-Branch Graph Transformer for Quantum Error Mitigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--batch",       type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--val-freq",    type=int,   default=5)
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--patience",    type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--lambda-bndry",type=float, default=DEFAULT_LAMBDA_BNDRY)
    parser.add_argument("--dropout",     type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight-decay",type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip",   type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--hidden-dim",  type=int,   default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-heads",   type=int,   default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-layers",  type=int,   default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--global-dim",  type=int,   default=DEFAULT_GLOBAL_DIM)
    parser.add_argument("--fusion-dim",  type=int,   default=DEFAULT_FUSION_DIM)
    parser.add_argument("--max-shards",  type=int,   default=DEFAULT_MAX_SHARDS)
    parser.add_argument("--shard-dir",   type=str,   default=str(SHARD_DIR))
    parser.add_argument("--output-dir",  type=str,   default=str(OUTPUT_DIR))
    # Auto-detect physical CPU cores for worker count.
    # Clamp between 4 and 8; stay within available cores.
    _cpu_auto = max(4, min(8, (os.cpu_count() or 4) // 2))
    parser.add_argument("--num-workers",    type=int, default=_cpu_auto)
    parser.add_argument("--prefetch-factor", type=int, default=8,
                        help="Batches queued per worker in host RAM (default 8). "
                             "Higher = more buffer against shard-boundary gaps.")
    args = parser.parse_args()

    # Windows cp1252 console can't encode Unicode math symbols
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device setup ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_scaler = torch.amp.GradScaler() if use_amp else None

    if device.type == "cuda":
        # Auto-tune cuDNN kernels for the fixed graph sizes once per layer shape.
        # Typically yields 5–20% throughput improvement on repeated workloads.
        torch.backends.cudnn.benchmark = True

    # ── Seed ─────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # persistent_workers keeps worker processes alive across epochs, eliminating
    # process teardown/spin-up latency and page-cache invalidation at each
    # epoch boundary.  Requires num_workers > 0.
    _nw             = args.num_workers
    _use_persist    = _nw > 0
    _prefetch       = args.prefetch_factor if _nw > 0 else None
    _pin            = device.type == "cuda"

    # ── Banner ───────────────────────────────────────────────────────────
    print("=" * 75)
    print("  QEMFormer GNN -- Dual-Branch Graph Transformer")
    print("  Architecture: TransformerConv (edge-conditioned) + Global MLP")
    print("  Target: Residual Delta = E_ideal - E_noisy(lam=1)")
    print("=" * 75)
    print(f"  Device            : {device}")
    print(f"  AMP               : {use_amp}")
    print(f"  Epochs            : {args.epochs}")
    print(f"  Batch size        : {args.batch}")
    print(f"  Learning rate     : {args.lr}")
    print(f"  Weight decay      : {args.weight_decay}")
    print(f"  Dropout           : {args.dropout}")
    print(f"  Grad clip         : {args.grad_clip}")
    print(f"  Hidden dim        : {args.hidden_dim}")
    print(f"  Attention heads   : {args.num_heads}")
    print(f"  Conv layers       : {args.num_layers}")
    print(f"  Global dim        : {args.global_dim}")
    print(f"  Fusion dim        : {args.fusion_dim}")
    print(f"  Patience          : {args.patience}")
    print(f"  Lam_boundary      : {args.lambda_bndry}")
    print(f"  Shard dir         : {args.shard_dir}")
    print(f"  Max shards        : {args.max_shards or 'all'}")
    print(f"  Output dir        : {output_dir}")
    print(f"  --- Data Pipeline ---")
    print(f"  Num workers       : {_nw}")
    print(f"  Persistent workers: {_use_persist}")
    print(f"  Prefetch factor   : {_prefetch}")
    print(f"  Pin memory        : {_pin}")
    print(f"  CUDA prefetcher   : {device.type == 'cuda'} (dual-stream async)")
    print("=" * 75)

    # ── Data Loading ─────────────────────────────────────────────────────
    print("\n  Loading datasets...")

    train_ds = SplitShardDataset(
        shard_dir=args.shard_dir, split="train",
        max_shards=args.max_shards, shuffle=True, seed=args.seed,
    )
    val_ds = SplitShardDataset(
        shard_dir=args.shard_dir, split="val",
        max_shards=args.max_shards, shuffle=False, seed=args.seed,
    )
    test_ds = SplitShardDataset(
        shard_dir=args.shard_dir, split="test",
        max_shards=args.max_shards, shuffle=False, seed=args.seed,
    )

    print(f"  Train : {train_ds}")
    print(f"  Val   : {val_ds}")
    print(f"  Test  : {test_ds}")

    # ── DataLoader construction ───────────────────────────────────────────
    # PyGDataLoader: uses the optimized PyG Batch collation inside workers.
    # persistent_workers=True:  worker processes survive across epoch boundaries.
    # prefetch_factor=8:        each worker pre-queues 8 batches in host RAM,
    #                           giving enough runway to hide shard-load latency.
    # pin_memory=True:          enables DMA from page-locked RAM to VRAM.
    # CUDAPrefetcher (applied in train/evaluate):  dual-stream async H2D.
    _loader_kwargs = dict(
        batch_size=args.batch,
        num_workers=_nw,
        pin_memory=_pin,
        persistent_workers=_use_persist,
        **(dict(prefetch_factor=_prefetch) if _prefetch is not None else {}),
    )
    train_loader = PyGDataLoader(train_ds, **_loader_kwargs)
    val_loader   = PyGDataLoader(val_ds,   **_loader_kwargs)
    test_loader  = PyGDataLoader(test_ds,  **_loader_kwargs)

    # ── Model ────────────────────────────────────────────────────────────
    model = QEMFormerGNN(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        global_feat_dim=GLOBAL_FEAT_DIM,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        global_dim=args.global_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = model.count_parameters()
    print(f"\n  Model parameters: {n_params:,}")
    print(f"  Model architecture:\n{model}\n")

    if hasattr(torch, "compile") and device.type == "cuda" and os.name != "nt":
        model = torch.compile(model)

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Estimate steps per epoch for OneCycleLR — use ceil + buffer to prevent
    # stepping past total_steps when IterableDataset yields an extra partial batch
    steps_per_epoch = max(math.ceil(len(train_ds) / args.batch), 1)
    total_steps = steps_per_epoch * args.epochs + 10  # +10 safety buffer
    scheduler_step_count = 0  # track actual steps taken

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    criterion = PhysicsInformedLoss(
        lambda_bndry=args.lambda_bndry,
        huber_delta=1.0,   # Fix 4: was 0.5 — wider gradient signal for recoverable errors
    )

    # ── Training Loop ────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_mae": [],  "val_mae": [],
        "train_rmse": [], "val_rmse": [],
        "train_r2": [],   "val_r2": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    # CSV log
    log_path = output_dir / "training_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=[
        "epoch", "train_loss", "train_huber", "train_bndry", "train_mae",
        "train_rmse", "train_r2", "val_loss", "val_huber", "val_bndry",
        "val_mae", "val_rmse", "val_r2", "lr", "time_s",
    ])
    log_writer.writeheader()

    print("-" * 105)
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | "
          f"{'Train MAE':>10} | {'Val MAE':>9} | {'Train R²':>8} | {'Val R²':>8} | "
          f"{'LR':>8} | Time")
    print("-" * 105)

    t_total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.perf_counter()

        # Update epoch for shard shuffling
        train_ds.set_epoch(epoch)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            amp_scaler=amp_scaler, scheduler=scheduler,
            grad_clip=args.grad_clip,
        )

        # Validate (skip if no val shards)
        has_val = len(val_ds) > 0
        if has_val:
            if epoch % args.val_freq == 0 or epoch == args.epochs:
                val_metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
            else:
                val_metrics = train_metrics.copy()
        else:
            val_metrics = train_metrics  # fallback: use train metrics for early stopping

        # Record history
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            history[f"{split}_loss"].append(metrics["loss"])
            history[f"{split}_mae"].append(metrics["mae"])
            history[f"{split}_rmse"].append(metrics["rmse"])
            history[f"{split}_r2"].append(metrics["r2"])

        epoch_time = time.perf_counter() - t_epoch
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = val_metrics["loss"] < best_val_loss
        star = " *" if is_best else ""

        # Console output: matching mlp_mitiq.py format
        print(f"{epoch:>6} | {train_metrics['loss']:>11.6f} | {val_metrics['loss']:>10.6f} | "
              f"{train_metrics['mae']:>10.6f} | {val_metrics['mae']:>9.6f} | {train_metrics['r2']:>8.4f} | {val_metrics['r2']:>8.4f} | "
              f"{current_lr:.2e} | {epoch_time:>5.1f}s{star}")

        # CSV log
        log_writer.writerow({
            "epoch": epoch,
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_huber": f"{train_metrics['huber']:.6f}",
            "train_bndry": f"{train_metrics['bndry']:.6f}",
            "train_mae": f"{train_metrics['mae']:.6f}",
            "train_rmse": f"{train_metrics['rmse']:.6f}",
            "train_r2": f"{train_metrics['r2']:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_huber": f"{val_metrics['huber']:.6f}",
            "val_bndry": f"{val_metrics['bndry']:.6f}",
            "val_mae": f"{val_metrics['mae']:.6f}",
            "val_rmse": f"{val_metrics['rmse']:.6f}",
            "val_r2": f"{val_metrics['r2']:.6f}",
            "lr": f"{current_lr:.2e}",
            "time_s": f"{epoch_time:.1f}",
        })
        log_file.flush()

        # Early stopping
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_gnn_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val loss {best_val_loss:.6f} at epoch {best_epoch})")
                break

    log_file.close()
    total_time = time.perf_counter() - t_total_start

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_gnn_model.pth")

    print("-" * 75)
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # ── Load best model for evaluation ───────────────────────────────────
    best_model_path = output_dir / "best_gnn_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # Use test set if available, otherwise fall back to train set for plots
    has_test = len(test_ds) > 0
    eval_loader = test_loader if has_test else train_loader
    eval_label  = "Test" if has_test else "Train (no test shards)"

    print("\n" + "=" * 75)
    print(f"  {eval_label} Set Evaluation (best model)")
    print("=" * 75)

    test_results = evaluate(
        model, eval_loader, criterion, device,
        use_amp=use_amp, collect_details=True,
    )

    print(f"  {eval_label} Loss : {test_results['loss']:.6f}")
    print(f"  {eval_label} MAE  : {test_results['mae']:.6f}")
    print(f"  {eval_label} RMSE : {test_results['rmse']:.6f}")
    print(f"  {eval_label} R2   : {test_results['r2']:.6f}")

    details = test_results["details"]

    # ── Diagnostic Plots ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  Generating Diagnostic Plots")
    print("=" * 75)

    # Plot 1: Training curves
    try:
        plot_training_curves(history, output_dir)
    except Exception as e:
        print(f"    [Error] Training curves: {e}")

    # Plot 2: Predicted vs Ideal scatter
    try:
        plot_predictions_scatter(details["pred"], details["ideal"], output_dir)
    except Exception as e:
        print(f"    [Error] Predictions scatter: {e}")

    # Plot 3: Error residual histogram
    try:
        plot_residual_histogram(details["y_res"], details["delta_hat"], output_dir)
    except Exception as e:
        print(f"    [Error] Residual histogram: {e}")

    # Plot 4: Survival probability histogram
    try:
        plot_gamma_histogram(details["gamma"], output_dir)
    except Exception as e:
        print(f"    [Error] Gamma histogram: {e}")

    # Plot 5: Attention weight heatmap — need a single-graph sample
    try:
        print("    Extracting attention weights...")
        sample_batch = None
        for batch in test_loader:
            sample_batch = batch
            break
        if sample_batch is not None:
            plot_attention_heatmap(model, sample_batch, device, output_dir,
                                  num_heads=args.num_heads)
    except Exception as e:
        print(f"    [Error] Attention heatmap: {e}")

    # Plot 6: Error vs circuit depth
    try:
        error = details["pred"] - details["ideal"]
        plot_error_vs_depth(error, details["depth"], output_dir)
    except Exception as e:
        print(f"    [Error] Error vs depth: {e}")

    # Plot 7: ZNE trajectory samples
    try:
        plot_zne_trajectories(test_loader, model, device, output_dir, n_samples=6)
    except Exception as e:
        print(f"    [Error] ZNE trajectories: {e}")

    # Plot 8: Feature importance via permutation
    try:
        print("    Computing feature importance (permutation method)...")
        plot_feature_importance(
            model, eval_loader, device, output_dir, use_amp=use_amp
        )
    except Exception as e:
        print(f"    [Error] Feature importance: {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  Summary")
    print("=" * 75)
    print(f"  Model parameters : {n_params:,}")
    print(f"  Total train time : {total_time:.1f}s")
    print(f"  Best epoch       : {best_epoch}")
    print(f"  Best val loss    : {best_val_loss:.6f}")
    print(f"  Test R2          : {test_results['r2']:.6f}")
    print(f"  Test MAE         : {test_results['mae']:.6f}")
    print(f"  Test RMSE        : {test_results['rmse']:.6f}")
    print(f"\n  Outputs saved to : {output_dir}")
    print(f"    best_gnn_model.pth")
    print(f"    final_gnn_model.pth")
    print(f"    training_log.csv")
    print(f"    training_curves.png")
    print(f"    predictions_scatter.png")
    print(f"    residual_histogram.png")
    print(f"    gamma_histogram.png")
    print(f"    attention_heatmap.png")
    print(f"    error_vs_depth.png")
    print(f"    zne_trajectories.png")
    print(f"    feature_importance.png")
    print("=" * 75)


if __name__ == "__main__":
    main()
