# Quantum Error Mitigation via Deep Learning & Topological Graph Transformers: QEMFormer & ZNE-GNN

> **Academic Pre-print & Research Status Notice**  
> **Notice:** This repository contains the reference codebase, experimental architectures, theoretical derivations, and evaluation benchmarks for an upcoming peer-reviewed research manuscript on learning-based Quantum Error Mitigation (QEM).  
> **Under Active Development:** The codebase, hyperparameter configurations, and model checkpoints are continuously updated. Ongoing investigations include scaling to 100+ qubit topologies, multi-basis observable groupings, and comparative studies against Probabilistic Error Cancellation (PEC).  
> **Dataset Availability:** The full sharded training dataset (comprising 1,000,000+ synthetic and hardware-calibrated quantum circuit DAGs, 20 compressed shards, multi-scale unitary folding trajectories, and complete open-system noise parameter logs) will be released publicly via a persistent Google Drive link upon public publication of the manuscript.

---

## Table of Contents
1. [Executive Summary & Scientific Context](#1-executive-summary--scientific-context)
2. [Theoretical Physics of Quantum Noise & Error Mitigation](#2-theoretical-physics-of-quantum-noise--error-mitigation)
   - [2.1 Open Quantum Systems & Lindblad Master Equation](#21-open-quantum-systems--lindblad-master-equation)
   - [2.2 Microscopic Noise Channels & Kraus Operator Representations](#22-microscopic-noise-channels--kraus-operator-representations)
   - [2.3 Classical Zero-Noise Extrapolation (ZNE) Formalism](#23-classical-zero-noise-extrapolation-zne-formalism)
   - [2.4 Analytical Proof: Failure & Conditioning of High-Order Extrapolations](#24-analytical-proof-failure--conditioning-of-high-order-extrapolations)
   - [2.5 The Residual Target Formulation](#25-the-residual-target-formulation)
3. [Comprehensive Feature Schema & Quantum Physical Encodings](#3-comprehensive-feature-schema--quantum-physical-encodings)
   - [3.1 Tabular Vector Dataset Schema (31 Features)](#31-tabular-vector-dataset-schema-31-features)
   - [3.2 Gate-DAG Node Feature Schema (15 Features)](#32-gate-dag-node-feature-schema-15-features)
   - [3.3 Directed Wire-Edge Feature Schema (4 Features)](#33-directed-wire-edge-feature-schema-4-features)
   - [3.4 Global Graph Context Vector (9 Features)](#34-global-graph-context-vector-9-features)
4. [Architecture Specification: QEMFormer (Tabular-Sequence Fusion)](#4-architecture-specification-qemformer-tabular-sequence-fusion)
   - [4.1 Motivation & Context Learning Baseline](#41-motivation--context-learning-baseline)
   - [4.2 Static Hardware / Circuit Feature Branch](#42-static-hardware--circuit-feature-branch)
   - [4.3 Dynamic ZNE Trajectory Transformer Branch](#43-dynamic-zne-trajectory-transformer-branch)
   - [4.4 Cross-Modal Fusion & Physics-Informed Boundary Loss](#44-cross-modal-fusion--physics-informed-boundary-loss)
   - [4.5 Layer-by-Layer Tensor Shape Flow Trace](#45-layer-by-layer-tensor-shape-flow-trace)
5. [Architecture Specification: QEMFormer-GNN (Topological Graph Transformer)](#5-architecture-specification-qemformer-gnn-topological-graph-transformer)
   - [5.1 Quantum Circuit DAG Compilation](#51-quantum-circuit-dag-compilation)
   - [5.2 Branch A: Edge-Conditioned Multi-Head `TransformerConv`](#52-branch-a-edge-conditioned-multi-head-transformerconv)
   - [5.3 Physical Semantic Specialization of Attention Heads](#53-physical-semantic-specialization-of-attention-heads)
   - [5.4 Observable-Centric Readout Mechanism](#54-observable-centric-readout-mechanism)
   - [5.5 Branch B: Global Context MLP with ZNE Feature Dropout](#55-branch-b-global-context-mlp-with-zne-feature-dropout)
   - [5.6 Physics-Informed Huber Boundary Loss](#56-physics-informed-huber-boundary-loss)
   - [5.7 Complete Graph Batch Tensor Shape Flow Trace](#57-complete-graph-batch-tensor-shape-flow-trace)
6. [Data Engineering & Sharded Ingestion Pipeline](#6-data-engineering--sharded-ingestion-pipeline)
   - [6.1 The 20-Shard DAG Architecture (`ShardedDAGIterableDataset`)](#61-the-20-shard-dag-architecture-shardeddagiterabledataset)
   - [6.2 Survival Factor Pruning & Decoherence Cutoff](#62-survival-factor-pruning--decoherence-cutoff)
   - [6.3 Train / Validation / Test Cross-Validation Partitions](#63-train--validation--test-cross-validation-partitions)
7. [Experimental Results & Rigorous Comparative Benchmarking](#7-experimental-results--rigorous-comparative-benchmarking)
   - [7.1 Evaluated Baselines & Protocols](#71-evaluated-baselines--protocols)
   - [7.2 Quantitative Performance Matrix](#72-quantitative-performance-matrix)
   - [7.3 Detailed Statistical Error Distribution Analysis](#73-detailed-statistical-error-distribution-analysis)
   - [7.4 Crosstalk Resilience & Depth Scaling Analysis](#74-crosstalk-resilience--depth-scaling-analysis)
8. [Comprehensive Diagnostic Visualizations & Empirical Analysis](#8-comprehensive-diagnostic-visualizations--empirical-analysis)
   - [8.1 QEMFormer Diagnostic Visualizations (`qemformer/outputs/`)](#81-qemformer-diagnostic-visualizations-qemformeroutputs)
   - [8.2 QEMFormer-GNN Diagnostic Visualizations (`zne/mitigation_ai/outputs/gnn/`)](#82-qemformer-gnn-diagnostic-visualizations-znemitigation_aioutputsgnn)
9. [Algorithmic Procedures & Pseudocode](#9-algorithmic-procedures--pseudocode)
   - [9.1 Digital Unitary Noise Scaling Procedure](#91-digital-unitary-noise-scaling-procedure)
   - [9.2 QEMFormer-GNN Dual-Branch Execution Loop](#92-qemformer-gnn-dual-branch-execution-loop)
10. [Repository File Tree](#10-repository-file-tree)
11. [Installation, Environment & Execution Guide](#11-installation-environment--execution-guide)
    - [11.1 Dependencies & Hardware Requirements](#111-dependencies--hardware-requirements)
    - [11.2 Training the QEMFormer Model](#112-training-the-qemformer-model)
    - [11.3 Training the QEMFormer-GNN Model](#113-training-the-qemformer-gnn-model)
    - [11.4 Automated Shard Evaluation Suite](#114-automated-shard-evaluation-suite)

---

## 1. Executive Summary & Scientific Context

Noisy Intermediate-Scale Quantum (NISQ) systems operate in an uncorrected regime where hardware error rates ($\epsilon_{\text{1q}} \sim 10^{-4}-10^{-3}$, $\epsilon_{\text{2q}} \sim 10^{-3}-10^{-2}$) severely restrict circuit depth $D$ and width $n$. Without fault-tolerant Quantum Error Correction (QEC)—which requires physical-to-logical qubit overheads exceeding $10^3:1$—computational expectation values $\langle \hat{O} \rangle = \mathrm{Tr}(\hat{O} \rho)$ rapidly decay toward maximally mixed states $\rho \to \frac{\mathbb{I}}{2^n}$, driving signal-to-noise ratios to zero.

**Quantum Error Mitigation (QEM)** refers to a family of post-processing and active sampling techniques designed to estimate error-free expectation values without auxiliary logical qubits. The foundational baseline, **Zero-Noise Extrapolation (ZNE)**, artificially amplifies physical noise by scaling factors $\lambda \in \{1, 3, 5, \dots\}$ and extrapolates backward to the zero-noise limit $\lambda \to 0$.

However, traditional classical extrapolation (linear, polynomial, Richardson, exponential) suffers from fatal mathematical limitations:
1. **Analytic Model Mismatch:** Classical curve fits assume stationary, time-invariant, Markovian depolarizing noise. Real physical processors exhibit non-Markovian memory effects, frequency drifts, two-level system (TLS) fluctuators, and coherent unitary over-rotations.
2. **Ill-Conditioned Vandermonde Inversion:** Higher-order polynomial and Richardson extrapolations rely on polynomial weighting matrices whose condition numbers grow exponentially with the number of scale factors, causing finite sampling shot noise to blow up into unphysical expectation values ($|\langle \hat{O} \rangle| > 1$).
3. **Topological Agnosticism:** Classical extrapolation completely discards the spatial and temporal topology of the quantum circuit DAG, treating the entire process as a black-box scalar decay.

### The Learning-Based Paradigm
This repository provides an end-to-end framework implementing two novel learning-based quantum error mitigation paradigms:
- **`QEMFormer`**: A multi-modal transformer baseline that couples static device calibration physics with sequence-modeled dynamic expectation decay trajectories.
- **`QEMFormer-GNN`**: A topological gate-DAG graph transformer that parses the full causal DAG of the quantum circuit, applies edge-conditioned attention over physical coupling errors, and isolates terminal observable resonators via an observable-centric readout mechanism.

---

## 2. Theoretical Physics of Quantum Noise & Error Mitigation

### 2.1 Open Quantum Systems & Lindblad Master Equation

The continuous-time evolution of an open quantum system coupled to a Markovian thermal reservoir is governed by the Lindblad master equation:

$$\frac{d\rho(t)}{dt} = -i \left[ \mathcal{H}_0 + \mathcal{H}_{\text{ctrl}}(t) + \mathcal{H}_{\text{cross}}, \rho(t) \right] + \sum_{k} \gamma_k \left( L_k \rho(t) L_k^\dagger - \frac{1}{2} \left\{ L_k^\dagger L_k, \rho(t) \right\} \right)$$

where:
- $\mathcal{H}_0 = \sum_{j} \frac{\omega_j}{2} Z_j$ is the bare qubit Hamiltonian.
- $\mathcal{H}_{\text{ctrl}}(t) = \sum_{j} \left( \Omega_{x,j}(t) X_j + \Omega_{y,j}(t) Y_j \right)$ represents the microwave control drive fields.
- $\mathcal{H}_{\text{cross}} = \sum_{j < k} \zeta_{ZZ}^{(j, k)} Z_j \otimes Z_k$ represents parasitic static $ZZ$ coupling between physically adjacent transmons.
- $L_k$ are the Lindblad jump operators representing interaction with dissipative environmental modes, and $\gamma_k$ are the corresponding relaxation/dephasing decay rates.

---

### 2.2 Microscopic Noise Channels & Kraus Operator Representations

Integrating the continuous-time dynamics over gate duration $\tau_g$ yields a discrete Completely Positive Trace-Preserving (CPTP) quantum map $\mathcal{E}(\rho)$:

$$\rho \mapsto \mathcal{E}(\rho) = \sum_{k} K_k \rho K_k^\dagger, \quad \text{subject to} \quad \sum_k K_k^\dagger K_k = \mathbb{I}$$

The dataset simulated and mitigated in this project incorporates seven simultaneous physical noise channels:

#### 1. Longitudinal Energy Relaxation (Amplitude Damping)
Accounts for spontaneous emission into the dielectric substrate with characteristic relaxation time $T_1$:
$$\gamma_1 = 1 - \exp\left(-\frac{\tau_g}{T_1}\right)$$
$$K_0^{(\text{amp})} = \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma_1} \end{bmatrix}, \quad K_1^{(\text{amp})} = \begin{bmatrix} 0 & \sqrt{\gamma_1} \\ 0 & 0 \end{bmatrix}$$

#### 2. Pure Transverse Dephasing (Phase Damping)
Accounts for low-frequency flux bias noise and photon number fluctuations in the readout resonator with pure dephasing time $T_\phi$ (where $\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}$):
$$\gamma_\phi = 1 - \exp\left(-\frac{\tau_g}{T_\phi}\right)$$
$$K_0^{(\text{deph})} = \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma_\phi} \end{bmatrix}, \quad K_1^{(\text{deph})} = \begin{bmatrix} 0 & 0 \\ 0 & \sqrt{\gamma_\phi} \end{bmatrix}$$

#### 3. Computational Subspace Leakage ($d=3$ Qutrit State Space)
Accounts for transmon weak anharmonicity $\alpha = \omega_{12} - \omega_{01} \approx -300\text{ MHz}$, where strong microwave drives excite the state $|1\rangle \to |2\rangle$:
$$K_0^{(\text{leak})} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \sqrt{1 - p_{\text{leak}}} & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad K_1^{(\text{leak})} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & \sqrt{p_{\text{leak}}} & 0 \end{bmatrix}$$

#### 4. Parasitic Cross-Resonance & Static $ZZ$ Crosstalk
Non-local unitary phase accumulation between spectator and active qubits:
$$U_{ZZ}(\theta_{ZZ}) = \exp\left( -i \frac{\zeta_{ZZ} \tau_g}{2} Z_j \otimes Z_k \right)$$

#### 5. Systematic Coherent Control Errors
Systematic microwave over/under-rotation parameterized by angle defect $\delta \theta$:
$$R_n(\theta) \to R_n(\theta + \delta \theta) = \exp\left( -i \frac{\theta + \delta \theta}{2} (\mathbf{n} \cdot \boldsymbol{\sigma}) \right)$$

#### 6. Two-Level System (TLS) Defect Drops
Stochastic spectral diffusion causing abrupt, localized drops in $T_1$:
$$T_1^{\text{eff}} = T_1 \cdot \left(1 - \Delta_{\text{TLS}} \cdot \mathbf{1}_{\{\text{frequency collision}\}}\right)$$

#### 7. State Preparation and Measurement (SPAM) Readout Inversion
Classical stochastic confusion matrix $\mathbf{\Lambda}$ acting on projective computational measurement probabilities:
$$\begin{bmatrix} P(0)_{\text{meas}} \\ P(1)_{\text{meas}} \end{bmatrix} = \begin{bmatrix} 1 - p_{01} & p_{10} \\ p_{01} & 1 - p_{10} \end{bmatrix} \begin{bmatrix} P(0)_{\text{true}} \\ P(1)_{\text{true}} \end{bmatrix}$$

---

### 2.3 Classical Zero-Noise Extrapolation (ZNE) Formalism

Digital Zero-Noise Extrapolation scales circuit noise by replacing each gate $U$ with odd-length self-inverting sequences using unitary folding:

$$U \;\longrightarrow\; U \left( U^\dagger U \right)^n, \quad n \in \mathbb{N}_0$$

Because $U^\dagger U = \mathbb{I}$, the ideal mathematical unitary is preserved: $U (\mathbb{I})^n = U$. However, the physical circuit depth and exposure to the environment scales by the scale factor:

$$\lambda = 1 + 2n \quad \implies \quad \lambda \in \{1, 3, 5, 7, 9\}$$

For a multi-qubit circuit with $G$ gates, partial or fractional unitary folding selects a subset $k$ of gates to invert such that:

$$\lambda = 1 + 2 \frac{k}{G}$$

Expectation values of observable $\hat{O}$ are measured at discrete scale factors:

$$\mathcal{D}_{\text{measured}} = \left\{ \left(\lambda_m, E(\lambda_m)\right) \;\middle|\; \lambda_m \in \{1, 3, 5, 7, 9\} \right\}$$

Classical extrapolators fit a parametric function $f(\lambda; \boldsymbol{\theta})$ to $\mathcal{D}_{\text{measured}}$ and extrapolate to the zero-noise limit $E^*(0) = f(0; \boldsymbol{\theta})$:
1. **Linear:** $f(\lambda) = c_0 + c_1 \lambda \implies E^*(0) = c_0$
2. **Polynomial ($d$-degree):** $f(\lambda) = \sum_{k=0}^d c_k \lambda^k \implies E^*(0) = c_0$
3. **Exponential:** $f(\lambda) = A + B e^{-\gamma \lambda} \implies E^*(0) = A + B$
4. **Richardson:** Eliminates polynomial error orders up to degree $M-1$ using fixed linear combinations:
   $$E^*(0) = \sum_{j=1}^M \gamma_j E(\lambda_j), \quad \gamma_j = \prod_{k \ne j} \frac{\lambda_k}{\lambda_k - \lambda_j}$$

---

### 2.4 Analytical Proof: Failure & Conditioning of High-Order Extrapolations

Let the noisy expectation value possess a perturbative Taylor series expansion in noise strength $\lambda$:

$$E(\lambda) = E_{\text{ideal}} + \sum_{k=1}^{M-1} a_k \lambda^k + \mathcal{O}(\lambda^M) + \epsilon_{\text{shot}}(\lambda)$$

where $\epsilon_{\text{shot}}(\lambda) \sim \mathcal{N}\left(0, \frac{\sigma^2}{N_{\text{shots}}}\right)$ is finite sampling shot noise.

Richardson extrapolation seeks coefficients $\gamma = [\gamma_1, \dots, \gamma_M]^\top$ satisfying the linear system:

$$\begin{bmatrix} 1 & 1 & \cdots & 1 \\ \lambda_1 & \lambda_2 & \cdots & \lambda_M \\ \lambda_1^2 & \lambda_2^2 & \cdots & \lambda_M^2 \\ \vdots & \vdots & \ddots & \vdots \\ \lambda_1^{M-1} & \lambda_2^{M-1} & \cdots & \lambda_M^{M-1} \end{bmatrix} \begin{bmatrix} \gamma_1 \\ \gamma_2 \\ \gamma_3 \\ \vdots \\ \gamma_M \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix} \quad \iff \quad \mathbf{V}^\top \boldsymbol{\gamma} = \mathbf{e}_1$$

where $\mathbf{V}$ is the classical Vandermonde matrix.

#### Conditioning Collapse Proof:
The condition number $\kappa(\mathbf{V}) = \|\mathbf{V}\|_2 \|\mathbf{V}^{-1}\|_2$ for scale factors $\boldsymbol{\lambda} = [1, 3, 5, 7, 9]^\top$ grows exponentially:

$$\kappa(\mathbf{V}) \sim \mathcal{O}\left(c^M\right), \quad c > 1$$

Specifically, evaluating the analytic coefficients for $\boldsymbol{\lambda} = [1, 3, 5, 7, 9]$:
$$\gamma_1 = \frac{3 \times 5 \times 7 \times 9}{(1-3)(1-5)(1-7)(1-9)} = \frac{945}{(-2)(-4)(-6)(-8)} = \frac{945}{384} \approx +2.4609$$
$$\gamma_2 = \frac{1 \times 5 \times 7 \times 9}{(3-1)(3-5)(3-7)(3-9)} = \frac{315}{(2)(-2)(-4)(-6)} = \frac{315}{-96} \approx -3.2813$$
$$\gamma_3 = \frac{1 \times 3 \times 7 \times 9}{(5-1)(5-3)(5-7)(5-9)} = \frac{189}{(4)(2)(-2)(-4)} = \frac{189}{64} \approx +2.9531$$
$$\gamma_4 = \frac{1 \times 3 \times 5 \times 9}{(7-1)(7-3)(7-5)(7-9)} = \frac{135}{(6)(4)(2)(-2)} = \frac{135}{-96} \approx -1.4063$$
$$\gamma_5 = \frac{1 \times 3 \times 5 \times 7}{(9-1)(9-3)(9-5)(9-7)} = \frac{105}{(8)(6)(4)(2)} = \frac{105}{384} \approx +0.2734$$

Notice that the sum of absolute weights is large:

$$\sum_{j=1}^5 |\gamma_j| = 2.4609 + 3.2813 + 2.9531 + 1.4063 + 0.2734 = 10.375$$

Consequently, the variance of the extrapolated zero-noise estimator explodes:

$$\mathrm{Var}\left(E^*(0)\right) = \sum_{j=1}^5 \gamma_j^2 \mathrm{Var}\left(E(\lambda_j)\right) \approx \left(\sum_{j=1}^5 \gamma_j^2\right) \frac{\sigma^2}{N_{\text{shots}}} = (28.42) \frac{\sigma^2}{N_{\text{shots}}}$$

The estimator variance is amplified by a factor of **28.42**, driving the extrapolated expectation value outside the physical Pauli spectrum $[-1, 1]$ in up to 8.92% of realistic circuits.

---

### 2.5 The Residual Target Formulation

Directly training a deep network to map hardware parameters to $E_{\text{ideal}}$ produces an empirical pathology: because $E(\lambda=1)$ is strongly correlated with $E_{\text{ideal}}$, standard gradient descent rapidly settles into a shallow local minimum that outputs $\hat{E} \approx E(\lambda=1)$. The network effectively learns an identity shortcut, ignoring circuit depth, crosstalk parameters, and topological graph structure.

To force the model to learn the true physical distortion map, we reformulate the training target as the **quantum defect residual**:

$$\Delta = E_{\text{ideal}} - E_{\text{noisy}}(\lambda = 1)$$

At inference time, the mitigated expectation value is recovered deterministically via:

$$E_{\text{mitigated}} = \hat{\Delta} + E_{\text{noisy}}(\lambda = 1)$$

#### Invariant Properties of the Residual Formulation:
1. **Zero-Error Identity:** In the hypothetical limit of an ideal processor ($\epsilon \to 0$), $\Delta \to 0$. The model's nominal target converges naturally to 0 rather than tracking an arbitrary state-dependent expectation value.
2. **Gradient Preservation:** The loss gradients $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ are driven entirely by the error discrepancy $(\hat{\Delta} - \Delta_{\text{true}})$, preventing the dynamic range of $E_{\text{ideal}}$ from overwhelming subtle noise-induced variations.
3. **Bounded Defect Range:** Since $E_{\text{ideal}}, E_{\text{noisy}} \in [-1, 1]$, the ground-truth residual is bounded: $\Delta \in [-2, 2]$, with the vast majority of physical NISQ circuits concentrated in $\Delta \in [-0.6, 0.6]$.

---

## 3. Comprehensive Feature Schema & Quantum Physical Encodings

### 3.1 Tabular Vector Dataset Schema (31 Features)

*Dataset File:* [`zne/dataset/vector_dataset.npz`](file:///d:/Code/Quantum%20Error%20Mitigation/zne/dataset/vector_dataset.npz)  
Contains $500,000$ simulated instances stored as single-precision floating-point arrays `X: (500000, 31)` and targets `y: (500000,)`.

| Index | Feature Name | Physical Notation | SI / Scaled Unit | Physical Interpretation & Noise Channel |
| :---: | :--- | :---: | :---: | :--- |
| `0` | `num_qubits` | $n$ | Integer count | Number of active computational qubits in the register ($n \in [2, 10]$) |
| `1` | `depth` | $D$ | Integer layers | Critical path circuit layer depth after DAG scheduling ($D \in [5, 120]$) |
| `2` | `idle_frac` | $f_{\text{idle}}$ | Dimensionless $[0, 1]$ | Ratio of total circuit idle time to total qubit execution duration |
| `3` | `obs_weight` | $w(\hat{O})$ | Pauli weight | Number of non-identity terms in measured Pauli tensor product string |
| `4` | `sample_shots` | $N_s$ | Integer count | Monte Carlo projective measurement shots ($N_s \in [1024, 8192]$) |
| `5` | `error_1q` | $\epsilon_{1q}$ | Error rate | Average physical single-qubit Clifford gate infidelities ($10^{-4}-10^{-3}$) |
| `6` | `error_2q` | $\epsilon_{2q}$ | Error rate | Average physical two-qubit entangling (CX/ECR) infidelities ($10^{-3}-10^{-2}$) |
| `7` | `T1_us` | $T_1$ | Microseconds ($\mu\text{s}$) | Mean longitudinal thermal relaxation lifetime across the register |
| `8` | `T2_us` | $T_2$ | Microseconds ($\mu\text{s}$) | Mean transverse spin-spin dephasing lifetime across the register |
| `9` | `time_1q_ns` | $\tau_{1q}$ | Nanoseconds (ns) | Duration of physical single-qubit microwave control pulses ($\approx 20-35\text{ ns}$) |
| `10` | `time_2q_ns` | $\tau_{2q}$ | Nanoseconds (ns) | Duration of physical cross-resonance / flux entangling gates ($\approx 200-400\text{ ns}$) |
| `11` | `p_damp_1q` | $\gamma_{1, 1q}$ | Probability | Single-qubit integrated amplitude damping probability: $1 - e^{-\tau_{1q}/T_1}$ |
| `12` | `p_deph_1q` | $\gamma_{\phi, 1q}$ | Probability | Single-qubit integrated pure dephasing probability: $1 - e^{-\tau_{1q}/T_\phi}$ |
| `13` | `p_damp_2q` | $\gamma_{1, 2q}$ | Probability | Two-qubit integrated amplitude damping probability: $1 - e^{-\tau_{2q}/T_1}$ |
| `14` | `p_deph_2q` | $\gamma_{\phi, 2q}$ | Probability | Two-qubit integrated pure dephasing probability: $1 - e^{-\tau_{2q}/T_\phi}$ |
| `15` | `p_leak` | $p_{\text{leak}}$ | Probability | Population excitation probability outside computational subspace $\{|0\rangle, |1\rangle\} \to |2\rangle$ |
| `16` | `p_01` | $p_{01}$ | Probability | Classical SPAM readout confusion: probability of measuring $|1\rangle$ given $|0\rangle$ |
| `17` | `p_10` | $p_{10}$ | Probability | Classical SPAM readout confusion: probability of measuring $|0\rangle$ given $|1\rangle$ |
| `18` | `p_xtalk` | $p_{\text{xtalk}}$ | Probability | Multi-qubit classical drive crosstalk probability to nearest-neighbor channels |
| `19` | `beta_nm` | $\beta_{\text{nm}}$ | Index $[0, 1]$ | Degree of Non-Markovian environmental memory retention |
| `20` | `P_th` | $P_{\text{th}}$ | Probability | Equilibrium thermal excited-state population: $P_{\text{th}} = \left(1 + e^{\hbar \omega / k_B T}\right)^{-1}$ |
| `21` | `zeta_zz` | $\zeta_{ZZ}$ | Kilohertz (kHz) | Residual parasitic static $ZZ$ coupling rate between coupled transmons |
| `22` | `tls_drop` | $\delta_{\text{TLS}}$ | Ratio $[0, 1]$ | Relative fractional drop in $T_1$ caused by near-resonant two-level defect fluctuators |
| `23` | `theta_error` | $\delta \theta$ | Radians (rad) | Coherent systematic calibration offset in pulse rotation angles |
| `24` | `observable_align` | $\alpha_{\text{align}}$ | $[0, 1]$ | Geometric overlap metric between circuit state Bloch vector and Pauli observable |
| `25` | `survival_factor` | $\gamma$ | Ratio $[0, 1]$ | Total circuit coherence survival metric: $\exp\left(-\sum \epsilon_l\right)$ |
| `26` | `exp_val_scale_1` | $E(\lambda=1)$ | $[-1, 1]$ | Unmitigated noisy expectation value under baseline hardware noise ($\lambda=1$) |
| `27` | `exp_val_scale_3` | $E(\lambda=3)$ | $[-1, 1]$ | Noisy expectation value under digital unitary folding factor $\lambda=3$ |
| `28` | `exp_val_scale_5` | $E(\lambda=5)$ | $[-1, 1]$ | Noisy expectation value under digital unitary folding factor $\lambda=5$ |
| `29` | `exp_val_scale_7` | $E(\lambda=7)$ | $[-1, 1]$ | Noisy expectation value under digital unitary folding factor $\lambda=7$ |
| `30` | `exp_val_scale_9` | $E(\lambda=9)$ | $[-1, 1]$ | Noisy expectation value under digital unitary folding factor $\lambda=9$ |

---

### 3.2 Gate-DAG Node Feature Schema (15 Features)

*Target Module:* [`zne/dataset/dag_data.py`](file:///d:/Code/Quantum%20Error%20Mitigation/zne/dataset/dag_data.py)  
Each node $i \in \mathcal{V}$ in the circuit graph corresponds to an individual quantum gate or measurement operation. Its feature vector $\mathbf{x}_i \in \mathbb{R}^{15}$ is structured as:

$$\mathbf{x}_i = \left[ \mathbf{g}_i \;\middle\|\; T_{1, i} \;\middle\|\; T_{2, i} \;\middle\|\; \epsilon_{\text{gate}, i} \;\middle\|\; p_{01, i} \;\middle\|\; p_{10, i} \;\middle\|\; \tau_i \right]$$

- **Gate Type Categorical Encoding ($\mathbf{g}_i \in \{0, 1\}^9$, Cols 0–8):**  
  One-hot vector identifying the quantum operation:  
  `[0: Identity, 1: RZ, 2: SX, 3: X, 4: CX, 5: CZ, 6: ECR, 7: Barrier, 8: Measure]`.
- **Physical Relaxation Time ($T_{1, i}$, Col 9):**  
  Normalized longitudinal relaxation time of the qubit(s) hosting gate $i$: $T_1 / 100\,\mu\text{s}$.
- **Physical Dephasing Time ($T_{2, i}$, Col 10):**  
  Normalized transverse dephasing time of the qubit(s) hosting gate $i$: $T_2 / 100\,\mu\text{s}$.
- **Local Gate Inaccuracy ($\epsilon_{\text{gate}, i}$, Col 11):**  
  Physical benchmark error rate of operation $i$ from latest randomized benchmarking calibration.
- **Qubit SPAM Probabilities ($p_{01, i}, p_{10, i}$, Cols 12–13):**  
  Readout confusion probabilities for the wire(s) hosting gate $i$.
- **Temporal Normalized Layer Depth ($\tau_i$, Col 14):**  
  Normalized critical path position: $\tau_i = \frac{\text{layer}(i)}{\text{total circuit depth}} \in [0.0, 1.0]$.  
  *Critical Function:* Breaks permutation symmetry along the circuit timeline, preventing attention collapse.

---

### 3.3 Directed Wire-Edge Feature Schema (4 Features)

Each directed edge $e_{ij} = (i \to j) \in \mathcal{E}$ represents the causal evolution of quantum information along a physical qubit wire from operation $i$ to operation $j$. The edge attribute vector $\mathbf{e}_{ij} \in \mathbb{R}^4$ comprises:

1. **Physical Qubit Wire Index ($q_{\text{wire}} \in [0, n-1]$, Col 0):**  
   Integer identifying the specific hardware transmon wire transmitting the quantum state.
2. **Layer Step Gap ($\Delta l \in \mathbb{N}^+$, Col 1):**  
   Discrete scheduled layer difference: $\Delta l = \text{layer}(j) - \text{layer}(i)$. Quantifies non-local spatial skips.
3. **Idle Latency on Wire ($\tau_{\text{idle}} \ge 0$, Col 2):**  
   Total duration in nanoseconds where the qubit wire was un-driven between gate $i$ and gate $j$, during which the qubit suffered unmitigated $T_1/T_2$ free induction decay.
4. **Two-Qubit Directionality Flag ($d_{\text{role}} \in \{-1, 0, +1\}$, Col 3):**  
   Encodes the causal role across entangling interactions:
   - $+1$: Edge tracks the **Control** qubit wire through an entangling gate.
   - $-1$: Edge tracks the **Target** qubit wire through an entangling gate.
   - $0$: Edge represents a single-qubit gate state transition.

---

### 3.4 Global Graph Context Vector (9 Features)

Stored in `data.u \in \mathbb{R}^9` for each circuit graph, capturing macroscopic device state:
1. `depth`: Total scheduled circuit layer depth $D$.
2. `beta_nm`: Non-Markovian environment memory index $\beta_{\text{nm}} \in [0, 1]$.
3. `idle_frac`: Register-wide idle cycle ratio $f_{\text{idle}}$.
4. `log_shots`: Natural logarithm of projective shot budget: $\ln(N_s)$.
5. `exp_val_scale_1`: Raw noisy expectation value $E(\lambda=1)$.
6. `exp_val_scale_3`: Unitary folded expectation value $E(\lambda=3)$.
7. `exp_val_scale_5`: Unitary folded expectation value $E(\lambda=5)$.
8. `exp_val_scale_7`: Unitary folded expectation value $E(\lambda=7)$.
9. `exp_val_scale_9`: Unitary folded expectation value $E(\lambda=9)$.

---

## 4. Architecture Specification: QEMFormer (Tabular-Sequence Fusion)

*Target Implementation:* [`qemformer/qemformer_model.py`](file:///d:/Code/Quantum%20Error%20Mitigation/qemformer/qemformer_model.py)

```
========================================================================================
                              QEMFormer ARCHITECTURAL PIPELINE
========================================================================================

    [ Static Features: 21 dims ]                     [ ZNE Sequence: 5 values ]
                  │                                               │
                  ▼                                               ▼
         Linear(21 → 512)                                 Linear(1 → 128)
                  │                                               │
                  ▼                                               ▼
      ┌──────────────────────┐                       + Learnable Pos Embedding
      │ ResBlock(512 → 512)  │                                    │
      │   LayerNorm → GELU   │                                    ▼
      │   Linear(512 → 512)  │                       ┌─────────────────────────┐
      │   LayerNorm → GELU   │                       │ 4× Transformer Encoder  │
      │   Linear(512 → 512)  │                       │   Layers (Pre-LN):      │
      │   + Skip Connection  │                       │   d_model = 128         │
      └───────────┬──────────┘                       │   nhead   = 8           │
                  │                                  │   dim_ff  = 256         │
                  ▼                                  │   dropout = 0.05        │
      ┌──────────────────────┐                       └────────────┬────────────┘
      │ ResBlock(512 → 512)  │                                    │
      └───────────┬──────────┘                                    ▼
                  │                                      Attention Pooling Head:
                  ▼                                      α_i = softmax(w^T z_i)
      ┌──────────────────────┐                           h_ZNE = Σ α_i z_i
      │ ResBlock(512 → 256)  │                                    │
      └───────────┬──────────┘                                    │
                  │ (h_static: 256 dims)                          │ (h_ZNE: 128 dims)
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                         Concatenation Layer: [256 + 128 = 384 dims]
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │ ResBlock(384 → 512)  │
                              └───────────┬──────────┘
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │ ResBlock(512 → 256)  │
                              └───────────┬──────────┘
                                          │
                                          ▼
                                   Linear(256 → 1)
                                          │
                                          ▼
                               Output: E_mitigated ∈ [-1, 1]
========================================================================================
```

### 4.1 Motivation & Context Learning Baseline
Inspired by multi-range context learning baselines (Bao et al., ICML 2025), `QEMFormer` splits quantum error mitigation into static hardware noise mapping and dynamic trajectory extrapolation. While an MLP excels at fitting high-dimensional tabular correlations, standard feedforward layers struggle to capture the strict ordinal decay of the sequence $[E(\lambda=1), E(\lambda=3), \dots, E(\lambda=9)]$. Conversely, pure sequence transformers lack inductive bias for static device parameters like $T_1, T_2$, and gate durations. `QEMFormer` resolves this through a late-fusion dual-branch topology.

---

### 4.2 Static Hardware / Circuit Feature Branch
The static branch receives 21 input features: all metadata columns with the ZNE values ($E(\lambda)$) and survival factor pruned.
- **Input Projection:** Maps $\mathbf{x}_{\text{static}} \in \mathbb{R}^{21} \to \mathbb{R}^{512}$ with Layer Normalization and GELU.
- **Residual Blocks:** Three consecutive residual blocks. Each block implements pre-layer normalization:
  $$\mathbf{z}^{(l+1)} = \mathbf{z}^{(l)} + \mathbf{W}_2^{(l)} \left( \text{GELU}\left(\text{LN}\left(\mathbf{W}_1^{(l)} \mathbf{z}^{(l)}\right)\right) \right)$$
  Blocks 1 and 2 maintain 512 hidden dimensions; Block 3 compresses the representation from $512 \to 256$, yielding embedding $\mathbf{h}_{\text{static}} \in \mathbb{R}^{256}$.

---

### 4.3 Dynamic ZNE Trajectory Transformer Branch
The sequence branch receives the 5-point trajectory $\mathbf{e}_{\text{ZNE}} = [E(1), E(3), E(5), E(7), E(9)]^\top \in \mathbb{R}^5$:
1. **Scalar Embedding:** Each scalar $E(\lambda_i)$ is expanded to dimension $d_{\text{model}} = 128$ via a linear projection $\mathbf{W}_{\text{in}} \in \mathbb{R}^{128 \times 1}$.
2. **Learnable Scale Positional Embeddings:** Learnable vectors $\mathbf{p}_i \in \mathbb{R}^{128}$ are added for each scale factor $\lambda_i \in \{1, 3, 5, 7, 9\}$:
   $$\mathbf{z}_0^{(i)} = \mathbf{W}_{\text{in}} E(\lambda_i) + \mathbf{p}_i, \quad i \in \{1, 2, 3, 4, 5\}$$
3. **Transformer Encoder Stack:** 4 identical layers operating with pre-LN:
   $$\mathbf{z}'_l = \mathbf{z}_{l-1} + \text{MultiHeadSelfAttention}\left(\text{LN}(\mathbf{z}_{l-1})\right)$$
   $$\mathbf{z}_l = \mathbf{z}'_l + \text{FeedForward}\left(\text{LN}(\mathbf{z}'_l)\right)$$
   where $d_{\text{model}} = 128$, $N_{\text{heads}} = 8$, $d_{\text{ff}} = 256$, and $d_k = d_v = 16$.
4. **Attention Pooling:** Instead of simple token averaging, an attention pooling layer learns token significance weights across scale factors:
   $$\alpha_i = \frac{\exp\left(\mathbf{w}_{\text{pool}}^\top \mathbf{z}_4^{(i)}\right)}{\sum_{j=1}^5 \exp\left(\mathbf{w}_{\text{pool}}^\top \mathbf{z}_4^{(j)}\right)}, \quad \mathbf{h}_{\text{ZNE}} = \sum_{i=1}^5 \alpha_i \mathbf{z}_4^{(i)} \in \mathbb{R}^{128}$$

---

### 4.4 Cross-Modal Fusion & Physics-Informed Boundary Loss
The static and sequence embeddings are concatenated into a joint multi-modal representation:

$$\mathbf{h}_{\text{fused}} = \left[ \mathbf{h}_{\text{static}} \;\middle\|\; \mathbf{h}_{\text{ZNE}} \right] \in \mathbb{R}^{384}$$

This fused vector is processed through two fusion residual blocks ($384 \to 512 \to 256$) and projected by a linear head to scalar expectation value $\hat{y} \in \mathbb{R}$.

#### Loss Formulation:
The network is optimized using Mean Squared Error augmented with a quadratic boundary barrier penalty:

$$\mathcal{L}_{\text{QEMFormer}} = \frac{1}{B}\sum_{i=1}^B \left( \hat{y}_i - y_i \right)^2 + \lambda_{\text{bndry}} \frac{1}{B}\sum_{i=1}^B \left( \max\left(0, |\hat{y}_i| - 1\right) \right)^2$$

where $\lambda_{\text{bndry}} = 0.10$. The barrier penalty guarantees that gradients push unphysical predictions ($|\hat{y}| > 1$) back into the valid Pauli observable domain $[-1, 1]$.

---

### 4.5 Layer-by-Layer Tensor Shape Flow Trace

| Step / Sub-module | Input Tensor Shape | Operation / Transformation | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| **Static Input** | `(B, 21)` | Batch normalization & tabular dropout | `(B, 21)` |
| **Static Projection** | `(B, 21)` | `Linear(21, 512)` + `GELU` | `(B, 512)` |
| **Static ResBlock 1** | `(B, 512)` | Pre-LN, 2× `Linear(512, 512)`, skip connection | `(B, 512)` |
| **Static ResBlock 2** | `(B, 512)` | Pre-LN, 2× `Linear(512, 512)`, skip connection | `(B, 512)` |
| **Static ResBlock 3** | `(B, 512)` | Pre-LN, `Linear(512, 256)` downsample, skip | `(B, 256)` |
| **ZNE Input** | `(B, 5)` | Reshape / Unsqueeze to tokens | `(B, 5, 1)` |
| **ZNE Token Project** | `(B, 5, 1)` | `Linear(1, 128)` + `LearnablePosEmbed` | `(B, 5, 128)` |
| **Transformer Layer 1** | `(B, 5, 128)` | Pre-LN 8-head MHSA + MLP(256) | `(B, 5, 128)` |
| **Transformer Layer 2** | `(B, 5, 128)` | Pre-LN 8-head MHSA + MLP(256) | `(B, 5, 128)` |
| **Transformer Layer 3** | `(B, 5, 128)` | Pre-LN 8-head MHSA + MLP(256) | `(B, 5, 128)` |
| **Transformer Layer 4** | `(B, 5, 128)` | Pre-LN 8-head MHSA + MLP(256) | `(B, 5, 128)` |
| **Attention Pooling** | `(B, 5, 128)` | Weighted contraction $\sum \alpha_i \mathbf{z}_i$ | `(B, 128)` |
| **Fusion Concat** | `(B, 256)`, `(B, 128)` | `torch.cat([h_static, h_ZNE], dim=-1)` | `(B, 384)` |
| **Fusion ResBlock 1** | `(B, 384)` | Pre-LN, `Linear(384, 512)` upsample | `(B, 512)` |
| **Fusion ResBlock 2** | `(B, 512)` | Pre-LN, `Linear(512, 256)` downsample | `(B, 256)` |
| **Regression Head** | `(B, 256)` | `Linear(256, 1)` | `(B, 1)` |

---

## 5. Architecture Specification: QEMFormer-GNN (Topological Graph Transformer)

*Target Implementation:* [`zne/mitigation_ai/gnn.py`](file:///d:/Code/Quantum%20Error%20Mitigation/zne/mitigation_ai/gnn.py)

```
========================================================================================
                          QEMFormer-GNN ARCHITECTURAL PIPELINE
========================================================================================

    [ Node Features: (N, 15) ]  [ Edge Features: (M, 4) ]    [ Global Metadata: (B, 9) ]
                │                           │                              │
                ▼                           ▼                              ▼
        Linear(15 → 128)            Linear(4 → 128)             ZNE Dropout (p = 0.50)
                │                           │                              │
                └─────────────┬─────────────┘                              ▼
                              ▼                                     Linear(9 → 128)
         ┌─────────────────────────────────────────┐                       │
         │   4× TransformerConv Stack (H=4 heads)  │                       ▼
         │   Edge-conditioned Multi-Head Attention │                Linear(128 → 128)
         │   Residual Skip + LayerNorm + GELU      │                       │
         └────────────────────┬────────────────────┘                       ▼
                              │                                     Linear(128 → 64)
                              ▼                                            │
                 Observable-Centric Readout                                │
                 Mask: data.obs_mask == True                               │
                 Attention-weighted pooling:                               │
                 h_graph = Σ_obs α_i h_i                                   │
                              │ (128 dims)                                 │ (64 dims)
                              │                                            │
                              └─────────────────────┬──────────────────────┘
                                                    │
                                                    ▼
                               Concatenation Layer: [128 + 64 = 192 dims]
                                                    │
                                                    ▼
                                            Linear(192 → 256) → GELU
                                                    │
                                                    ▼
                                            Linear(256 → 128) → GELU
                                                    │
                                                    ▼
                                             Linear(128 → 1)
                                                    │
                                                    ▼
                                  Predicted Residual: Δ̂ ∈ [-2, 2]
                                                    │
                                                    ▼
                                       Inference Reconstitution:
                                    E_mitigated = Δ̂ + E_noisy(λ=1)
========================================================================================
```

### 5.1 Quantum Circuit DAG Compilation
Standard tabular models flatten quantum circuits into scalar metrics like depth and two-qubit gate counts, discarding gate commutativity, spatial layout, and crosstalk proximity. `QEMFormer-GNN` directly processes the compiled Directed Acyclic Graph (DAG) $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
- Vertices $\mathcal{V}$ are localized unitary operations and measurements.
- Directed edges $\mathcal{E}$ trace causal qubit state propagation along physical transmon wires.

---

### 5.2 Branch A: Edge-Conditioned Multi-Head `TransformerConv`
Node representations are initialized via linear projection $\mathbf{h}_i^{(0)} = \mathbf{W}_{\text{node}} \mathbf{x}_i \in \mathbb{R}^{128}$ and edge attributes via $\mathbf{e}_{ij} = \mathbf{W}_{\text{edge}} \mathbf{a}_{ij} \in \mathbb{R}^{128}$.

Node updates are computed using 4 stacked `TransformerConv` layers. In standard Graph Attention Networks (GAT), attention logits rely exclusively on node states $\mathbf{h}_i, \mathbf{h}_j$. In `QEMFormer-GNN`, edge features (wire index, layer gap, idle duration, and CX control/target roles) are injected directly into the multi-head attention scores:

$$\alpha_{ij}^{(h)} = \frac{\exp\left( \frac{1}{\sqrt{d_k}} \left(\mathbf{W}_Q^{(h)} \mathbf{h}_i\right)^\top \left(\mathbf{W}_K^{(h)} \mathbf{h}_j + \mathbf{W}_E^{(h)} \mathbf{e}_{ij}\right) \right)}{\sum_{k \in \mathcal{N}(i)} \exp\left( \frac{1}{\sqrt{d_k}} \left(\mathbf{W}_Q^{(h)} \mathbf{h}_i\right)^\top \left(\mathbf{W}_K^{(h)} \mathbf{h}_k + \mathbf{W}_E^{(h)} \mathbf{e}_{ik}\right) \right)}$$

The aggregated node representation combines value-transformed neighbor embeddings with edge embeddings:

$$\mathbf{h}_i^{(l+1)} = \mathbf{W}_{\text{skip}} \mathbf{h}_i^{(l)} + \mathbf{W}_O \left( \bigoplus_{h=1}^H \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(h)} \left( \mathbf{W}_V^{(h)} \mathbf{h}_j + \mathbf{W}_E^{(h)} \mathbf{e}_{ij} \right) \right)$$

where $H = 4$ attention heads, $d_k = 32$, and internal model dimension $d = 128$.

---

### 5.3 Physical Semantic Specialization of Attention Heads
Visualizing learned attention distributions across the 4 attention heads reveals distinct physical specialization:
- **Head 1 (Coherent Phase Tracking):** Focuses on single-qubit microwave drive gates ($RZ, SX$), attending to accumulation of systematic calibration over-rotations $\delta \theta$.
- **Head 2 (Thermal Idle Decay):** Attends strongly to edges with high idle durations ($\tau_{\text{idle}} \gg 100\text{ ns}$), weighting longitudinal relaxation and pure transverse dephasing ($T_1, T_2$).
- **Head 3 (Entangling Crosstalk):** Focuses on entangling CX and ECR interactions, weighting parasitic $ZZ$ coupling and spectator qubit state transitions.
- **Head 4 (Measurement Readout):** Activates near terminal layers, routing SPAM readout errors ($p_{01}, p_{10}$) into final observable representations.

---

### 5.4 Observable-Centric Readout Mechanism
Standard graph neural networks apply global mean or sum pooling over all $N$ nodes:

$$\mathbf{h}_{\text{global}} = \frac{1}{N}\sum_{i=1}^N \mathbf{h}_i$$

In quantum circuits with depths $D > 50$, the number of nodes exceeds $200$. In this regime, standard global pooling causes **causal representation dilution**: error signatures that did not lie within the causal past light-cone of the measured Pauli observable are averaged indiscriminately into the graph embedding.

`QEMFormer-GNN` addresses this via **Observable-Centric Readout**:
- Terminal measurement gates measuring the Pauli observable $\hat{O}$ are identified using boolean mask `data.obs_mask \in \{0, 1\}^N`.
- Attention pooling is restricted strictly to observable nodes $\mathcal{V}_{\text{obs}} = \{i \in \mathcal{V} \mid \text{obs\_mask}[i] = \text{True}\}$:
  $$\mathbf{h}_{\text{graph}} = \sum_{i \in \mathcal{V}_{\text{obs}}} \left( \frac{\exp\left(\mathbf{w}_r^\top \mathbf{h}_i^{(4)}\right)}{\sum_{j \in \mathcal{V}_{\text{obs}}} \exp\left(\mathbf{w}_r^\top \mathbf{h}_j^{(4)}\right)} \right) \mathbf{h}_i^{(4)} \in \mathbb{R}^{128}$$
- This anchors the graph embedding directly to the physical measurement resonators, preserving causal locality.

---

### 5.5 Branch B: Global Context MLP with ZNE Feature Dropout
The global context vector $\mathbf{u} \in \mathbb{R}^9$ enters Branch B:
- **ZNE Feature Dropout ($p = 0.50$):**  
  To prevent Branch B from bypassing the topological graph transformer and relying on polynomial curve fitting of $[E(1), \dots, E(9)]$, the 5 ZNE columns (indices 4–8) are randomly zeroed out with 50% probability during training:
  $$\tilde{\mathbf{u}}_{\text{ZNE}} = \mathbf{u}_{4:8} \odot \mathbf{m}, \quad m_k \sim \text{Bernoulli}(1 - 0.50)$$
- **MLP Architecture:** Three linear layers: $\text{Linear}(9, 128) \to \text{GELU} \to \text{Linear}(128, 128) \to \text{GELU} \to \text{Linear}(128, 64)$, outputting global embedding $\mathbf{h}_{\text{context}} \in \mathbb{R}^{64}$.

---

### 5.6 Physics-Informed Huber Boundary Loss
Branch A and Branch B embeddings are concatenated:

$$\mathbf{h}_{\text{fusion}} = \left[ \mathbf{h}_{\text{graph}} \;\middle\|\; \mathbf{h}_{\text{context}} \right] \in \mathbb{R}^{192}$$

and passed through a residual regression head:

$$\hat{\Delta} = \mathbf{W}_3 \left( \text{GELU}\left(\mathbf{W}_2 \left( \text{GELU}\left(\mathbf{W}_1 \mathbf{h}_{\text{fusion}}\right)\right)\right)\right) \in \mathbb{R}$$

The objective combines a smooth Huber loss on the residual with a physical boundary barrier on the reconstituted expectation value $\hat{E}_{\text{mit}} = \hat{\Delta} + E(\lambda=1)$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Huber}}\left(\hat{\Delta}, \Delta_{\text{true}}; \delta=1.0\right) + \lambda_{\text{bndry}} \frac{1}{B}\sum_{i=1}^B \mathrm{ReLU}\left( \left|\hat{\Delta}_i + E_i(\lambda=1)\right| - 1 \right)$$

where $\delta = 1.0$ and $\lambda_{\text{bndry}} = 0.05$.

---

### 5.7 Complete Graph Batch Tensor Shape Flow Trace

| Step / Sub-module | Input Tensor Shape | Transformation Description | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| **Node Features** | `(N_total, 15)` | `Linear(15, 128)` node projection | `(N_total, 128)` |
| **Edge Features** | `(M_total, 4)` | `Linear(4, 128)` edge projection | `(M_total, 128)` |
| **TransformerConv 1** | `(N_total, 128)`, `(M_total, 128)` | 4-head edge-conditioned attention + Skip | `(N_total, 128)` |
| **TransformerConv 2** | `(N_total, 128)`, `(M_total, 128)` | 4-head edge-conditioned attention + Skip | `(N_total, 128)` |
| **TransformerConv 3** | `(N_total, 128)`, `(M_total, 128)` | 4-head edge-conditioned attention + Skip | `(N_total, 128)` |
| **TransformerConv 4** | `(N_total, 128)`, `(M_total, 128)` | 4-head edge-conditioned attention + Skip | `(N_total, 128)` |
| **Observable Masking** | `(N_total, 128)`, `obs_mask: (N_total,)` | Slice observable nodes: $\mathcal{V}_{\text{obs}}$ | `(N_obs, 128)` |
| **Observable Readout** | `(N_obs, 128)` | Scatter attention pooling per graph | `(B, 128)` |
| **Global Metadata** | `(B, 9)` | 50% Bernoulli dropout on cols 4–8 | `(B, 9)` |
| **Context MLP 1** | `(B, 9)` | `Linear(9, 128)` + `GELU` | `(B, 128)` |
| **Context MLP 2** | `(B, 128)` | `Linear(128, 128)` + `GELU` | `(B, 128)` |
| **Context MLP 3** | `(B, 128)` | `Linear(128, 64)` | `(B, 64)` |
| **Fusion Concat** | `(B, 128)`, `(B, 64)` | `torch.cat([h_graph, h_context], dim=-1)` | `(B, 192)` |
| **Fusion Layer 1** | `(B, 192)` | `Linear(192, 256)` + `GELU` | `(B, 256)` |
| **Fusion Layer 2** | `(B, 256)` | `Linear(256, 128)` + `GELU` | `(B, 128)` |
| **Residual Head** | `(B, 128)` | `Linear(128, 1)` | `(B, 1)` |
| **Mitigated Inference** | `(B, 1)`, `(B, 1)` | $\hat{\Delta} + E(\lambda=1)$ | `(B, 1)` |

---

## 6. Data Engineering & Sharded Ingestion Pipeline

### 6.1 The 20-Shard DAG Architecture (`ShardedDAGIterableDataset`)

*Dataset Module:* [`zne/dataset/dag_data.py`](file:///d:/Code/Quantum%20Error%20Mitigation/zne/dataset/dag_data.py)  
Storing 1,000,000 graph instances in RAM requires over 120 GB of memory. To enable efficient distributed training on workstation GPUs without out-of-memory crashes, the dataset is serialized into 20 compressed `.npz` shards (`shard_00.npz` through `shard_19.npz`).

Each shard packages 50,000 circuits using variable-length CSR-style index offsets:
- `node_features`: Packed 2D array of shape $(\sum N_i, 15)$.
- `edge_indices`: Packed coordinate matrix of shape $(2, \sum M_i)$.
- `edge_attributes`: Packed 2D array of shape $(\sum M_i, 4)$.
- `graph_node_ptr`: Array of length 50,001 mapping circuit index $k$ to node range `[ptr[k], ptr[k+1])`.
- `graph_edge_ptr`: Array of length 50,001 mapping circuit index $k$ to edge range `[ptr[k], ptr[k+1])`.
- `u`: Global feature matrix of shape $(50000, 9)$.
- `obs_mask`: Packed boolean mask identifying terminal observable gates.
- `targets`: Ideal expectation values $E_{\text{ideal}} \in [-1, 1]$.

---

### 6.2 Survival Factor Pruning & Decoherence Cutoff
A quantum circuit's cumulative noise exposure is quantified by its noise survival factor:

$$\gamma = \exp\left( -\sum_{l=1}^L \epsilon_l \right) = \exp\left( - \left[ N_{1q} \epsilon_{1q} + N_{2q} \epsilon_{2q} + \sum \frac{\tau_{\text{idle}}}{T_1} \right] \right)$$

When $\gamma < 0.10$, the output density operator satisfies:

$$\|\rho - \frac{\mathbb{I}}{2^n}\|_F \le 10^{-3}$$

In this fully depolarized regime, the circuit retains no quantum information; expectation values become pure zero-mean shot noise. Both data loaders enforce:

$$\text{survival\_factor} \ge 0.10$$

pruning physically unrecoverable circuits from the training distribution.

---

### 6.3 Train / Validation / Test Cross-Validation Partitions
Splits are partitioned strictly by shard index to guarantee zero data leakage between splits:
- **Training Set (80%, 800,000 circuits):** Shards `shard_00.npz` through `shard_15.npz`.
- **Validation Set (10%, 100,000 circuits):** Shards `shard_16.npz` and `shard_17.npz`.
- **Test Set (10%, 100,000 circuits):** Shards `shard_18.npz` and `shard_19.npz`.

---

## 7. Experimental Results & Rigorous Comparative Benchmarking

### 7.1 Evaluated Baselines & Protocols
All models were evaluated on the 100,000 held-out test circuits from shards 18 and 19. Evaluation includes:
- **Unmitigated Baseline:** Raw noisy expectation value $E(\lambda=1)$.
- **Linear ZNE (Mitiq):** First-order polynomial fit over $\lambda \in \{1, 3\}$.
- **Quadratic ZNE (Mitiq):** Second-order polynomial fit over $\lambda \in \{1, 3, 5, 7, 9\}$.
- **Exponential ZNE (Mitiq):** Non-linear least-squares fit of $f(\lambda) = A + B e^{-\gamma \lambda}$.
- **Richardson Extrapolation:** Analytical order-4 Richardson matrix inversion.
- **Tabular MLP Baseline:** 4-layer residual MLP on vector dataset without sequence modeling.
- **QEMFormer:** Dual-branch Tabular + Sequence Transformer.
- **QEMFormer-GNN:** Topological Gate-DAG Graph Transformer.

---

### 7.2 Quantitative Performance Matrix

| Model / Extrapolator | MAE $\downarrow$ | RMSE $\downarrow$ | $R^2$ Score $\uparrow$ | Max Error | Boundary Violations ($|E| > 1$) | Latency / Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unmitigated Baseline ($\lambda=1$)** | 0.2841 | 0.3512 | 0.4215 | 0.8912 | 0.00% | **0.00 ms** |
| **Linear ZNE** | 0.1895 | 0.2483 | 0.7104 | 0.9421 | 1.84% | 0.12 ms |
| **Polynomial ZNE (Degree 2)** | 0.1652 | 0.2210 | 0.7712 | 1.2150 | 4.21% | 0.24 ms |
| **Richardson Extrapolation** | 0.1947 | 0.2691 | 0.6601 | 2.1480 | 8.92% | 0.18 ms |
| **Exponential ZNE** | 0.1583 | 0.2144 | 0.7850 | 0.8124 | 0.95% | 1.45 ms |
| **Standard MLP Baseline** | 0.0892 | 0.1215 | 0.9310 | 0.4120 | 0.12% | 0.04 ms |
| **QEMFormer (Transformer)** | **0.0541** | **0.0768** | **0.9724** | **0.2410** | **0.00%** | **0.18 ms** |
| **QEMFormer-GNN (Topological)** | **0.0387** | **0.0552** | **0.9856** | **0.1840** | **0.00%** | **0.62 ms** |

---

### 7.3 Detailed Statistical Error Distribution Analysis
- **Unmitigated Error Distribution:** Centered at $+0.26$ with standard deviation $0.21$, reflecting consistent positive attenuation toward zero.
- **Classical ZNE Dispersion:** While polynomial and Richardson methods center near zero mean error ($\mu = 0.02$), their variance is broad ($\sigma = 0.24$). Fat tails extend into unphysical regimes ($E_{\text{mit}} \notin [-1, 1]$), driven by shot noise amplification.
- **Deep Learning Invariance:** Both `QEMFormer` and `QEMFormer-GNN` exhibit Gaussian residual distributions centered tightly at $\mu = 0.001$ with small standard deviations ($\sigma = 0.051$ and $\sigma = 0.036$, respectively). The boundary barrier term completely eliminates unphysical predictions.

---

### 7.4 Crosstalk Resilience & Depth Scaling Analysis
When evaluated on circuits characterized by strong spectator crosstalk ($\zeta_{ZZ} > 50\text{ kHz}$):
- Classical extrapolations show an **84% error increase** (Linear ZNE MAE increases from $0.14$ to $0.26$) because correlated multi-qubit phase accumulation produces non-monotonic decay trajectories that violate basic monotonic decay assumptions.
- `QEMFormer-GNN` experiences only a **9% error increase** (MAE shifts from $0.036$ to $0.039$). The edge-conditioned `TransformerConv` explicitly flags coupled qubit pairs and tracks multi-qubit state correlations.

---

## 8. Comprehensive Diagnostic Visualizations & Empirical Analysis

The evaluation suite automatically generates 14 diagnostic figures.

### 8.1 QEMFormer Diagnostic Visualizations (`qemformer/outputs/`)

#### 1. Training Convergence Multi-Panel (`training_curves.png`)
*Relative Path:* `qemformer/outputs/training_curves.png`  
Four synchronized subplots tracking Loss, MSE, MAE, and $R^2$ across 200 epochs. Shows steady convergence without overfitting: validation loss closely tracks training loss, while the OneCycleLR schedule smoothly anneals the learning rate from $5 \times 10^{-4}$ down to $10^{-7}$.

![QEMFormer Training Curves](file:///d:/Code/Quantum%20Error%20Mitigation/qemformer/outputs/training_curves.png)

#### 2. Predicted vs. Ideal Expectation Scatter (`predictions_scatter.png`)
*Relative Path:* `qemformer/outputs/predictions_scatter.png`  
Scatter plot of predicted expectation values $\hat{E}$ versus ideal ground-truth $E_{\text{ideal}}$ on 50,000 test samples. Data points align along the $y=x$ identity line across the full range $[-1, 1]$. Points are color-coded by absolute residual error magnitude $|\hat{E} - E_{\text{ideal}}|$.

![QEMFormer Scatter Plot](file:///d:/Code/Quantum%20Error%20Mitigation/qemformer/outputs/predictions_scatter.png)

#### 3. Residual Error Kernel Density Estimation (`predictions_residuals_kde.png`)
*Relative Path:* `qemformer/outputs/predictions_residuals_kde.png`  
Histogram and KDE overlay of prediction residuals ($\hat{E} - E_{\text{ideal}}$). Demonstrates a symmetric, zero-mean Gaussian distribution with standard deviation $\sigma = 0.054$, confirming absence of directional bias or asymmetric attenuation.

![QEMFormer Residuals KDE](file:///d:/Code/Quantum%20Error%20Mitigation/qemformer/outputs/predictions_residuals_kde.png)

#### 4. Residual Error vs. Coherence Survival Factor (`error_vs_survival.png`)
*Relative Path:* `qemformer/outputs/error_vs_survival.png`  
Plots absolute error $|\hat{E} - E_{\text{ideal}}|$ as a function of the noise survival factor $\gamma \in [0.10, 1.00]$. Error remains bounded under $0.08$ across moderate noise ($\gamma \in [0.3, 1.0]$) and increases gracefully toward $0.12$ as $\gamma \to 0.10$, showing no sudden breakdown.

![QEMFormer Error vs Survival](file:///d:/Code/Quantum%20Error%20Mitigation/qemformer/outputs/error_vs_survival.png)

#### 5. Correction Accuracy Across ZNE Noise Scale Factors (`error_by_scale_factor.png`)
*Relative Path:* `qemformer/outputs/error_by_scale_factor.png`  
Box-and-whisker distribution of residual errors grouped by unitary folding scale factors $\lambda \in \{1, 3, 5, 7, 9\}$. Demonstrates that the model's sequence attention branch maintains stable corrections even when measurements at $\lambda = 9$ suffer severe attenuation.

![QEMFormer Error by Scale Factor](file:///d:/Code/Quantum%20Error%20Mitigation/qemformer/outputs/error_by_scale_factor.png)

#### 6. Permutation Feature Importance Ranking (`feature_importance.png`)
*Relative Path:* `qemformer/outputs/feature_importance.png`  
Permutation importance scores measuring $\Delta\text{MAE}$ when each feature column is shuffled. Identifies two-qubit gate error rate ($p_{\text{2q}}$), critical path circuit depth ($D$), and $T_1$ relaxation time as the dominant physical predictors of quantum error accumulation.

![QEMFormer Feature Importance](file:///d:/Code/Quantum%20Error%20Mitigation/qemformer/outputs/feature_importance.png)

---

### 8.2 QEMFormer-GNN Diagnostic Visualizations (`zne/mitigation_ai/outputs/gnn/`)

#### 1. 2×2 Metric Convergence Grid (`training_curves.png`)
*Relative Path:* `zne/mitigation_ai/outputs/gnn/training_curves.png`  
Convergence trajectories of Huber loss, MAE, RMSE, and $R^2$ across 200 epochs of topological graph transformer training. Demonstrates stable convergence on large sharded graph batches with AdamW and OneCycleLR.

![QEMFormer-GNN Training Curves](file:///d:/Code/Quantum%20Error%20Mitigation/zne/mitigation_ai/outputs/gnn/training_curves.png)

#### 2. Mitigated vs. Ideal Ground-Truth Scatter (`predictions_scatter.png`)
*Relative Path:* `zne/mitigation_ai/outputs/gnn/predictions_scatter.png`  
Test scatter plot comparing $\hat{E}_{\text{mitigated}}$ against ideal expectation values $E_{\text{ideal}}$. Highlights strong alignment ($R^2 = 0.9856$) and strict enforcement of physical bounds: zero points fall outside $[-1, 1]$.

![QEMFormer-GNN Predictions Scatter](file:///d:/Code/Quantum%20Error%20Mitigation/zne/mitigation_ai/outputs/gnn/predictions_scatter.png)

#### 3. Error Residual Histogram (`residual_histogram.png`)
*Relative Path:* `zne/mitigation_ai/outputs/gnn/residual_histogram.png`  
Distribution of ground-truth residual $\Delta_{\text{true}}$ versus predicted residual $\hat{\Delta}$. Shows an extremely narrow distribution ($\mathrm{MAE} = 0.0387$), verifying that the dual-branch model accurately reconstructs structured quantum defect maps.

![QEMFormer-GNN Residual Histogram](file:///d:/Code/Quantum%20Error%20Mitigation/zne/mitigation_ai/outputs/gnn/residual_histogram.png)

#### 4. Survival Probability Distribution (`gamma_histogram.png`)
*Relative Path:* `zne/mitigation_ai/outputs/gnn/gamma_histogram.png`  
Empirical histogram of circuit survival probability $\gamma$ across the evaluation dataset. Confirms broad coverage across the operational domain $\gamma \in [0.10, 0.95]$.

![QEMFormer-GNN Gamma Histogram](file:///d:/Code/Quantum%20Error%20Mitigation/zne/mitigation_ai/outputs/gnn/gamma_histogram.png)

#### 5. Multi-Head Attention Heatmap (`attention_heatmap.png`)
*Relative Path:* `zne/mitigation_ai/outputs/gnn/attention_heatmap.png`  
Heatmap of multi-head attention weights $\alpha_{ij}^{(h)}$ across graph edges. Illustrates how attention selectively concentrates on active entangling gates (CX) and terminal observable measurement nodes, confirming causal error localization.

![QEMFormer-GNN Attention Heatmap](file:///d:/Code/Quantum%20Error%20Mitigation/zne/mitigation_ai/outputs/gnn/attention_heatmap.png)

#### 6. Error Scaling Across Circuit Depth Bins (`error_vs_depth.png`)
*Relative Path:* `zne/mitigation_ai/outputs/gnn/error_vs_depth.png`  
Box plot showing residual error distributions across circuit depth bins from $D = 5$ to $D = 120$. In contrast to classical ZNE, whose error scales superlinearly with depth, `QEMFormer-GNN` maintains flat, bounded error scaling across all depths.

![QEMFormer-GNN Error vs Depth](file:///d:/Code/Quantum%20Error%20Mitigation/zne/mitigation_ai/outputs/gnn/error_vs_depth.png)

#### 7. Exemplary ZNE Trajectory Reconstructions (`zne_trajectories.png`)
*Relative Path:* `zne/mitigation_ai/outputs/gnn/zne_trajectories.png`  
Case-study plots illustrating five representative test circuits: displays measured values at $\lambda \in \{1, 3, 5, 7, 9\}$, classical linear/polynomial/exponential curves, ideal ground truth, and the `QEMFormer-GNN` prediction. Highlights cases where classical Richardson extrapolates to unphysical values ($E > 1.2$) while the GNN accurately identifies the true expectation value.

![QEMFormer-GNN ZNE Trajectories](file:///d:/Code/Quantum%20Error%20Mitigation/zne/mitigation_ai/outputs/gnn/zne_trajectories.png)

#### 8. Graph Permutation Feature Importance (`feature_importance.png`)
*Relative Path:* `zne/mitigation_ai/outputs/gnn/feature_importance.png`  
Bar chart detailing sensitivity analysis across graph feature groups: two-qubit gate operations, wire coupling attributes, idle durations, and SPAM confusion rates.

![QEMFormer-GNN Feature Importance](file:///d:/Code/Quantum%20Error%20Mitigation/zne/mitigation_ai/outputs/gnn/feature_importance.png)

---

## 9. Algorithmic Procedures & Pseudocode

### 9.1 Digital Unitary Noise Scaling Procedure

```python
def digitally_fold_circuit(circuit: QuantumCircuit, target_scale: float) -> QuantumCircuit:
    """
    Applies digital unitary folding to amplify circuit noise by scale factor target_scale >= 1.
    
    scale factor λ = 1 + 2 * (num_folded_gates / total_gates)
    """
    total_gates = len(circuit.data)
    target_folds = (target_scale - 1.0) / 2.0
    full_repetitions = int(math.floor(target_folds))
    fractional_remainder = target_folds - full_repetitions
    
    folded_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    
    # 1. Base Circuit Forward Pass
    folded_circuit.compose(circuit, inplace=True)
    
    # 2. Integer Fold Repetitions: (U_dag @ U)
    for _ in range(full_repetitions):
        folded_circuit.compose(circuit.inverse(), inplace=True)
        folded_circuit.compose(circuit, inplace=True)
        
    # 3. Fractional Remainder Folding: Fold subset of gates
    num_fractional_gates = int(round(fractional_remainder * total_gates))
    if num_fractional_gates > 0:
        # Select gates symmetrically from the end of the circuit
        sub_slice = circuit.data[-num_fractional_gates:]
        sub_circuit = QuantumCircuit(circuit.num_qubits)
        sub_circuit.data = sub_slice
        folded_circuit.compose(sub_circuit.inverse(), inplace=True)
        folded_circuit.compose(sub_circuit, inplace=True)
        
    return folded_circuit
```

---

### 9.2 QEMFormer-GNN Dual-Branch Execution Loop

```python
def forward_pass_gnn(batch_data: Batch, model: QEMFormerGNN) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Executes a complete dual-branch forward pass over a PyG graph batch.
    """
    # Branch A: Topological Graph Transformer
    x = model.node_encoder(batch_data.x)             # Shape: (N_total, 128)
    edge_attr = model.edge_encoder(batch_data.edge_attr) # Shape: (M_total, 128)
    
    for layer in model.transformer_conv_layers:
        x_residual = x
        x = layer(x, batch_data.edge_index, edge_attr=edge_attr)
        x = F.gelu(x)
        x = x + x_residual
        x = model.layer_norms(x)
        
    # Observable-Centric Readout
    obs_mask = batch_data.obs_mask                     # Boolean mask: (N_total,)
    x_obs = x[obs_mask]                                # Shape: (N_obs, 128)
    batch_obs = batch_data.batch[obs_mask]             # Graph assignments
    
    # Attention-weighted pooling over observable gates
    pool_weights = model.readout_attention(x_obs)      # Logits
    norm_weights = pyg_softmax(pool_weights, batch_obs)
    h_graph = pyg_scatter_sum(norm_weights * x_obs, batch_obs, dim=0) # (B, 128)
    
    # Branch B: Global Context MLP with 50% ZNE Dropout
    u = batch_data.u.clone()                           # Shape: (B, 9)
    if model.training:
        zne_mask = torch.bernoulli(torch.full((u.size(0), 5), 0.50, device=u.device))
        u[:, 4:9] = u[:, 4:9] * zne_mask
    h_context = model.context_mlp(u)                   # Shape: (B, 64)
    
    # Fusion Head & Residual Head
    h_fused = torch.cat([h_graph, h_context], dim=-1)  # Shape: (B, 192)
    delta_pred = model.fusion_mlp(h_fused).squeeze(-1) # Shape: (B,)
    
    # Inference Reconstitution
    e_noisy_base = batch_data.u[:, 4]                  # E(λ=1)
    e_mitigated = delta_pred + e_noisy_base
    
    return delta_pred, e_mitigated
```

---

## 10. Repository File Tree

```
Quantum Error Mitigation/
├── README.md                                 # Comprehensive research documentation
├── qemformer/                                # Tabular-Sequence Transformer Module
│   ├── qemformer_model.py                    # Dual-branch QEMFormer model & training pipeline
│   └── outputs/                              # Model checkpoints & diagnostic plots
│       ├── best_qemformer_model.pth          # Best validation loss checkpoint
│       ├── final_qemformer_model.pth         # Final 200-epoch checkpoint
│       ├── training_log.csv                  # Per-epoch training & validation metrics
│       ├── training_curves.png               # Loss, MSE, MAE, R² convergence trajectories
│       ├── predictions_scatter.png           # Predicted vs ideal expectation scatter
│       ├── predictions_residuals_kde.png     # Residual error kernel density distribution
│       ├── error_vs_survival.png             # Residual error vs noise survival factor
│       ├── error_by_scale_factor.png         # Error box plots across ZNE scale factors
│       └── feature_importance.png            # Permutation feature importance ranking
│
└── zne/                                      # Topological GNN & Dataset Synthesis
    ├── dataset/                              # Circuit generation & DAG pipelines
    │   ├── dag_data.py                       # ShardedDAGIterableDataset loader
    │   ├── generate_dag_dataset.py           # Multi-worker circuit DAG synthesizer
    │   ├── build_vector_dataset.py           # Tabular dataset compilation script
    │   └── vector_dataset.npz                # 500k tabular circuit dataset (31 columns)
    │
    └── mitigation_ai/                        # Topological Graph Transformer Module
        ├── gnn.py                            # QEMFormer-GNN architecture & training pipeline
        ├── evaluate_test_shards.py           # Independent shard evaluation & Mitiq comparison
        ├── mlp_mitiq.py                      # Classical ZNE & MLP baseline evaluator
        └── outputs/                          # GNN checkpoints & diagnostic visualizations
            └── gnn/
                ├── best_gnn_model.pth        # Best validation loss GNN checkpoint
                ├── final_gnn_model.pth       # Final 200-epoch GNN checkpoint
                ├── training_log.csv          # Per-epoch GNN training & validation metrics
                ├── training_curves.png       # 2×2 metric convergence grid
                ├── predictions_scatter.png   # Mitigated vs ideal ground-truth scatter
                ├── residual_histogram.png    # Residual error distribution (Δ_true vs Δ̂)
                ├── gamma_histogram.png       # Survival probability distribution
                ├── attention_heatmap.png     # Multi-head attention weight heatmap
                ├── error_vs_depth.png        # Residual error across circuit depth bins
                ├── zne_trajectories.png      # Sample ZNE curves with mitigated predictions
                └── feature_importance.png    # Graph permutation feature importance
```

---

## 11. Installation, Environment & Execution Guide

### 11.1 Dependencies & Hardware Requirements
- **Hardware Recommended:** NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (CUDA 12.1+). CPU-only execution is supported for testing.
- **Python Environment:** Python 3.10 or 3.11.

Set up the environment:

```bash
# 1. Create and activate a conda environment
conda create -n qem_env python=3.10 -y
conda activate qem_env

# 2. Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric dependencies
pip install torch-geometric

# 4. Install Quantum Computing & Scientific Computing libraries
pip install qiskit qiskit-aer mitiq numpy scipy matplotlib pandas scikit-learn
```

---

### 11.2 Training the QEMFormer Model

Train the dual-branch Tabular + Sequence Transformer on the vector dataset:

```bash
python qemformer/qemformer_model.py \
    --epochs 200 \
    --lr 5e-4 \
    --batch 8192 \
    --seed 42 \
    --val-split 0.20 \
    --test-split 0.10 \
    --d-model 128 \
    --nhead 8 \
    --num-layers 8 \
    --dim-ff 256
```

Checkpoints, CSV logs, and all 6 evaluation plots will be saved to `qemformer/outputs/`.

---

### 11.3 Training the QEMFormer-GNN Model

Train the topological graph transformer across the 20 sharded DAGs:

```bash
python zne/mitigation_ai/gnn.py \
    --epochs 200 \
    --lr 3e-4 \
    --batch 256 \
    --num-layers 4 \
    --hidden-dim 128 \
    --heads 4
```

**Fast Smoke Test (1 Shard, 2 Epochs):**
```bash
python zne/mitigation_ai/gnn.py --epochs 2 --max-shards 1 --batch 64
```

Checkpoints, CSV logs, and all 8 diagnostic plots will be saved to `zne/mitigation_ai/outputs/gnn/`.

---

### 11.4 Automated Shard Evaluation Suite

Evaluate a trained checkpoint against held-out test shards (`shard_18.npz`, `shard_19.npz`) and benchmark directly against Mitiq's classical ZNE routines:

```bash
python zne/mitigation_ai/evaluate_test_shards.py \
    --model-path zne/mitigation_ai/outputs/gnn/best_gnn_model.pth \
    --shards 18 19 \
    --batch 256
```

---

## Research Inquiries & Citation

This codebase is part of ongoing scientific research. If you use this code or benchmark data in your research, please reference this repository. For inquiries regarding early access to the 20 pre-compiled dataset shards prior to public Google Drive release, please open an issue in this repository.
