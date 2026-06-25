# QCCA: Robust Federated Aggregation Defense Based on Multi-metric Consistency for Heterogeneous Data

Official implementation of **QCCA** (Quad-metric Consistency-based Clustering and Adaptive Aggregation), a server-side robust aggregation framework that defends Federated Learning (FL) against backdoor attacks under strong data heterogeneity (non-IID conditions).

This codebase is built on top of and extends **AlignIns** (Xu et al., *Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection*, CVPR 2025). We thank the AlignIns authors for their excellent open-source release.

## Method

QCCA decouples statistical deviation from malicious behavior through three sequential stages:

1. **Quad-Metric Consistency Features (QMCF)** — four orthogonal metrics computed for each client update: Temporal Direction Alignment (TDA), Group Center Consistency (MeanCos), Masked Principal Sign Alignment (MPSA), and Update Scale Intensity (GradNorm). All four metrics are robustly standardized using a Median-based Z-score (MZ-score).
2. **Cluster-level Behavioral Trial Evaluation** — client updates are partitioned via K-means (K=3); each cluster's representative update is temporarily applied to the global model and evaluated on small server-side auxiliary datasets (clean and backdoor-triggered) to obtain a behavioral risk score.
3. **Gating + Dynamic Softmax Weighting** — high-risk clusters are continuously suppressed via temperature-scaled Softmax weighting rather than hard rejection, followed by adaptive norm clipping and weighted aggregation.

Under strong non-IID conditions (Dirichlet α = 0.3, CIFAR-10/ResNet-9), QCCA reduces the average Attack Success Rate (ASR) from 51.25% (FedAvg) to 10.51%, while maintaining a Clean ACC comparable to the undefended baseline.

### Method-to-code mapping

| Paper | Implementation |
|---|---|
| AlignIns baseline | `origin_alignins` |
| QCCA (Section 3 of the paper) | `origin_alignins_clustering_weighted3_dynamic_softmax_gated` |

See `src/federated.py` for implementation details and additional configuration options.

## Hardware Requirements

- Reproducing the key experiments reported in the paper requires an **NVIDIA RTX 2080Ti (single GPU)**. Results obtained on other GPUs may show small numerical fluctuations due to differences in floating-point implementations across hardware, and may not exactly match the numbers reported in the paper.
- If a 2080Ti is not available locally, it can be rented from cloud GPU platforms such as AutoDL or Featurize.

## Environment Setup

Two equivalent environment specifications are provided:

- `env.yaml` — environment with pinned conda build hashes (recommended for exact reproduction).
- `fl_env.yml` — same set of dependencies, configured to use Tsinghua University mirror channels (recommended for users in mainland China with slow access to the default conda/PyPI servers).

Create and activate the environment:

```bash
conda env create -f env.yaml
conda activate fl
```

To install the environment at a custom path, e.g. on a rented GPU instance (such as Featurize):

```bash
conda env create -f env.yaml -p /path/to/my_fl_env
conda activate /path/to/my_fl_env
```

Core dependencies: Python 3.9.5, PyTorch 1.8.0 (CUDA 11.1.1). All remaining dependencies are listed in `env.yaml` / `fl_env.yml`.

## Datasets

- **CIFAR-10 / CIFAR-100**: downloaded automatically via `torchvision` on first run.

If automatic download fails (e.g. due to network restrictions), you can manually download the datasets from their official pages:

| Dataset | Official Download Page |
|---|---|
| CIFAR-10 | https://www.cs.toronto.edu/~kriz/cifar.html |
| CIFAR-100 | https://www.cs.toronto.edu/~kriz/cifar.html |

Download the Python version of each archive (`cifar-10-python.tar.gz` / `cifar-100-python.tar.gz`), extract it, and place the resulting folder under your torchvision data root (default: `~/data/`). torchvision will detect the extracted files automatically and skip the download step.

## Key Configuration Parameters

The following parameters control the QCCA defense and can be inspected or modified in `src/federated.py`:

| Parameter | Description |
|---|---|
| `cluster_imbalance_ratio` | Cluster imbalance ratio used during K-means partitioning |
| `suspicious_weight` / `benign_weight` | Aggregation weights assigned to suspicious / benign clients |
| `gated2_bd_delta_max` | Maximum allowed change in backdoor accuracy for the gating mechanism |
| `strict_factor` | Strictness factor for the gating threshold |
| `cluster_mz_metrics` | Set of MZ-score-standardized QMCF metrics used for clustering |

## Quick Start

An example launch script is provided, e.g. `fl_code_a/run_non_iid_0.3.sh` (30 clients, 8 malicious clients, non-IID Dirichlet α = 0.3).

1. Activate the environment (adjust the path to match your own setup):

```bash
conda activate /home/featurize/work/fl_code_a/my_fl_env
```

2. Make the script executable and launch it in the background, redirecting logs to a file:

```bash
chmod +x run_non_iid_0.3.sh
nohup ./run_non_iid_0.3.sh > run_0.3.out 2>&1 &
```

3. To stop a running experiment:

```bash
pkill -f federated.py
```

## Acknowledgements

This framework is built upon and extends [AlignIns](https://github.com/JiiahaoXU/AlignIns) ([arXiv:2503.07978](https://arxiv.org/abs/2503.07978)). We thank the authors for their excellent open-source contribution.

## Citation

The paper describing QCCA is currently under review. A BibTeX citation entry will be added here once the venue and DOI have been finalized.
