# HAD-MSF: Multi-domain Neural Fusion with State-Space and Graph Modeling for Hyperspectral Anomaly Detection

[![Paper](https://img.shields.io/badge/Paper-Neural%20Networks-blue)](https://www.sciencedirect.com/science/article/pii/S0893608026011299)

**Official code repository** for the journal paper:

**HAD-MSF: Multi-domain neural fusion with state-space and graph modeling for hyperspectral anomaly detection**  
*Neural Networks* (2026).  
📄 [Paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608026011299)

This repository makes HAD-MSF available to support reproducible research in **hyperspectral anomaly detection (HAD)**.

## Overview

HAD-MSF combines state-space modeling and graph-based feature learning to capture complementary information for hyperspectral anomaly detection. The implementation contains modules for spectral band selection, multi-domain feature extraction, separation training, and anomaly-map evaluation.

> **Current release scope:** The available training pipeline supports reproduction on the **Pavia** dataset. It does not currently provide a one-command reproduction of all experiments in the article.

## Repository Structure

```text
HAD-MSF/
├── main.py              # Pavia experiment runner
├── model.py             # HAD-MSF model
├── select_bands.py      # SIFDR-inspired band selection
├── evaluation.py        # Additional evaluation protocols
├── metric.py            # Metric utilities
├── utils.py             # Normalization and utilities
├── requirements.txt     # Python dependencies
├── dataset/
│   ├── loader.py        # Dataset loader
│   └── Pavia/
│       └── Pavia.mat    # Pavia dataset
├── SeT/                 # Separation-training components
└── vmunet/              # State-space/model components
```

## Installation

**Requirements:** Python 3.10–3.12, a CUDA-enabled NVIDIA GPU, and compatible PyTorch/CUDA components. The runner requires the **official `mamba-ssm`** implementation.

```bash
git clone https://github.com/JzXu123/HAD-MSF.git
cd HAD-MSF

conda create -n had-msf python=3.10 -y
conda activate had-msf
```

Install a CUDA-enabled PyTorch build following the [official installation instructions](https://pytorch.org/get-started/locally/), then install the dependencies:

```bash
pip install -r requirements.txt
```

If the Mamba CUDA extensions require separate installation or rebuilding:

```bash
pip install mamba-ssm causal-conv1d --no-build-isolation
```

> CUDA extension compatibility depends on the local PyTorch, CUDA, and compiler versions. CPU-only execution is not supported by the current `main.py`.

## Dataset

The Pavia dataset is included at:

```text
dataset/Pavia/Pavia.mat
```

The MATLAB file must contain a three-dimensional hyperspectral cube and a corresponding two-dimensional anomaly ground-truth map. **Ground-truth labels are used for evaluation, not supervised model training.**

## Reproduction

Run the Pavia experiment with the default configuration:

```bash
python main.py
```

Or specify the main hyperparameters explicitly:

```bash
python main.py \
  --data-root dataset/Pavia/Pavia.mat \
  --num-bs 64 \
  --epochs 30 \
  --max-iter 5 \
  --lr 1e-4 \
  --lmda 1e-3 \
  --vss-depth 3 \
  --state-dim 4 \
  --seed 42 \
  --device cuda:0 \
  --output-root results_publishable
```

The program carries out SIFDR-based band selection, Z-score normalization, separation training, anomaly-map construction, and quantitative evaluation. Use `python main.py --help` for all available command-line options.

### Default parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--num-bs` | `64` | Number of selected bands |
| `--epochs` | `30` | Epochs per separation-training iteration |
| `--max-iter` | `5` | Training iterations |
| `--lr` | `1e-4` | Learning rate |
| `--lmda` | `1e-3` | Loss weighting coefficient |
| `--vss-depth` | `3` | State-space model depth |
| `--state-dim` | `4` | State dimension |
| `--seed` | `42` | Random seed |
| `--device` | `cuda:0` | GPU device |

## Outputs and Evaluation

The default output directory is `results_publishable/Pavia/`. It contains a `run_config.json` with the experiment configuration and selected band indices, alongside a model-specific output folder containing:

- `Pavia_result.mat`: model output, anomaly detection map, ground truth, and evaluation arrays.
- `Pavia_metrics.json`: AUC evaluation metrics.
- `Pavia_curves.png`: detection-map and evaluation plots (if Matplotlib is available).

The script also prints **AUC(Pd, Pf)**. Please report the configuration, random seed, dataset, software/hardware environment, and evaluation protocol when comparing results.

## Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@article{xu2026hadmsf,
  title   = {HAD-MSF: Multi-domain neural fusion with state-space and graph modeling for hyperspectral anomaly detection},
  author  = {Xu, Jinzhuang and Zhang, Chenglong and Wang, Xiaoxue and Yang, Xiaoli and Han, Yuping and Ma, Xiaopeng},
  journal = {Neural Networks},
  volume  = {206},
  pages   = {109674},
  year    = {2026},
  doi     = {10.1016/j.neunet.2026.109674},
  url     = {https://www.sciencedirect.com/science/article/pii/S0893608026011299}
}
```

## Acknowledgements

We sincerely thank the authors and contributors of **MSNet** and **SIFDR** for their valuable research and open-source contributions to the hyperspectral imaging community. Their work has provided important inspiration and reference implementations for reproducible hyperspectral anomaly detection research.

**MSNet — Self-Supervised Multiscale Network With Enhanced Separation Training for Hyperspectral Anomaly Detection**

H. Liu, X. Su, X. Shen, and X. Zhou, “MSNet: Self-Supervised Multiscale Network With Enhanced Separation Training for Hyperspectral Anomaly Detection,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 62, pp. 1–13, Art. no. 5520313, 2024. [DOI: 10.1109/TGRS.2024.3411009](https://doi.org/10.1109/TGRS.2024.3411009).

```bibtex
@article{liu2024msnet,
  title   = {MSNet: Self-Supervised Multiscale Network With Enhanced Separation Training for Hyperspectral Anomaly Detection},
  author  = {Liu, H. and Su, X. and Shen, X. and Zhou, X.},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {62},
  pages   = {1--13},
  year    = {2024},
  note    = {Art. no. 5520313},
  doi     = {10.1109/TGRS.2024.3411009}
}
```

**SIFDR — A Real-Time Unsupervised Hyperspectral Band Selection via Spatial-Spectral Information Fusion-Based Downscaled Region**

C. Zhang, L. Mou, X. Yang, X. Zheng, X. Xiang Zhu, and X. Ma, “A Real-Time Unsupervised Hyperspectral Band Selection via Spatial-Spectral Information Fusion-Based Downscaled Region,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 62, pp. 1–14, Art. no. 5534514, 2024. [DOI: 10.1109/TGRS.2024.3474976](https://doi.org/10.1109/TGRS.2024.3474976).

```bibtex
@article{zhang2024sifdr,
  title   = {A Real-Time Unsupervised Hyperspectral Band Selection via Spatial-Spectral Information Fusion-Based Downscaled Region},
  author  = {Zhang, C. and Mou, L. and Yang, X. and Zheng, X. and Zhu, X. Xiang and Ma, X.},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {62},
  pages   = {1--14},
  year    = {2024},
  note    = {Art. no. 5534514},
  doi     = {10.1109/TGRS.2024.3474976}
}
```

We specifically acknowledge the **MSNet team** for its open research on self-supervised enhanced separation training and the **SIFDR team** for its contributions to spatial-spectral hyperspectral band selection. We encourage users to cite these original works when building upon their ideas or implementations.

## Questions and Feedback

For questions about reproducing the experiments or reporting implementation issues, please [open an issue](https://github.com/JzXu123/HAD-MSF/issues).
