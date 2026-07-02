# 🔬 Bayesian Machine Learning & Deep Ensembles Research

[![Course](https://img.shields.io/badge/Course-Eurecom_ASI-blue.svg)](https://www.eurecom.fr)
[![Language](https://img.shields.io/badge/Language-Python_/_LaTeX-green.svg)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch_/_JAX-orange.svg)]()

This repository contains the research paper, implementation code, and course materials for the **ASI (Algorithmic Statistics for Internet)** course at **Eurecom** (Sophia-Antipolis, France). The project focuses on **Uncertainty Quantification in Deep Learning using Deep Ensembles**.

---

## 📖 Research Overview
Deep Neural Networks (DNNs) are notoriously overconfident, even when making incorrect predictions. This research investigates **Deep Ensembles** as a practical and robust method for uncertainty estimation, comparing them against traditional Bayesian approaches (like Monte Carlo Dropout and Laplace Approximation) on regression and classification tasks (MNIST & Out-of-Distribution notMNIST).

### Key Highlights
- **Methodology**: Training multiple neural networks with different random initializations to capture epistemic uncertainty.
- **Evaluation**: Testing model calibration under dataset shift and out-of-distribution (OOD) scenarios using MNIST vs. notMNIST.
- **Results**: Deep Ensembles significantly outperform single-model baselines in calibration, accuracy, and robust OOD detection.

---

## 📁 Repository Structure

```
├── main.pdf                         # Main research paper
├── titre.pdf                        # Title page / Cover page
├── report.pdf / report.tex          # Project research report
├── ProjMan_Part3_RevisionSheet.pdf  # Project management revision sheet
│
├── code/                            # Implementation source code
│   ├── uci_regression.py            # UCI dataset regression experiments
│   ├── plot_uci_boxplots.py         # Visualizing regression metrics
│   └── mnist_ensemble.py            # Deep Ensembles on MNIST / OOD notMNIST
│
├── figures/                         # Generated plots and visualizations
│   └── data/                        # MNIST & notMNIST evaluation datasets
│
└── course/                          # Complete Eurecom ASI Course Materials
    ├── slides/                      # Weekly course slides (Probability, Regression, GPs, MCMC, VI, etc.)
    ├── exercises/                   # Weekly exercises and worksheets
    ├── labs/                        # Jupyter notebooks for hands-on labs
    ├── tutorials/                   # JAX and Python tutorials
    └── exam_2025.pdf                # Past exam paper
```

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed with the following packages:
```bash
pip install torch torchvision numpy matplotlib scipy jax jaxlib
```

### Running the Experiments

1. **UCI Regression Benchmarks**:
   ```bash
   python code/uci_regression.py
   python code/plot_uci_boxplots.py
   ```

2. **MNIST Ensemble & OOD Analysis**:
   ```bash
   python code/mnist_ensemble.py
   ```

---

## ✍️ Author
- **Ahmed Chaoui** — Engineering Student at Eurecom (Sophia-Antipolis, France)
