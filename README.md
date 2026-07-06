# Deep Ensembles for Predictive Uncertainty

A reproducibility-oriented academic project exploring predictive uncertainty with deep ensembles on regression, classification and out-of-distribution evaluation tasks.

> **Project status:** academic research project. Claims should be interpreted from the committed report and reproduced experiments, not from the README alone.

## Scope

The repository studies how independently initialized neural networks can improve predictive uncertainty estimates compared with a single-model baseline. The implementation includes regression experiments and MNIST/notMNIST classification and OOD evaluation.

## What is included

```text
├── code/
│   ├── uci_regression.py
│   ├── plot_uci_boxplots.py
│   └── mnist_ensemble.py
├── figures/
├── report.pdf
├── report.tex
└── requirements.txt
```

Course slides, past exams, revision sheets and unrelated teaching material should not be stored in this public project repository.

## Setup

```bash
git clone https://github.com/chaoui-ahmed/bayesian-deep-ensembles.git
cd bayesian-deep-ensembles
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python code/uci_regression.py
python code/plot_uci_boxplots.py
python code/mnist_ensemble.py
```

## Results policy

The README intentionally avoids unsupported statements such as “significantly outperforms” unless the following are visible and reproducible:

- exact dataset and split;
- number of ensemble members;
- random seeds;
- accuracy or RMSE;
- negative log-likelihood;
- calibration metric;
- OOD metric such as AUROC;
- baseline comparison;
- hardware and runtime.

Add a compact results table here once those values are extracted from the final report or regenerated from code.

## Limitations

- Results depend on architecture, optimization and ensemble size.
- OOD performance depends heavily on the chosen in- and out-of-distribution datasets.
- Academic experiments are not automatically production-ready uncertainty systems.

## Author

**Ahmed Taha Chaoui** — Engineering student at EURECOM, focused on data science and machine learning.
