"""
Deep Ensembles — Toy Regression Experiment
===========================================
Reproduction of the 1D toy regression experiment from:
  "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
  Lakshminarayanan et al. (2017), NeurIPS.

This script implements three experimental protocols:
  1. Single NN trained with MSE loss (point prediction, no uncertainty)
  2. Single NN trained with NLL loss (learns mean and variance)
  3. Ensemble of M=5 NNs trained with NLL (well-calibrated uncertainty)

Framework: JAX (following ASI course Tutorial 2)
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_TRAIN = 20          # number of training points
X_RANGE = (-4, 4)     # training x range
X_TEST_RANGE = (-6, 6)  # test x range (wider, to see OOD behavior)
N_TEST = 200          # number of test points
HIDDEN_DIM = 50       # hidden layer size
LR = 0.01             # learning rate
N_EPOCHS = 5000       # training epochs
M_ENSEMBLE = 5        # ensemble size

# Output directory for figures
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Matplotlib style
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ===========================================================================
# 1. Data Generation
# ===========================================================================
def generate_data(key, n=N_TRAIN, x_range=X_RANGE, noise_std=3.0):
    """Generate toy regression data: y = x^3 + epsilon, epsilon ~ N(0, 9)."""
    key_x, key_eps = random.split(key)
    x = random.uniform(key_x, shape=(n, 1), minval=x_range[0], maxval=x_range[1])
    eps = noise_std * random.normal(key_eps, shape=(n, 1))
    y = x ** 3 + eps
    return x, y


# ===========================================================================
# 2. Neural Network (MLP) in JAX
# ===========================================================================
def init_params(key, input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=2):
    """
    Initialize a single-hidden-layer MLP.
    Output dim = 2 for NLL (mu, log_var) or 1 for MSE (mu only).
    Uses He initialization for weights.
    """
    k1, k2 = random.split(key)
    params = {
        'W1': random.normal(k1, (input_dim, hidden_dim)) * jnp.sqrt(2.0 / input_dim),
        'b1': jnp.zeros(hidden_dim),
        'W2': random.normal(k2, (hidden_dim, output_dim)) * jnp.sqrt(2.0 / hidden_dim),
        'b2': jnp.zeros(output_dim),
    }
    return params


def forward(params, x):
    """Forward pass: input -> hidden (ReLU) -> output."""
    h = jnp.maximum(0, x @ params['W1'] + params['b1'])  # ReLU
    out = h @ params['W2'] + params['b2']
    return out


def softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    return jnp.logaddexp(x, 0.0)


def predict_gaussian(params, x):
    """
    Predict mean and variance from network output.
    Returns mu, var (both shape (n, 1)).
    """
    out = forward(params, x)
    mu = out[:, 0:1]
    log_var = out[:, 1:2]
    var = softplus(log_var) + 1e-6  # enforce positivity
    return mu, var


# ===========================================================================
# 3. Loss Functions
# ===========================================================================
def mse_loss(params, x, y):
    """Mean Squared Error loss (Protocol 1)."""
    out = forward(params, x)
    mu = out[:, 0:1]  # only use first output
    return jnp.mean((y - mu) ** 2)


def nll_loss(params, x, y):
    """
    Negative Log-Likelihood loss for Gaussian output (Protocol 2 & 3).
    Eq. (2) from the report:
      L = mean[ 0.5 * log(var) + 0.5 * (y - mu)^2 / var ]
    """
    mu, var = predict_gaussian(params, x)
    return jnp.mean(0.5 * jnp.log(var) + 0.5 * (y - mu) ** 2 / var)


# ===========================================================================
# 4. Adam Optimizer (from scratch, following JAX tutorial style)
# ===========================================================================
def init_adam(params, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8):
    """Initialize Adam optimizer state."""
    state = {
        'm': jax.tree.map(jnp.zeros_like, params),     # first moment
        'v': jax.tree.map(jnp.zeros_like, params),     # second moment
        't': 0,
        'lr': lr,
        'beta1': beta1,
        'beta2': beta2,
        'eps': eps,
    }
    return state


def adam_step(params, grads, state):
    """Perform one Adam update step."""
    t = state['t'] + 1
    lr, beta1, beta2, eps = state['lr'], state['beta1'], state['beta2'], state['eps']

    m = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, state['m'], grads)
    v = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g ** 2, state['v'], grads)

    # Bias correction
    m_hat = jax.tree.map(lambda m: m / (1 - beta1 ** t), m)
    v_hat = jax.tree.map(lambda v: v / (1 - beta2 ** t), v)

    # Update params
    params = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
                          params, m_hat, v_hat)

    state = {**state, 'm': m, 'v': v, 't': t}
    return params, state


# ===========================================================================
# 5. Training Loop
# ===========================================================================
def train(params, x, y, loss_fn, n_epochs=N_EPOCHS, lr=LR, verbose=True):
    """Train a network with the given loss function."""
    opt_state = init_adam(params, lr=lr)
    grad_fn = jit(grad(loss_fn))
    loss_fn_jit = jit(loss_fn)

    losses = []
    for epoch in range(n_epochs):
        grads = grad_fn(params, x, y)
        params, opt_state = adam_step(params, grads, opt_state)

        if epoch % 1000 == 0 or epoch == n_epochs - 1:
            loss_val = float(loss_fn_jit(params, x, y))
            losses.append(loss_val)
            if verbose:
                print(f"  Epoch {epoch:5d}/{n_epochs} | Loss: {loss_val:.4f}")

    return params, losses


# ===========================================================================
# 6. Ensemble Prediction (Eq. 5 from the report)
# ===========================================================================
def ensemble_predict(all_params, x_test):
    """
    Combine M ensemble members using Eq. (5):
      mu* = (1/M) sum_m mu_m
      var* = (1/M) sum_m [var_m + mu_m^2] - mu*^2
    
    Returns mu_star, var_star, aleatoric, epistemic.
    """
    M = len(all_params)
    mus = []
    vars_ = []

    for params in all_params:
        mu, var = predict_gaussian(params, x_test)
        mus.append(mu)
        vars_.append(var)

    mus = jnp.stack(mus, axis=0)       # (M, N, 1)
    vars_ = jnp.stack(vars_, axis=0)   # (M, N, 1)

    # Ensemble mean
    mu_star = jnp.mean(mus, axis=0)    # (N, 1)

    # Total variance (Eq. 5)
    var_star = jnp.mean(vars_ + mus ** 2, axis=0) - mu_star ** 2

    # Decomposition
    aleatoric = jnp.mean(vars_, axis=0)               # average predicted noise
    epistemic = jnp.mean(mus ** 2, axis=0) - mu_star ** 2  # variance of means

    return mu_star, var_star, aleatoric, epistemic


# ===========================================================================
# 7. Visualization
# ===========================================================================
# Color palette
COLOR_DATA = '#2d3436'
COLOR_MEAN = '#0984e3'
COLOR_BAND = '#74b9ff'
COLOR_TRUE = '#636e72'
COLOR_ALEATORIC = '#00b894'
COLOR_EPISTEMIC = '#e17055'


def plot_true_function(ax, x_test):
    """Plot the ground truth y = x^3."""
    ax.plot(x_test, x_test ** 3, '--', color=COLOR_TRUE, alpha=0.5,
            linewidth=1.2, label=r'True $y = x^3$')


def plot_single_mse(ax, params, x_train, y_train, x_test):
    """Plot Protocol 1: Single NN + MSE."""
    out = forward(params, x_test)
    mu = out[:, 0]

    plot_true_function(ax, x_test.squeeze())
    ax.scatter(x_train.squeeze(), y_train.squeeze(), c=COLOR_DATA,
               s=30, zorder=5, label='Training data', edgecolors='white', linewidth=0.5)
    ax.plot(x_test.squeeze(), mu, color=COLOR_MEAN, linewidth=2, label=r'Predicted $\mu(x)$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('(a) Single NN — MSE Loss')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(-200, 200)


def plot_single_nll(ax, params, x_train, y_train, x_test):
    """Plot Protocol 2: Single NN + NLL."""
    mu, var = predict_gaussian(params, x_test)
    mu = mu.squeeze()
    std = jnp.sqrt(var).squeeze()
    x_t = x_test.squeeze()

    plot_true_function(ax, x_t)
    ax.fill_between(x_t, mu - 3 * std, mu + 3 * std, alpha=0.25,
                    color=COLOR_BAND, label=r'$\mu \pm 3\sigma$')
    ax.scatter(x_train.squeeze(), y_train.squeeze(), c=COLOR_DATA,
               s=30, zorder=5, label='Training data', edgecolors='white', linewidth=0.5)
    ax.plot(x_t, mu, color=COLOR_MEAN, linewidth=2, label=r'Predicted $\mu(x)$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('(b) Single NN — NLL Loss')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(-200, 200)


def plot_ensemble(ax, all_params, x_train, y_train, x_test, show_decomposition=False):
    """Plot Protocol 3: Ensemble of 5 NNs + NLL."""
    mu_star, var_star, aleatoric, epistemic = ensemble_predict(all_params, x_test)
    mu = mu_star.squeeze()
    std = jnp.sqrt(var_star).squeeze()
    x_t = x_test.squeeze()

    plot_true_function(ax, x_t)

    if show_decomposition:
        std_alea = jnp.sqrt(aleatoric).squeeze()
        std_epis = jnp.sqrt(epistemic).squeeze()
        ax.fill_between(x_t, mu - 3 * std, mu + 3 * std, alpha=0.15,
                        color=COLOR_BAND, label=r'Total $\mu \pm 3\sigma$')
        ax.fill_between(x_t, mu - 3 * std_alea, mu + 3 * std_alea, alpha=0.25,
                        color=COLOR_ALEATORIC, label='Aleatoric')
        ax.fill_between(x_t, mu - 3 * std_epis, mu + 3 * std_epis, alpha=0.25,
                        color=COLOR_EPISTEMIC, label='Epistemic')
    else:
        ax.fill_between(x_t, mu - 3 * std, mu + 3 * std, alpha=0.25,
                        color=COLOR_BAND, label=r'$\mu \pm 3\sigma$')

    # Plot individual ensemble members (faint)
    for i, params in enumerate(all_params):
        mu_i, _ = predict_gaussian(params, x_test)
        label = 'Individual NNs' if i == 0 else None
        ax.plot(x_t, mu_i.squeeze(), color=COLOR_MEAN, alpha=0.15,
                linewidth=0.8, label=label)

    ax.scatter(x_train.squeeze(), y_train.squeeze(), c=COLOR_DATA,
               s=30, zorder=5, label='Training data', edgecolors='white', linewidth=0.5)
    ax.plot(x_t, mu, color=COLOR_MEAN, linewidth=2, label=r'Ensemble $\mu_*(x)$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('(c) Deep Ensemble (M=5) — NLL Loss')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax.set_ylim(-200, 200)


# ===========================================================================
# 8. Main
# ===========================================================================
def main():
    print("=" * 60)
    print("Deep Ensembles — Toy Regression Experiment")
    print("=" * 60)

    # --- Data ---
    key = random.PRNGKey(SEED)
    key, data_key = random.split(key)
    x_train, y_train = generate_data(data_key)
    x_test = jnp.linspace(X_TEST_RANGE[0], X_TEST_RANGE[1], N_TEST).reshape(-1, 1)

    print(f"\nTraining data: {N_TRAIN} points in [{X_RANGE[0]}, {X_RANGE[1]}]")
    print(f"Test range: [{X_TEST_RANGE[0]}, {X_TEST_RANGE[1]}] ({N_TEST} points)")

    # ---------------------------------------------------------------
    # Protocol 1: Single NN + MSE
    # ---------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Protocol 1: Single NN with MSE Loss")
    print("-" * 60)
    key, init_key = random.split(key)
    params_mse = init_params(init_key, output_dim=2)  # 2 outputs but only use first
    params_mse, losses_mse = train(params_mse, x_train, y_train, mse_loss)

    # ---------------------------------------------------------------
    # Protocol 2: Single NN + NLL
    # ---------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Protocol 2: Single NN with NLL Loss")
    print("-" * 60)
    key, init_key = random.split(key)
    params_nll = init_params(init_key, output_dim=2)
    params_nll, losses_nll = train(params_nll, x_train, y_train, nll_loss)

    # ---------------------------------------------------------------
    # Protocol 3: Ensemble of M=5 NNs + NLL
    # ---------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"Protocol 3: Ensemble of M={M_ENSEMBLE} NNs with NLL Loss")
    print("-" * 60)
    ensemble_params = []
    for m in range(M_ENSEMBLE):
        print(f"\n  --- Ensemble member {m + 1}/{M_ENSEMBLE} ---")
        key, init_key = random.split(key)
        params_m = init_params(init_key, output_dim=2)
        params_m, _ = train(params_m, x_train, y_train, nll_loss)
        ensemble_params.append(params_m)

    # ---------------------------------------------------------------
    # Generate Figures
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)

    # --- Combined 3-panel figure (main figure for the report) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    plot_single_mse(axes[0], params_mse, x_train, y_train, x_test)
    plot_single_nll(axes[1], params_nll, x_train, y_train, x_test)
    plot_ensemble(axes[2], ensemble_params, x_train, y_train, x_test)
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')
    fig.suptitle('Toy Regression: Deep Ensembles vs. Single Networks',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'toy_regression_combined.pdf')
    fig.savefig(FIG_DIR / 'toy_regression_combined.png')
    print(f"  Saved: {FIG_DIR / 'toy_regression_combined.pdf'}")

    # --- Individual figures ---
    # Figure 1: MSE
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    plot_single_mse(ax1, params_mse, x_train, y_train, x_test)
    ax1.set_title('Single NN — MSE Loss')
    plt.tight_layout()
    fig1.savefig(FIG_DIR / 'single_nn_mse.pdf')
    fig1.savefig(FIG_DIR / 'single_nn_mse.png')
    print(f"  Saved: {FIG_DIR / 'single_nn_mse.pdf'}")

    # Figure 2: NLL
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    plot_single_nll(ax2, params_nll, x_train, y_train, x_test)
    ax2.set_title('Single NN — NLL Loss')
    plt.tight_layout()
    fig2.savefig(FIG_DIR / 'single_nn_nll.pdf')
    fig2.savefig(FIG_DIR / 'single_nn_nll.png')
    print(f"  Saved: {FIG_DIR / 'single_nn_nll.pdf'}")

    # Figure 3: Ensemble
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    plot_ensemble(ax3, ensemble_params, x_train, y_train, x_test)
    ax3.set_title(f'Deep Ensemble (M={M_ENSEMBLE}) — NLL Loss')
    plt.tight_layout()
    fig3.savefig(FIG_DIR / 'ensemble_nll.pdf')
    fig3.savefig(FIG_DIR / 'ensemble_nll.png')
    print(f"  Saved: {FIG_DIR / 'ensemble_nll.pdf'}")

    # Figure 4: Ensemble with uncertainty decomposition
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    plot_ensemble(ax4, ensemble_params, x_train, y_train, x_test, show_decomposition=True)
    ax4.set_title(f'Uncertainty Decomposition — Deep Ensemble (M={M_ENSEMBLE})')
    plt.tight_layout()
    fig4.savefig(FIG_DIR / 'uncertainty_decomposition.pdf')
    fig4.savefig(FIG_DIR / 'uncertainty_decomposition.png')
    print(f"  Saved: {FIG_DIR / 'uncertainty_decomposition.pdf'}")

    # --- Loss curves ---
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    epochs_log = list(range(0, N_EPOCHS, 1000)) + [N_EPOCHS - 1]
    ax5.plot(epochs_log, losses_mse, 'o-', label='MSE Loss', color='#e74c3c', markersize=4)
    ax5.plot(epochs_log, losses_nll, 's-', label='NLL Loss', color='#3498db', markersize=4)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title('Training Loss Curves')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    fig5.savefig(FIG_DIR / 'loss_curves.pdf')
    fig5.savefig(FIG_DIR / 'loss_curves.png')
    print(f"  Saved: {FIG_DIR / 'loss_curves.pdf'}")

    plt.close('all')

    print("\n" + "=" * 60)
    print("All done! Figures saved in:", FIG_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
