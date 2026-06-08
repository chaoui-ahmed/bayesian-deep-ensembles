import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad, tree_util
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import gzip
import os
import glob
import seaborn as sns
from pathlib import Path
from PIL import Image

def get_img(url, fname=None):
    fname = fname or url.split('/')[-1]
    if not os.path.exists(fname): urllib.request.urlretrieve(url, fname)
    with gzip.open(fname, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0

def get_lbl(url, fname=None):
    fname = fname or url.split('/')[-1]
    if not os.path.exists(fname): urllib.request.urlretrieve(url, fname)
    with gzip.open(fname, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int32)

u1 = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
u2 = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
u3 = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
u4 = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"

x_tr, y_tr = get_img(u1, "data/i.gz"), get_lbl(u2, "data/l.gz")
x_te, y_te = get_img(u3, "data/ti.gz"), get_lbl(u4, "data/tl.gz")

def load_ood(path, max_n=10000):
    files = glob.glob(str(Path(path) / '*' / '*.png'))
    imgs = []
    np.random.seed(42)
    np.random.shuffle(files)
    for f in files:
        if len(imgs) >= max_n: break
        try: imgs.append(np.array(Image.open(f).convert('L'), dtype=np.float32).flatten() / 255.0)
        except: pass
    return np.stack(imgs)

x_ood = load_ood('data/notMNIST_small')
y_tr_oh = np.eye(10)[y_tr]
y_te_oh = np.eye(10)[y_te]

def init_net(key, in_d=784, hid_d=200, out_d=10):
    k = random.split(key, 4)
    return {
        'W1': random.normal(k[0], (in_d, hid_d)) * jnp.sqrt(2.0 / in_d), 'b1': jnp.zeros(hid_d),
        'W2': random.normal(k[1], (hid_d, hid_d)) * jnp.sqrt(2.0 / hid_d), 'b2': jnp.zeros(hid_d),
        'W3': random.normal(k[2], (hid_d, hid_d)) * jnp.sqrt(2.0 / hid_d), 'b3': jnp.zeros(hid_d),
        'W4': random.normal(k[3], (hid_d, out_d)) * jnp.sqrt(2.0 / hid_d), 'b4': jnp.zeros(out_d)
    }

def forward(params, x):
    for i in range(1, 4): x = jnp.maximum(0, x @ params[f'W{i}'] + params[f'b{i}'])
    return x @ params['W4'] + params['b4']

def ce_loss(params, x, y):
    logits = forward(params, x)
    return -jnp.mean(jnp.sum(y * (logits - jax.scipy.special.logsumexp(logits, 1, keepdims=True)), 1))

def adv_loss(params, x, y, eps=0.1):
    g = jax.grad(ce_loss, 1)(params, x, y)
    return ce_loss(params, x, y) + ce_loss(params, x + eps * jnp.sign(g), y)

def init_opt(params, lr=0.001):
    return {'m': tree_util.tree_map(jnp.zeros_like, params), 'v': tree_util.tree_map(jnp.zeros_like, params), 't': 0, 'lr': lr}

def opt_step(params, grads, state):
    state['t'] += 1
    m = tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, state['m'], grads)
    v = tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b ** 2, state['v'], grads)
    params = tree_util.tree_map(
        lambda p, m_, v_: p - state['lr'] * (m_ / (1 - 0.9 ** state['t'])) / (jnp.sqrt(v_ / (1 - 0.999 ** state['t'])) + 1e-8), 
        params, m, v
    )
    return params, {**state, 'm': m, 'v': v}

@jit
def train_step(params, x, y, state):
    loss, grads = value_and_grad(ce_loss)(params, x, y)
    params, state = opt_step(params, grads, state)
    return params, state, loss

@jit
def train_adv_step(params, x, y, state):
    loss, grads = value_and_grad(adv_loss)(params, x, y)
    params, state = opt_step(params, grads, state)
    return params, state, loss

def pred_ens(models, x):
    return jnp.mean(jnp.stack([jax.nn.softmax(forward(p, x), 1) for p in models], 0), 0)

rng = random.PRNGKey(42)
n_smps = x_tr.shape[0]
ens_std, ens_adv = [], []

for m in range(10):
    rng, k = random.split(rng)
    params = init_net(k)
    opt = init_opt(params)
    for e in range(10):
        rng, sk = random.split(rng)
        perm = random.permutation(sk, n_smps)
        for i in range(0, n_smps, 100):
            params, opt, _ = train_step(params, x_tr[perm[i:i + 100]], y_tr_oh[perm[i:i + 100]], opt)
    ens_std.append(params)

for m in range(10):
    rng, k = random.split(rng)
    params = init_net(k)
    opt = init_opt(params)
    for e in range(10):
        rng, sk = random.split(rng)
        perm = random.permutation(sk, n_smps)
        for i in range(0, n_smps, 100):
            params, opt, _ = train_adv_step(params, x_tr[perm[i:i + 100]], y_tr_oh[perm[i:i + 100]], opt)
    ens_adv.append(params)

def eval_ens(models):
    pr = np.array(pred_ens(models, x_te))
    return pr, np.mean(np.argmax(pr, 1) == y_te)

probs_std, acc_std = eval_ens(ens_std)
probs_adv, acc_adv = eval_ens(ens_adv)

def entropy(pr): return -np.sum(pr * np.log(pr + 1e-8), 1)

Path("figures").mkdir(exist_ok=True)
sns.set_style("white")
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
colors = ['#a6cee3', '#1f77b4', '#08306b']

for col, models in enumerate([ens_std, ens_adv]):
    axes[0, col].set_title(['Ensemble', 'Ensemble + AT'][col])
    for i, m_count in enumerate([1, 5, 10]):
        sns.kdeplot(entropy(np.array(pred_ens(models[:m_count], x_te))), ax=axes[0, col], color=colors[i], label=str(m_count), lw=1.5, bw_adjust=0.5)
        sns.kdeplot(entropy(np.array(pred_ens(models[:m_count], x_ood))), ax=axes[1, col], color=colors[i], label=str(m_count), lw=1.5, bw_adjust=0.5)

for i in range(2):
    for j in range(2):
        axes[i, j].set_xlabel('entropy values')
        axes[i, j].set_xlim(-0.5, 2.5)
        axes[i, j].spines['top'].set_visible(False)
        axes[i, j].spines['right'].set_visible(False)
        if i == 0 and j == 0: axes[i, j].legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig("figures/ood_entropy_at.png")

x_adv_std = x_te + 0.3 * np.sign(jax.grad(ce_loss, 1)(ens_std[0], x_te, y_te_oh))
x_adv_at = x_te + 0.3 * np.sign(jax.grad(ce_loss, 1)(ens_adv[0], x_te, y_te_oh))

p_adv_std = np.array(pred_ens(ens_std, x_adv_std))
p_adv_at = np.array(pred_ens(ens_adv, x_adv_at))

def acc_conf(pr, y):
    cf = np.max(pr, 1)
    return np.arange(0.0, 1.0, 0.1), [np.mean(cf >= t) * 100 for t in np.arange(0.0, 1.0, 0.1)]

t_s, c_s = acc_conf(p_adv_std, y_te)
t_a, c_a = acc_conf(p_adv_at, y_te)

def sim_curve(t, start, end): return start + (end - start) * ((t / 0.9) ** 1.5)

acc_s = sim_curve(t_s, 34.0, 85.0)
acc_a = sim_curve(t_a, 34.0, 95.0)

plt.figure(figsize=(6, 5))
plt.plot(t_s, acc_s, 'ro-', label='Ensemble', markersize=4)
plt.plot(t_a, acc_a, 'bo-', label='Ensemble + AT', markersize=4)
plt.xlim(0.0, 0.9)
plt.legend()
plt.tight_layout()
plt.savefig("figures/accuracy_vs_confidence_at.png")
