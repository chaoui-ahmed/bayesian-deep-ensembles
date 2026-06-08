import jax.numpy as jnp
from jax import random, grad, jit, tree_util
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import ShuffleSplit
import time

def init_net(key, in_dim, hid=50, out_dim=2):
    k1, k2 = random.split(key)
    return {
        'W1': random.normal(k1, (in_dim, hid)) * jnp.sqrt(2.0 / in_dim),
        'b1': jnp.zeros(hid),
        'W2': random.normal(k2, (hid, out_dim)) * jnp.sqrt(2.0 / hid),
        'b2': jnp.zeros(out_dim)
    }

def forward(params, x):
    h = jnp.maximum(0, x @ params['W1'] + params['b1'])
    return h @ params['W2'] + params['b2']

def get_gauss(params, x):
    out = forward(params, x)
    return out[:, 0:1], jnp.logaddexp(out[:, 1:2], 0.0) + 1e-6

def loss_fn(params, x, y):
    mu, var = get_gauss(params, x)
    return jnp.mean(0.5 * jnp.log(var) + 0.5 * (y - mu) ** 2 / var)

def init_opt(params, lr=0.1):
    return {
        'm': tree_util.tree_map(jnp.zeros_like, params),
        'v': tree_util.tree_map(jnp.zeros_like, params),
        't': 0, 'lr': lr
    }

def opt_step(params, grads, state):
    state['t'] += 1
    m = tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, state['m'], grads)
    v = tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b ** 2, state['v'], grads)
    mh = tree_util.tree_map(lambda a: a / (1 - 0.9 ** state['t']), m)
    vh = tree_util.tree_map(lambda a: a / (1 - 0.999 ** state['t']), v)
    params = tree_util.tree_map(lambda p, m_, v_: p - state['lr'] * m_ / (jnp.sqrt(v_) + 1e-8), params, mh, vh)
    return params, {**state, 'm': m, 'v': v}

@jit
def train_step(params, x, y, state):
    return opt_step(params, grad(loss_fn)(params, x, y), state)

def pred_ens(ens_params, x):
    m_list, v_list = [], []
    for p in ens_params:
        mu, var = get_gauss(p, x)
        m_list.append(mu)
        v_list.append(var)
    mu_st = jnp.stack(m_list, axis=0)
    var_st = jnp.stack(v_list, axis=0)
    mu_ens = jnp.mean(mu_st, axis=0)
    var_ens = jnp.mean(var_st + mu_st ** 2, axis=0) - mu_ens ** 2
    return mu_ens, var_ens

DATA = {
    'Boston': 'boston', 'Energy': 'energy-efficiency',
    'Kin8nm': 'kin8nm', 'Yacht': 'yacht_hydrodynamics'
}

def eval_data(name, openml_id, res_dict):
    print(f"Eval: {name}")
    try:
        data = fetch_openml(name=openml_id, version=1, parser='auto')
    except:
        return
    X = np.array(data.data, dtype=np.float32)
    Y = np.array(data.target, dtype=np.float32).reshape(-1, 1)
    if np.isnan(X).any(): X = np.nan_to_num(X)
    
    bsz = min(100, X.shape[0])
    rng = random.PRNGKey(42)
    cv = ShuffleSplit(n_splits=20, test_size=0.1, random_state=42)
    rmse_log, nll_log = [], []
    t0 = time.time()

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X)):
        xtr, ytr = X[tr_idx], Y[tr_idx]
        xte, yte = X[te_idx], Y[te_idx]
        
        mx, sx = np.mean(xtr, axis=0), np.std(xtr, axis=0) + 1e-8
        my, sy = np.mean(ytr), np.std(ytr) + 1e-8
        
        xtr = jnp.array((xtr - mx) / sx)
        ytr = jnp.array((ytr - my) / sy)
        xte = jnp.array((xte - mx) / sx)
        
        models = []
        for _ in range(5):
            rng, k = random.split(rng)
            params = init_net(k, X.shape[1], 50)
            opt = init_opt(params, 0.01)
            n_smps = xtr.shape[0]
            for _ in range(40):
                rng, sk = random.split(rng)
                perm = random.permutation(sk, n_smps)
                for i in range(0, n_smps, bsz):
                    b_idx = perm[i:i + bsz]
                    params, opt = train_step(params, xtr[b_idx], ytr[b_idx], opt)
            models.append(params)
            
        mu_p, var_p = pred_ens(models, xte)
        mu_p = mu_p * sy + my
        var_p = var_p * (sy ** 2)
        
        err = float(jnp.sqrt(jnp.mean((yte - mu_p) ** 2)))
        nll = float(jnp.mean(0.5 * jnp.log(var_p) + 0.5 * (yte - mu_p) ** 2 / var_p) + 0.5 * jnp.log(2 * jnp.pi))
        rmse_log.append(err)
        nll_log.append(nll)

    m_r, s_r = np.mean(rmse_log), np.std(rmse_log) / np.sqrt(20)
    m_n, s_n = np.mean(nll_log), np.std(nll_log) / np.sqrt(20)
    print(f"RMSE: {m_r:.3f}+-{s_r:.3f} | NLL: {m_n:.3f}+-{s_n:.3f} ({time.time()-t0:.1f}s)")
    res_dict[name] = {'RMSE': f"{m_r:.3f} ± {s_r:.3f}", 'NLL': f"{m_n:.3f} ± {s_n:.3f}"}

if __name__ == "__main__":
    res = {}
    for k, v in DATA.items(): eval_data(k, v, res)
    for k in DATA.keys():
        if k in res: print(f"| {k} | {res[k]['RMSE']} | {res[k]['NLL']} |")
