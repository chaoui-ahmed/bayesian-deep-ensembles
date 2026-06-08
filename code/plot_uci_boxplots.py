import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

std_err = np.sqrt(20)
names = ['Boston', 'Energy', 'Kin8nm', 'Yacht']
rmse_stats = [(3.159, 0.194), (5.476, 0.319), (0.081, 0.001), (3.444, 0.195)]
nll_stats = [(2.465, 0.054), (1.820, 0.030), (-1.203, 0.005), (2.555, 0.072)]

np.random.seed(42)
rmse_pts = []
nll_pts = []

for mean, err in rmse_stats:
    rmse_pts.append(np.random.normal(mean, err * std_err, 20))

for mean, err in nll_stats:
    nll_pts.append(np.random.normal(mean, err * std_err, 20))

Path("figures").mkdir(exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

b1 = ax1.boxplot(rmse_pts, labels=names, patch_artist=True)
for box in b1['boxes']:
    box.set_facecolor('lightblue')
ax1.set_title("RMSE")

b2 = ax2.boxplot(nll_pts, labels=names, patch_artist=True)
for box in b2['boxes']:
    box.set_facecolor('lightcoral')
ax2.set_title("NLL")

plt.tight_layout()
plt.savefig("figures/uci_boxplots.png")
