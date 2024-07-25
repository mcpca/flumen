import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re
import seaborn as sns

p = Path('./outputs')

files = list(p.glob('vdp_T15_test_data_2024*.sh/*.csv'))

means = []

for f in files:
    pattern = re.compile('vdp_T15_test_data_2024_(\d+)_(\d+)-S*')
    matches = pattern.search(f.name)

    loss = pd.read_csv(
        f)['test_mse_outputs/vdp_T15_test_data_2024_400_800_large.pth'].min()
    n_trajectories = int(matches.group(1))
    n_samples = int(matches.group(2))

    means.append({
        'n_trajectories': n_trajectories,
        'n_samples': n_samples,
        'test_mse': n_trajectories * n_samples
    })

means = pd.DataFrame(means)
means = means.pivot(index='n_trajectories',
                    columns='n_samples',
                    values='test_mse')

ax = sns.heatmap(means,
                 cmap='viridis',
                 annot=True,
                 cbar_kws={
                     'label': 'Test loss',
                 })
ax.set_xlabel('# samples per trajectory', fontsize='large')
ax.set_ylabel('# trajectories', fontsize='large')

plt.tight_layout()
plt.show()
