'''
    Plot validation MSE and training time, for experiments where a single
    parameter is varied.

    To run:
        python plot_stats.py [path to .csv training stats] [parameter]
'''

import pandas
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import sys

font = {'size': 30}
matplotlib.rc('font', **font)


def main():
    data = pandas.read_csv(sys.argv[1])

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=False)
    fig.set_size_inches((15, 9))

    nv = data['noise_std'] ** 2
    test_coef = data['test_mse'] / nv

    sns.lineplot(x=data['noise_std'], y=test_coef, ax=ax,
                 label='Test error')
    ax.set_ylabel(r"$\frac{\ell_T(\varphi_\sigma)}{\sigma^2}$")
    # ax.set_ylim(0.0, 0.5)
    ax.set_xlabel(r"Standard deviation of the noise ($\sigma$)")
    ax.get_legend().remove()

    fig.tight_layout()
    fig.savefig(sys.argv[1] + ".pdf")


if __name__ == "__main__":
    main()
