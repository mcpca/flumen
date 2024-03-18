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


def main():
    data = pandas.read_csv(sys.argv[1])

    nv = data['noise_std']**2
    test_coef = data['test_mse'] / nv

    ax = sns.lineplot(x=data['noise_std'], y=test_coef, label='Test error')
    ax.grid()

    ax.set_ylabel(r"$\frac{\ell_T(\varphi_\sigma)}{\sigma^2}$",
                  usetex=True,
                  fontsize=20)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Standard deviation of the noise ($\sigma$)",
                  fontsize='large')
    ax.get_legend().remove()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
