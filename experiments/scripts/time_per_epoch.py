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

font = {'size': 22}
matplotlib.rc('font', **font)


def main():
    data = pandas.read_csv(sys.argv[1])
    param = sys.argv[2]

    fig, ax = plt.subplots()

    data['time_per_epoch'] = data['train_time'] / data['n_epochs']

    sns.lineplot(data,
                 x=param,
                 y='time_per_epoch',
                 ax=ax)

    ax.set_yscale('log')
    ax.set_xlabel(param)

    fig.tight_layout()
    plt.show()
    # fig.savefig(sys.argv[1] + ".pdf")


if __name__ == "__main__":
    main()
