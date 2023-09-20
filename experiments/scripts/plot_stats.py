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
    outputs = list(sys.argv[3:])

    fig, axs = plt.subplots(len(outputs), 1,
                            sharex=True,
                            sharey=False)

    if len(outputs) == 1:
        axs = [axs]

    fig.set_size_inches((15, len(outputs)*9))

    y_min = data[outputs].min().min()
    y_max = data[outputs].max().max()

    for (k, output) in enumerate(outputs):
        sns.lineplot(data=data,
                     x=param,
                     y=output,
                     ax=axs[k])

        axs[k].set_ylabel(output)
        axs[k].set_ylim(y_min, y_max)
        # axs[k].set_yscale('log')

    axs[-1].set_xlabel(param)

    fig.tight_layout()
    plt.show()
    # fig.savefig(sys.argv[1] + ".pdf")


if __name__ == "__main__":
    main()
