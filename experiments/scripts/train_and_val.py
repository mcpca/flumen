import pandas
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sys

# font = {'size': 22}
# matplotlib.rc('font', **font)


def main():
    data = pandas.read_csv(sys.argv[1])

    fig, ax = plt.subplots()

    sns.scatterplot(data=data,
                    x='train_mse',
                    y='val_mse',
                    hue='batch_size',
                    ax=ax)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('equal')

    low_lim = min(np.min(data['train_mse']), np.min(data['val_mse']))
    low_lim -= np.abs(low_lim) / 10

    ax.set_xlim(low_lim)
    ax.set_ylim(low_lim)

    # fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
