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

    data = data[(data['encoder_depth'] == 2) & (data['decoder_depth'] == 2)
                & (data['lr'] > 1e-4)]

    fig, ax = plt.subplots()

    sns.boxplot(data=data,
                 x='lr',
                 y='val_mse',
                 hue='control_rnn_size',
                 ax=ax)

    ax.set_ylabel("Validation loss")
    ax.set_xlabel("Learning rate")

    ax.set_yscale('log')
    ax.legend(title="# hidden states")

    fig.tight_layout()
    plt.show()
    # fig.savefig(sys.argv[2])


if __name__ == "__main__":
    main()
