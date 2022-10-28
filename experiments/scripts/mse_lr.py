import pandas
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sys

font = {'size': 22}
matplotlib.rc('font', **font)


def main():
    data = pandas.read_csv(sys.argv[1])

    control_rnn_size = np.unique(np.sort(data['control_rnn_size'].to_numpy()))

    fig, axs = plt.subplots(2,
                            int(np.ceil(len(control_rnn_size) / 2)),
                            sharex=True,
                            sharey=True)

    fig.set_size_inches((20, 15))

    for idx, crs in enumerate(control_rnn_size):
        ax = axs.reshape(-1)[idx]
        crs = control_rnn_size[idx]

        sns.lineplot(data=data[data['control_rnn_size'] == crs],
                     x='lr',
                     y='val_mse',
                     hue='encoder_size',
                     ax=ax)

        # ax.set_ylim(1e-3, 1e-2)
        ax.set_title("$n_{LSTM} = "
                     f"{crs}$")
        ax.set_ylabel("Validation loss")
        ax.set_xlabel("Learning rate")

        if idx != 0:
            ax.legend().remove()

    first_ax = axs.reshape(-1)[0]
    first_ax.set_xscale('log')
    first_ax.set_yscale('log')
    first_ax.legend(title="$n_e / dim(Z)$")

    fig.tight_layout()
    fig.savefig(sys.argv[2])


if __name__ == "__main__":
    main()
