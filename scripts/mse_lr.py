import pandas
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import sys

def main():
    data = pandas.read_csv(sys.argv[1])

    control_rnn_size = np.unique(np.sort(data['control_rnn_size'].to_numpy()))

    fig, axs = plt.subplots(2, int(np.ceil(len(control_rnn_size) / 2)))
    fig.set_size_inches((20, 15))
    fig.set_dpi(120)

    for idx, crs in enumerate(control_rnn_size):
        ax = axs.reshape(-1)[idx]
        crs = control_rnn_size[idx]

        sns.lineplot(data=data[data['control_rnn_size'] == crs],
                     x='lr',
                     y='val_mse',
                     hue='encoder_size',
                     ax=ax)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"{crs}")

    fig.tight_layout()
    fig.savefig('boxplot.png')


if __name__ == "__main__":
    main()
