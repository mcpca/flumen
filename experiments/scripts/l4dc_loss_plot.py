import pandas
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', labelsize=20)
TICK_SIZE = 12
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=TICK_SIZE)


def main():
    data = pandas.read_csv(sys.argv[1])

    fig, ax = plt.subplots()

    data = data[['train_mse', 'val_mse', 'test_mse']]
    names = {
        'train_mse': 'Train (windowed)',
        'val_mse': 'Validation',
        'test_mse': 'Test'
    }

    data = data.rename(columns=names)

    sns.boxplot(data=data, ax=ax)

    ax.set_yscale('log')
    ax.set_ylabel('Log loss')

    # sns.set_style('whitegrid')

    fig.tight_layout()
    plt.show()
    # fig.savefig(sys.argv[1] + ".pdf")


if __name__ == "__main__":
    main()
