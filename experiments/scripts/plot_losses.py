import pandas
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import sys


def main():
    data = pandas.read_csv(sys.argv[1])

    fig, ax = plt.subplots()

    sns.boxplot(data=data[['train_mse', 'val_mse', 'test_mse']], ax=ax)

    ax.set_yscale('log')

    fig.tight_layout()
    plt.show()
    # fig.savefig(sys.argv[1] + ".pdf")


if __name__ == "__main__":
    main()
