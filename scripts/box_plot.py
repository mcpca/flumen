import pandas
import seaborn as sns
from matplotlib import pyplot as plt
import sys

def main():
    data = pandas.read_csv(sys.argv[1])

    sns.boxplot(data=data[['train_mse', 'val_mse', 'test_mse']])

    plt.show()

if __name__ == "__main__":
    main()
