from utils import parse_args
from sim_and_train import sim_and_train


def main():
    args = parse_args()

    sim_and_train(args, load_data=True)


if __name__ == '__main__':
    main()
