from utils import parse_args
from flow_model_odedata import run_experiment


def main():
    args = parse_args()

    run_experiment(args, load_data=True)


if __name__ == '__main__':
    main()
