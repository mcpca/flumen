from flow_model_odedata import run_experiment
from flow_model_odedata.utils import parse_args


def main():
    args = parse_args()

    run_experiment(args, load_data=True)


if __name__ == '__main__':
    main()
