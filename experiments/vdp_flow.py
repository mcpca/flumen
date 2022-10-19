from dynamics import VanDerPol
from sequence_generators import GaussianSqWave
from utils import parse_args, print_gpu_info
from sim_and_train import run_experiment


def main():
    args = parse_args()

    dynamics = VanDerPol(1.0)
    control_generator = GaussianSqWave(period=5)

    run_experiment(args, dynamics, control_generator)


if __name__ == '__main__':
    print_gpu_info()
    main()
