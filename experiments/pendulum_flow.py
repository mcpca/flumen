from trajectory_generator import SinusoidalSequence
from dynamics import Pendulum
from utils import parse_args, print_gpu_info
from sim_and_train import run_experiment


def main():
    args = parse_args()

    dynamics = Pendulum(damping=0.01)
    control_generator = SinusoidalSequence()

    run_experiment(args, dynamics, control_generator)


if __name__ == '__main__':
    print_gpu_info()
    main()
