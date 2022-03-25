from trajectory import SinusoidalSequence
from dynamics import Pendulum
from utils import parse_args, print_gpu_info
from sim_and_train import sim_and_train


def main():
    args = parse_args()

    dynamics = Pendulum(damping=0.01)
    control_generator = SinusoidalSequence()

    sim_and_train(args, dynamics, control_generator)


if __name__ == '__main__':
    print_gpu_info()
    main()
