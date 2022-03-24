from trajectory import GaussianSqWave
from dynamics import FitzHughNagumo
from utils import parse_args, print_gpu_info
from sim_and_train import sim_and_train


def main():
    args = parse_args()

    dynamics = FitzHughNagumo(tau=1., a=-0.3, b=1.4)
    control_generator = GaussianSqWave(period=10)

    sim_and_train(args, dynamics, control_generator)


if __name__ == '__main__':
    print_gpu_info()
    main()
