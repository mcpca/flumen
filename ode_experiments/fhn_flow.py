from trajectory_generator import LogNormalSqWave
from dynamics import FitzHughNagumo
from utils import parse_args, print_gpu_info
from sim_and_train import sim_and_train
from math import log

def main():
    args = parse_args()

    dynamics = FitzHughNagumo(tau=0.8, a=-0.3, b=1.4)
    control_generator = LogNormalSqWave(mean=log(0.2), std=0.1, period=5)

    sim_and_train(args, dynamics, control_generator)


if __name__ == '__main__':
    print_gpu_info()
    main()
