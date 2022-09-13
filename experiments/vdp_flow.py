from flow_model.data import GaussianSqWave
from dynamics import Dynamics

from utils import parse_args, print_gpu_info
from sim_and_train import sim_and_train


class VanDerPol(Dynamics):
    def __init__(self, damping):
        super().__init__(2, 1)
        self.damping = damping

    def _dx(self, x, u):
        p, v = x

        dp = v
        dv = -p + (self.damping**2) * (1 - p**2) * v + u

        return (dp, dv)


def main():
    args = parse_args()

    dynamics = VanDerPol(1.5)
    control_generator = GaussianSqWave(period=5)

    sim_and_train(args, dynamics, control_generator)


if __name__ == '__main__':
    print_gpu_info()
    main()
