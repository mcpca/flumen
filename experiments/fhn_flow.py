from flow_model.data import LogNormalSqWave

from dynamics import Dynamics
from utils import parse_args, print_gpu_info
from sim_and_train import sim_and_train
from math import log


class FitzHughNagumo(Dynamics):

    def __init__(self, tau, a, b):
        super().__init__(2, 1)
        self.tau = tau
        self.a = a
        self.b = b

    def _dx(self, x, u):
        v, w = x

        dv = 50 * (v - v**3 - w + u)
        dw = (v - self.a - self.b * w) / self.tau

        return (dv, dw)


def main():
    args = parse_args()

    dynamics = FitzHughNagumo(tau=0.8, a=-0.3, b=1.4)
    control_generator = LogNormalSqWave(mean=log(0.2), std=0.1, period=5)

    sim_and_train(args, dynamics, control_generator)


if __name__ == '__main__':
    print_gpu_info()
    main()
