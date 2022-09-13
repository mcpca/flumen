from flow_model.data import SinusoidalSequence

from dynamics import Dynamics
from utils import parse_args, print_gpu_info
from sim_and_train import sim_and_train

import numpy as np


class Pendulum(Dynamics):
    def __init__(self, damping, freq=2 * np.pi):
        super().__init__(2, 1)
        self.damping = damping
        self.freq2 = freq ** 2

    def _dx(self, x, u):
        p, v = x

        dp = v
        dv = -self.freq2 * np.sin(p) - self.damping * v + u

        return (dp, dv)


def main():
    args = parse_args()

    dynamics = Pendulum(damping=0.01)
    control_generator = SinusoidalSequence()

    sim_and_train(args, dynamics, control_generator)


if __name__ == '__main__':
    print_gpu_info()
    main()
