import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import dynamics
import sequence_generators
from flow_model_odedata import TrajectoryGenerator
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import numpy as np


def parse_args():
    ap = ArgumentParser()

    ap.add_argument('--control_delta', type=float, default=1.0)
    ap.add_argument('--time_horizon', type=float, default=10.)
    ap.add_argument('--period', type=int, default=5)
    ap.add_argument('--std', type=float, default=1.)

    return ap.parse_args()


def main():
    args = parse_args()

    dyn = dynamics.Pendulum(freq=0.5, damping=0.1)
    inputs = sequence_generators.SinusoidalSequence(max_freq=1.0)
    inputs._amp_std = np.log(args.std)
    gen = TrajectoryGenerator(dyn, args.control_delta, inputs)

    while True:
        fig, ax = plt.subplots(2, 1, sharex=True)

        _, t, y, u = gen.get_example(time_horizon=args.time_horizon,
                                     n_samples=int(100 * args.time_horizon))
        ax[0].plot(t, y, 'k')
        ax[1].plot(args.control_delta * np.arange(u.size), u)
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()
