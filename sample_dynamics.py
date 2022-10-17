import sys
sys.path.append('..')

import dynamics
import trajectory
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

    dyn = dynamics.FitzHughNagumo(tau=0.8, a=-0.3, b=1.4)
    inputs = trajectory.LogNormalSqWave(mean=np.log(0.2), std=args.std, period=args.period)
    gen = trajectory.TrajectoryGenerator(dyn, args.control_delta, inputs)

    while True:
        fig, ax = plt.subplots(2, 1, sharex=True)

        _, t, y, u = gen.get_example(time_horizon=args.time_horizon,
                                     n_samples=int(10 * args.time_horizon))
        ax[0].plot(t, y, 'k')
        ax[1].plot(args.control_delta * np.arange(u.size), u)
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()
