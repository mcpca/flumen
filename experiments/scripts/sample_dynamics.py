import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import dynamics
import sequence_generators
import initial_state
from trajectory_sampler import TrajectorySampler
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

    dyn = dynamics.HodgkinHuxleyIB()
    inputs = sequence_generators.UniformSqWave(period=args.period)
    gen = TrajectorySampler(
        dynamics=dyn,
        control_delta=args.control_delta,
        control_generator=inputs,
        initial_state_generator=initial_state.HHIBInitialState(),
        method='BDF')

    n_dims = dyn.dims()[0]

    fig, ax = plt.subplots(n_dims + 1, 1, sharex=True)
    fig.canvas.mpl_connect('close_event', on_close_window)
    plt.ion()

    while True:
        _, t, y, u = gen.get_example(args.time_horizon,
                                     n_samples=int(1000 * args.time_horizon))

        for k in range(n_dims):
            ax[k].plot(t, y[:, k], 'k')

        ax[-1].plot(args.control_delta * np.arange(u.size), u)
        plt.draw()

        # Wait for key press
        skip = False
        while not skip:
            skip = plt.waitforbuttonpress()

        for ax_ in ax:
            ax_.clear()


def on_close_window(ev):
    sys.exit(0)


if __name__ == '__main__':
    main()
