import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import dynamics
import sequence_generators
import initial_state
from trajectory_sampler import TrajectorySampler
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import numpy as np


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', labelsize=18)
TICK_SIZE = 10
plt.rc('xtick', labelsize=TICK_SIZE)
plt.rc('ytick', labelsize=TICK_SIZE)
plt.rc('legend', fontsize=12)

def parse_args():
    ap = ArgumentParser()

    ap.add_argument('--control_delta', type=float, default=0.2)
    ap.add_argument('--time_horizon', type=float, default=15.)
    ap.add_argument('--period', type=int, default=5)
    ap.add_argument('--std', type=float, default=5.)
    ap.add_argument('--mask', type=str, default=None)

    return ap.parse_args()


def main():
    args = parse_args()

    colour_mask = [int(m) for m in args.mask.split(',')] if args.mask else []

    dyn = dynamics.VanDerPol(damping=1.0)
    inputs = sequence_generators.GaussianSqWave(period=args.period,
                                                std=args.std)

    gen = TrajectorySampler(
        dynamics=dyn,
        control_delta=args.control_delta,
        control_generator=inputs,
        initial_state_generator=initial_state.GaussianInitialState(2),
        method='BDF')

    n_dims = dyn.dims()[0]

    fig, ax = plt.subplots(n_dims + 1, 1, sharex=True)
    fig.canvas.mpl_connect('close_event', on_close_window)
    plt.ion()

    while True:
        _, t, y, u = gen.get_example(args.time_horizon,
                                     n_samples=int(1000 * args.time_horizon))

        for k in range(n_dims):
            plot_colour = 'r' if k in colour_mask else 'b'
            ax[k].plot(t, y[:, k], plot_colour)
            ax[k].set_ylabel(r"$x_{}$".format(k))

        ax[-1].step(np.arange(0., args.time_horizon, args.control_delta),
                    u[:-1],
                    where='post')
        ax[-1].set_ylabel(r"$u$")
        ax[-1].set_xlabel(r"$t$")
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
