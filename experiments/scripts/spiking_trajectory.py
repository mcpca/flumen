import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dynamics import HodgkinHuxleyFS
from sequence_generators import UniformSqWave
from initial_state import InitialStateGenerator
from trajectory_sampler import TrajectorySampler
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import numpy as np


def parse_args():
    ap = ArgumentParser()

    ap.add_argument('--control_delta', type=float, default=0.1)
    ap.add_argument('--time_horizon', type=float, default=10.)
    ap.add_argument('--period', type=int, default=10)

    return ap.parse_args()


class HHFSEquilibrium(InitialStateGenerator):

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng if rng else np.random.default_rng()
        self.hhfs = HodgkinHuxleyFS()

    def _sample_impl(self):
        return (self.hhfs.v_l / self.hhfs.v_scale, 0., 0., 0.)


def main():
    args = parse_args()

    dyn = HodgkinHuxleyFS()
    inputs = UniformSqWave(period=args.period)
    gen = TrajectorySampler(dynamics=dyn,
                            control_delta=args.control_delta,
                            control_generator=inputs,
                            initial_state_generator=HHFSEquilibrium(),
                            method='BDF')

    fig, ax = plt.subplots(2,
                           1,
                           sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]})

    _, t, y, u = gen.get_example(args.time_horizon,
                                 n_samples=int(10000 * args.time_horizon))

    ax[0].plot(dyn.time_scale * t, dyn.v_scale * y[:, 0], 'steelblue')
    ax[1].step(dyn.time_scale *
               np.arange(0., args.time_horizon, args.control_delta),
               u[:-1],
               c='darkolivegreen',
               where='post')

    ax[0].set_ylabel(r'$V$ [mV]')
    ax[1].set_ylabel(r'$u$ [$\mu$A]')
    ax[1].set_xlabel(r'$t$ [ms]')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    plt.show()


if __name__ == '__main__':
    main()
