import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import dynamics
import sequence_generators
import initial_state
from trajectory_sampler import TrajectorySampler
from argparse import ArgumentParser

from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import numpy as np


def parse_args():
    ap = ArgumentParser()

    ap.add_argument('--control_delta', type=float, default=0.1)
    ap.add_argument('--time_horizon', type=float, default=5.)
    ap.add_argument('--n_samples', type=int, default=10000)
    ap.add_argument('--period', type=int, default=10)

    return ap.parse_args()


def main():
    args = parse_args()

    dyn = dynamics.HodgkinHuxleyFS()
    inputs = sequence_generators.UniformSqWave(period=args.period)
    gen = TrajectorySampler(
        dynamics=dyn,
        control_delta=args.control_delta,
        control_generator=inputs,
        initial_state_generator=initial_state.HHFSInitialState(),
        method='BDF')

    n_dims = dyn.dims()[0]

    n_samples = int(args.n_samples * args.time_horizon)
    _, t, y, _ = gen.get_example(time_horizon=args.time_horizon,
                                 n_samples=n_samples)

    y_1 = y[:, 0].ravel()
    y_1 = (y_1 - np.min(y_1)) / np.ptp(y_1)

    # y_2 = y[:, 5].ravel()
    # y_2 = (y_2 - np.min(y_2)) / np.ptp(y_2)

    y_n = y_1
    y_p = y_n

    likelihood_ratio = y_p / np.mean(y_p)

    lr_bound = np.max(likelihood_ratio)

    rng = np.random.default_rng()
    u = rng.uniform(size=(len(likelihood_ratio), ))
    keep_idxs = (u <= (likelihood_ratio / lr_bound))

    print(
        f"#accepted = {np.sum(keep_idxs) / args.time_horizon}, 1/M = {100/lr_bound}%, true acceptance = {100 * np.sum(keep_idxs) / n_samples}%"
    )

    # peaks, _ = find_peaks(y[:, 0].ravel())
    # keep_idxs[peaks] = True

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, y[:, 0], 'k')
    ax[0].plot(t[keep_idxs], y[keep_idxs, 0], 'bo', alpha=0.3)
    # ax[1].plot(t, y[:, 5], 'k')
    # ax[1].plot(t[keep_idxs], y[keep_idxs, 5], 'bo', alpha=0.3)
    # ax[1].plot(t, u)
    ax[-1].plot(t, likelihood_ratio / lr_bound)

    ax[-1].set_ylabel("Desired density")
    ax[0].set_ylabel("Samples")

    plt.show()


if __name__ == '__main__':
    main()
