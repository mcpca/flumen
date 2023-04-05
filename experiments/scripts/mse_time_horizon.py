import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from flow_model import (Experiment, RawTrajectoryDataset, pack_model_inputs)
from trajectory_sampler import TrajectorySampler

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from os.path import isfile

PREFIX = 'mse_time_horizon_'

plt.rc('axes', labelsize=14)
TICK_SIZE = 11
plt.rc('xtick', labelsize=TICK_SIZE)
plt.rc('ytick', labelsize=TICK_SIZE)


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file")
    ap.add_argument('--t_max', type=float, default=100)
    ap.add_argument('--n_t_steps', type=int, default=20)
    ap.add_argument('--n_traj', type=int, default=25)
    ap.add_argument('--force', action='store_true')

    return ap.parse_args()


def main():
    args = parse_args()

    experiment: Experiment = torch.load(args.path,
                                        map_location=torch.device('cpu'))
    fname = PREFIX + experiment.train_id.hex + '.csv'

    if args.force or not isfile(fname):
        loss_vals = compute_loss_vals(args, experiment)
        loss_vals.to_csv(fname)
    else:
        loss_vals = pd.read_csv(fname)

    fig, ax = plt.subplots()

    sns.lineplot(x='Time horizon', y='Loss', data=loss_vals, ax=ax)
    ax.set_yscale('log')
    fig.tight_layout()
    plt.show()


def compute_loss_vals(args, experiment):
    model = experiment.load_model()
    model.eval()

    sampler: TrajectorySampler = experiment.generator.sampler
    delta = sampler._delta
    sampler.reset_rngs()

    th_vals = np.linspace(experiment.generator.time_horizon, args.t_max,
                          args.n_t_steps)
    loss_vals = []

    for time_horizon in th_vals:
        print(f"Time horizon = {time_horizon:.2f}")
        n_samples = int(200. * time_horizon /
                        experiment.generator.time_horizon)

        dset_raw = RawTrajectoryDataset.generate(sampler,
                                                 time_horizon,
                                                 n_trajectories=args.n_traj,
                                                 n_samples=n_samples,
                                                 noise_std=0.)

        for (x0, _, t, y, _, u) in dset_raw:
            x0_p, _, u_p = pack_model_inputs(x0, t, u.numpy(), delta)
            y_p = np.flip(experiment.predict(model, x0_p, u_p), 0)
            loss_vals.append([time_horizon, np.mean(np.square(y_p - y.numpy()))])

    return pd.DataFrame(loss_vals, columns=['Time horizon', 'Loss'])


if __name__ == '__main__':
    main()
