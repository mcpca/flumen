import torch
from torch.utils.data import DataLoader
from trajectory import TrajectoryDataset, TrajectoryGenerator, pack_model_inputs
from train import validate
from meta import Meta

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from os.path import dirname, isfile

PREFIX='mse_time_horizon_'


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file")
    ap.add_argument('--t_max', type=float, default=100)
    ap.add_argument('--n_t_steps', type=float, default=20)
    ap.add_argument('--n_mc', type=int, default=25)
    ap.add_argument('--force', action='store_true')

    return ap.parse_args()


def main():
    args = parse_args()

    meta: Meta = torch.load(args.path, map_location=torch.device('cpu'))
    fname = PREFIX + meta.train_id.hex + '.csv'

    if args.force or not isfile(fname):
        loss_vals = compute_loss_vals(args, meta)
        loss_vals.to_csv(fname)
    else:
        loss_vals = pd.read_csv(fname)

    fig, ax = plt.subplots()

    sns.lineplot(x='Time horizon', y='Loss', data=loss_vals, ax=ax)
    ax.set_yscale('log')
    plt.show()


def compute_loss_vals(args, meta):
    meta.set_root(dirname(__file__))
    model = meta.load_model()
    model.eval()

    generator: TrajectoryGenerator = meta.generator

    th_vals = np.linspace(15., args.t_max, args.n_t_steps)
    loss_vals = []

    for time_horizon in th_vals:
        n_samples = int(2000 * time_horizon / 15.)
        n_examples = int(100 * time_horizon / 15.)
        dset = TrajectoryDataset(generator, n_trajectories=args.n_mc,
                n_samples=n_samples,
                time_horizon=time_horizon,
                examples_per_traj=n_examples)

        dset.state[:] = ((dset.state[:] - meta.td_mean) @ meta.td_std_inv).type(torch.get_default_dtype())
        dset.init_state[:] = ((dset.init_state[:] - meta.td_mean) @ meta.td_std_inv).type(torch.get_default_dtype())

        dloader = DataLoader(dset, shuffle=False, batch_size=1024)
        loss_fn = torch.nn.MSELoss()
        print(f'T={time_horizon}')
        loss = validate(dloader, loss_fn, model, device=torch.device('cpu'))
        loss_vals.append([time_horizon, loss])

    return pd.DataFrame(loss_vals, columns=['Time horizon', 'Loss'])


if __name__ == '__main__':
    main()
