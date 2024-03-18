import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from trajectory_sampler import TrajectorySampler
from sequence_generators import SinusoidalSequence
from flow_model import Experiment, pack_model_inputs

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from os.path import dirname, isfile

PREFIX = 'mse_new_inputs_'

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

plt.rc('axes', labelsize=14)
TICK_SIZE = 12
plt.rc('xtick', labelsize=TICK_SIZE)
plt.rc('ytick', labelsize=TICK_SIZE)


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file")
    ap.add_argument('--n_mc', type=int, default=100)
    ap.add_argument('--force', action='store_true')

    return ap.parse_args()


def main():
    args = parse_args()

    experiment: Experiment = torch.load(args.path,
                                        map_location=torch.device('cpu'))
    print(experiment)
    fname = PREFIX + experiment.train_id.hex + '.csv'

    if args.force or not isfile(fname):
        loss_vals = compute_loss_vals(args, experiment)
        loss_vals.to_csv(fname)
    else:
        loss_vals = pd.read_csv(fname)

    fig, ax = plt.subplots()

    ax.set_yscale('log')
    sns.boxplot(x='Input distribution', y='Loss', data=loss_vals, ax=ax)
    fig.tight_layout()
    fig.savefig(PREFIX + experiment.train_id.hex + '.pdf')

    plt.show()


def compute_loss_vals(args, experiment: Experiment):
    model = experiment.load_model()
    model.eval()

    generator_p: TrajectorySampler = experiment.generator.sampler
    generator_p.reset_rngs()
    delta = generator_p._delta
    generator_q = TrajectorySampler(generator_p._dyn,
                                    delta,
                                    SinusoidalSequence(max_freq=0.2),
                                    method='RK45')

    generators = {'$P_u$': generator_p, '$Q_u$': generator_q}

    loss_vals = []

    multiplier = 1

    for gen_name, generator in generators.items():
        for _ in range(args.n_mc):
            x0, t, y, u = generator.get_example(
                time_horizon=multiplier * experiment.generator.time_horizon,
                n_samples=multiplier * 200)

            x0_feed, t_feed, u_feed, delta_feed = pack_model_inputs(
                x0, t, u, delta)

            y_pred = experiment.predict(model, x0_feed, u_feed, delta_feed)
            sq_error = np.square(y - np.flip(y_pred, 0))
            loss = np.mean(np.sum(sq_error, axis=1))

            loss_vals.append([gen_name, loss])

    return pd.DataFrame(loss_vals, columns=['Input distribution', 'Loss'])


if __name__ == '__main__':
    main()
