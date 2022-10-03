import torch
from trajectory import TrajectoryGenerator, pack_model_inputs, SinusoidalSequence
from meta import Meta

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from os.path import dirname, isfile

PREFIX = 'mse_new_inputs_'

plt.rc('axes', labelsize=18)
TICK_SIZE=14
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

    meta: Meta = torch.load(args.path, map_location=torch.device('cpu'))
    print(meta)
    fname = PREFIX + meta.train_id.hex + '.csv'

    if args.force or not isfile(fname):
        loss_vals = compute_loss_vals(args, meta)
        loss_vals.to_csv(fname)
    else:
        loss_vals = pd.read_csv(fname)

    fig, ax = plt.subplots()

    ax.set_yscale('log')
    sns.boxplot(x='Input distribution', y='Loss', data=loss_vals, ax=ax)
    fig.tight_layout()
    fig.savefig(PREFIX + meta.train_id.hex + '.pdf')

    plt.show()


def compute_loss_vals(args, meta):
    meta.set_root(dirname(__file__))
    model = meta.load_model()
    model.eval()

    generator_p: TrajectoryGenerator = meta.generator
    delta = generator_p._delta
    generator_q = TrajectoryGenerator(generator_p._dyn, delta,
                                      SinusoidalSequence())

    generators = {'$P_u$': generator_p, '$Q_u$': generator_q}

    loss_vals = []

    for gen_name, generator in generators.items():
        for _ in range(args.n_mc):
            x0, t, y, u = generator.get_example(
                time_horizon=meta.data_time_horizon,
                n_samples=200)

            x0_feed, t_feed, u_feed = pack_model_inputs(x0, t, u, delta)
            y_pred = meta.predict(model, t_feed, x0_feed, u_feed)
            sq_error = np.square(y - np.flip(y_pred, 0))
            loss = np.mean(np.sum(sq_error, axis=1))

            loss_vals.append([gen_name, loss])

    return pd.DataFrame(loss_vals, columns=['Input distribution', 'Loss'])


if __name__ == '__main__':
    main()
