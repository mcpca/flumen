import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from flow_model_odedata import (ODEExperiment, TrajectoryGenerator,
                                pack_model_inputs)
import matplotlib.pyplot as plt
import numpy as np

from os.path import dirname
from argparse import ArgumentParser

plt.rc('axes', labelsize=20)
TICK_SIZE = 14
plt.rc('xtick', labelsize=TICK_SIZE)
plt.rc('ytick', labelsize=TICK_SIZE)
plt.rc('legend', fontsize=15)


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file")
    ap.add_argument('--n_plots', type=int, default=1)
    ap.add_argument('--time_horizon', type=int, default=None)

    return ap.parse_args()


def main():
    args = parse_args()

    experiment: ODEExperiment = torch.load(args.path, map_location=torch.device('cpu'))
    model = experiment.load_model()
    model.eval()

    time_horizon = (args.time_horizon
                    if args.time_horizon else experiment.data_time_horizon)

    trajectory_generator: TrajectoryGenerator = experiment.generator
    delta = trajectory_generator._delta
    # trajectory_generator._seq_gen = SinusoidalSequence()

    for i_plot in range(args.n_plots):
        fig, ax = plt.subplots(1 + model.state_dim, 1, sharex=True)

        x0, t, y, u = trajectory_generator.get_example(
            time_horizon=time_horizon, n_samples=1000)

        x0_feed, t_feed, u_feed = pack_model_inputs(x0, t, u, delta)

        y_pred = experiment.predict(model, t_feed, x0_feed, u_feed)
        print(np.mean(np.square(y - np.flip(y_pred, 0))))

        for k, ax_ in enumerate(ax[:-1]):
            ax_.plot(t_feed, y_pred[:, k], 'k', label='Prediction')
            ax_.plot(t, y[:, k], 'b--', label='True state')
            ax_.set_ylabel(f"$x_{k+1}$")

        ax[-1].step(np.arange(0., time_horizon + delta, delta), u)
        ax[-1].set_ylabel("$u$")

        if i_plot == 0:
            ax[0].legend(loc='upper right')

        ax[-1].set_xlabel("$t$")

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        fig.savefig(f"prediction_{i_plot}.pdf")
        plt.close(fig)


if __name__ == '__main__':
    main()
