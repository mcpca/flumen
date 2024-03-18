import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from flow_model import Experiment, pack_model_inputs
from trajectory_sampler import TrajectorySampler
import matplotlib.pyplot as plt
import numpy as np

from os.path import dirname
from argparse import ArgumentParser

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', labelsize=18)
TICK_SIZE = 10
plt.rc('xtick', labelsize=TICK_SIZE)
plt.rc('ytick', labelsize=TICK_SIZE)
plt.rc('legend', fontsize=12)


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file")
    ap.add_argument('--plot_input', action='store_true')
    ap.add_argument('--no_legend', action='store_true')
    ap.add_argument('--n_plots', type=int, default=1)
    ap.add_argument('--time_horizon', type=int, default=None)

    return ap.parse_args()


def main():
    args = parse_args()

    experiment: Experiment = torch.load(args.path,
                                        map_location=torch.device('cpu'))
    model = experiment.load_model()
    model.eval()

    time_horizon = (args.time_horizon if args.time_horizon else
                    experiment.generator.time_horizon)

    sampler: TrajectorySampler = experiment.generator.sampler
    from sequence_generators import SinusoidalSequence
    sampler._seq_gen = SinusoidalSequence(0.2)
    sampler.reset_rngs()
    delta = sampler._delta

    n_plots = 1 + model.state_dim if args.plot_input else model.state_dim

    for i_plot in range(args.n_plots):
        fig, ax = plt.subplots(n_plots, 1, sharex=True)
        n_samples = 1 + int(100 * time_horizon)

        x0, t, y, u = sampler.get_example(time_horizon=time_horizon,
                                          n_samples=n_samples)

        x0_feed, t_feed, u_feed, deltas = pack_model_inputs(x0, t, u, delta)

        y_pred = experiment.predict(model, x0_feed, u_feed, deltas)
        print(np.mean(np.square(y - np.flip(y_pred, 0))))

        for k, ax_ in enumerate(ax[:model.state_dim]):
            ax_.plot(t_feed, y_pred[:, k], c='orange', label='Model output')
            ax_.plot(t, y[:, k], 'b--', label='True state')
            ax_.set_ylabel(f"$x_{k+1}$")

        if args.plot_input:
            ax[-1].step(np.arange(0., time_horizon, delta), u[:-1],
                        where='post')
            ax[-1].set_ylabel("$u$")

        if i_plot == 0 and not args.no_legend:
            ax[0].legend(loc='upper right')

        ax[-1].set_xlabel("$t$")

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        fig.savefig(f"prediction_{i_plot}.pdf")
        plt.close(fig)


if __name__ == '__main__':
    main()
