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
    sampler.reset_rngs()
    delta = sampler._delta

    time_scale = sampler._dyn.time_scale
    y_scale = sampler._dyn.v_scale

    n_plots = model.output_dim
    plot_heights = model.output_dim * [
        2,
    ]

    if args.plot_input:
        n_plots += 1
        plot_heights += [
            1,
        ]

    for i_plot in range(args.n_plots):
        fig, ax = plt.subplots(n_plots,
                               1,
                               sharex=True,
                               height_ratios=plot_heights)
        n_samples = 1 + int(2000 * time_horizon)

        x0, t, y, u = sampler.get_example(time_horizon=time_horizon,
                                          n_samples=n_samples)

        y = y[:, tuple(bool(v) for v in sampler._dyn.mask)]

        x0_feed, t_feed, u_feed, deltas_feed = pack_model_inputs(
            x0, t, u, delta)

        y_pred = experiment.predict(model, x0_feed, u_feed, deltas_feed)
        print(np.mean(np.square(y - np.flip(y_pred, 0))))

        for k, ax_ in enumerate(ax[:model.output_dim]):
            ax_.plot(time_scale * t_feed,
                     y_scale * y_pred[:, k],
                     'orange',
                     label='Prediction')

            ax_.plot(time_scale * t,
                     y_scale * y[:, k],
                     'b--',
                     # linewidth=0.5,
                     label='True state')

            y_label = f"$V_{k+1}$ [mV]" if model.output_dim > 1 else "$V$ [mV]"
            ax_.set_ylabel(y_label)

        if args.plot_input:
            ax[-1].step(time_scale * np.arange(0., time_horizon, delta),
                        u[:-1],
                        where="post")
            ax[-1].set_ylabel("$u$ [$\mu$A]")

        # if i_plot == 0:
        #     ax[0].legend(loc='upper right')

        ax[-1].set_xlabel("$t$ [ms]")

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        fig.savefig(f"prediction_{i_plot}.pdf")
        plt.close(fig)


if __name__ == '__main__':
    main()
