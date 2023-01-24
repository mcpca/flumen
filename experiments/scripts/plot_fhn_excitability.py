import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from flow_model import Experiment, pack_model_inputs
from trajectory_sampler import TrajectorySampler
from sequence_generators import SequenceGenerator
import matplotlib.pyplot as plt
import numpy as np

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
    ap.add_argument('--plot_input', action='store_true', default=None)

    return ap.parse_args()


def main():
    args = parse_args()

    experiment: Experiment = torch.load(args.path, map_location=torch.device('cpu'))
    model = experiment.load_model()
    model.eval()

    time_horizon = 24
    amp_seq = [0.0, 0.23, 0.23, 0.0, 0.23, 0.0, 0.0]

    tr_sampler: TrajectorySampler = experiment.generator.sampler

    class FixedInput(SequenceGenerator,):

        def __init__(self, amp_seq):
            super(FixedInput, self).__init__(None)

            self._period = tr_sampler._seq_gen._period
            self._amp_seq = np.array(amp_seq).reshape((-1, 1))

        def _sample_impl(self, time_range, delta):
            n_control_vals = int(1 +
                                 np.floor((time_range[1] - time_range[0]) / delta))

            control_seq = np.repeat(self._amp_seq, self._period, axis=0)[:n_control_vals]

            return control_seq

    tr_sampler._seq_gen = FixedInput(amp_seq)
    tr_sampler._seq_gen._rng = np.random.default_rng()
    tr_sampler.state_generator._rng = np.random.default_rng()
    delta = tr_sampler._delta

    # Figure out where to shade the plots
    input_seq = tr_sampler._seq_gen.sample((0, time_horizon), delta)
    input_active = ((input_seq > 0.2) & (input_seq < 0.25)).reshape((-1,))
    fill_start = np.where(input_active[:-1] < input_active[1:])[0]
    fill_stop = np.where(input_active[:-1] > input_active[1:])[0]
    control_times = np.arange(0, time_horizon + delta, delta)

    for i_plot in range(args.n_plots):
        n_plots = 1 + model.state_dim if args.plot_input else model.state_dim
        fig, ax = plt.subplots(n_plots, 1, sharex=True)

        x0, t, y, u = tr_sampler.get_example(
            time_horizon=time_horizon, n_samples=1000)

        x0_feed, t_feed, u_feed = pack_model_inputs(x0, t, u, delta)

        y_pred = experiment.predict(model, t_feed, x0_feed, u_feed)
        print(np.mean(np.square(y - np.flip(y_pred, 0))))

        for k, ax_ in enumerate(ax[:model.state_dim]):
            for (start, stop) in zip(fill_start, fill_stop):
                ax_.axvspan(control_times[start.item()], control_times[stop.item()], fc='k', alpha=0.2)
            ax_.plot(t_feed, y_pred[:, k], 'k', label='Prediction')
            ax_.plot(t, y[:, k], 'b--', label='True state')
            ax_.set_ylabel(f"$x_{k+1}$")

        if args.plot_input:
            ax[-1].axvspan(4, 12, fc='k', alpha=0.2)
            ax[-1].axvspan(16, 20, fc='k', alpha=0.2)
            ax[-1].step(np.arange(0., time_horizon + delta, delta), u)
            ax[-1].set_ylabel("$u$")

        # if i_plot == 0:
        #     ax[0].legend(loc='upper right')

        ax[-1].set_xlabel("$t$")

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        fig.savefig(f"prediction_{i_plot}.pdf")
        plt.close(fig)


if __name__ == '__main__':
    main()
