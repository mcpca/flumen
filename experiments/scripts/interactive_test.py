import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from flow_model import Experiment, pack_model_inputs
from trajectory_sampler import TrajectorySampler
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file")
    ap.add_argument('--print_info',
                    action='store_true',
                    help="Print training metadata and quit")

    ap.add_argument('--all_states',
                    action='store_true',
                    help="Plot all states")

    return ap.parse_args()


def main():
    args = parse_args()

    experiment: Experiment = torch.load(args.path,
                                        map_location=torch.device('cpu'))

    print(experiment)

    if args.print_info:
        print(experiment.args_str())
        print(vars(experiment.generator._seq_gen))
        return

    model = experiment.load_model()

    fig, ax = plt.subplots()
    ax.semilogy(experiment.train_loss, label='Train loss')
    ax.semilogy(experiment.val_loss, label='Validation loss')
    ax.semilogy(experiment.test_loss, label='Test loss')
    ax.legend()
    ax.vlines(np.argmin(experiment.val_loss),
              ymin=0,
              ymax=np.max(experiment.test_loss),
              colors='r',
              linestyles='dashed')
    ax.set_xlabel('Training epochs')
    fig.tight_layout()
    fig.savefig('loss.pdf')
    plt.show()

    model.eval()

    trajectory_generator: TrajectorySampler = experiment.generator.sampler
    delta = trajectory_generator._delta

    n_plots = 1 + model.state_dim if args.all_states else 2

    fig, ax = plt.subplots(n_plots, 1, sharex=True)
    fig.canvas.mpl_connect('close_event', on_close_window)
    plt.ion()

    with torch.no_grad():
        while True:
            time_horizon = experiment.generator.time_horizon

            x0, t, y, u = trajectory_generator.get_example(
                time_horizon=time_horizon,
                n_samples=int(1 + 1000 * time_horizon))

            x0_feed, t_feed, u_feed = pack_model_inputs(x0, t, u, delta)

            y_pred = experiment.predict(model, x0_feed, u_feed)

            sq_error = np.square(y - np.flip(y_pred, 0))
            print(np.mean(sq_error))

            for k, ax_ in enumerate(ax[:-1]):
                ax_.plot(t, y[:, k], 'b', label='True state')
                ax_.plot(t_feed, y_pred[:, k], c='orange', label='Prediction')
                ax_.set_ylabel(f"$x_{k+1}$")

            ax[0].legend()

            ax[-1].step(np.arange(0., time_horizon + delta, delta), u)
            ax[-1].set_ylabel("$u$")
            ax[-1].set_xlabel("$t$")

            fig.tight_layout()
            fig.subplots_adjust(hspace=0)
            plt.setp([a.get_xticklabels() for a in fig.axes[:-1]],
                     visible=False)

            plt.draw()

            # Wait for key press
            skip = False
            while not skip:
                skip = plt.waitforbuttonpress()

            for ax_ in ax:
                ax_.clear()


def on_close_window(ev):
    sys.exit(0)


if __name__ == '__main__':
    main()
