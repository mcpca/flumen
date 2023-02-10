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
    trajectory_generator._noise_std = 0.
    delta = trajectory_generator._delta

    with torch.no_grad():
        while True:
            fig, ax = plt.subplots(1 + model.state_dim, 1, sharex=True)

            time_horizon=2 * experiment.generator.time_horizon

            x0, t, y, u = trajectory_generator.get_example(
                time_horizon=time_horizon,
                n_samples=int(1 + 1000 * time_horizon))

            x0_feed, t_feed, u_feed = pack_model_inputs(x0, t, u, delta)

            y_pred = experiment.predict(model, t_feed, x0_feed, u_feed)

            sq_error = np.square(y - np.flip(y_pred, 0))
            print(np.mean(sq_error))

            for k, ax_ in enumerate(ax[:-1]):
                ax_.plot(t_feed, y_pred[:, k], 'k', label='Prediction')
                ax_.plot(t, y[:, k], 'b--', label='True state')
                ax_.set_ylabel(f"$x_{k+1}$")

            ax[0].legend()

            ax[-1].step(np.arange(0., time_horizon + delta, delta), u)
            ax[-1].set_ylabel("$u$")
            ax[-1].set_xlabel("$t$")

            fig.tight_layout()
            fig.subplots_adjust(hspace=0)
            plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

            plt.show()
            plt.close(fig)


if __name__ == '__main__':
    main()
