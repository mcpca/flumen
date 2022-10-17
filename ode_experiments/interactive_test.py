import torch
from meta import Meta
from trajectory_generator import TrajectoryGenerator, pack_model_inputs, SinusoidalSequence
import matplotlib.pyplot as plt
import numpy as np

from os.path import dirname
from argparse import ArgumentParser


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file")
    ap.add_argument('--print_info', action='store_true',
                    help="Print training metadata and quit")

    return ap.parse_args()


def main():
    args = parse_args()

    meta: Meta = torch.load(args.path, map_location=torch.device('cpu'))

    print(meta)

    if args.print_info:
        print(meta.args_str())
        print(vars(meta.generator._seq_gen))
        return

    meta.set_root(dirname(__file__))

    model = meta.load_model()

    fig, ax = plt.subplots()
    ax.semilogy(meta.train_loss, label='Train loss')
    ax.semilogy(meta.val_loss, label='Validation loss')
    ax.semilogy(meta.test_loss, label='Test loss')
    ax.legend()
    ax.vlines(
            np.argmin(meta.val_loss),
            ymin=0, ymax=np.max(meta.test_loss),
            colors='r',
            linestyles='dashed')
    ax.set_xlabel('Training epochs')
    fig.tight_layout()
    fig.savefig('loss.pdf')
    plt.show()

    model.eval()

    trajectory_generator: TrajectoryGenerator = meta.generator
    delta = trajectory_generator._delta
    trajectory_generator._seq_gen = SinusoidalSequence()

    with torch.no_grad():
        while True:
            fig, ax = plt.subplots()

            x0, t, y, u = trajectory_generator.get_example(time_horizon=20.,
                                                           n_samples=1000)

            x0_feed, t_feed, u_feed = pack_model_inputs(x0, t, u, delta)

            y_pred = meta.predict(model, t_feed, x0_feed, u_feed)

            sq_error = np.square(y - np.flip(y_pred, 0))
            print(np.mean(np.sum(sq_error, axis=1)))

            ax.plot(t_feed, y_pred, 'k', label='Prediction')
            ax.plot(t, y, 'b--', label='True state')

            ax.legend()

            plt.show()
            plt.close(fig)


if __name__ == '__main__':
    main()
