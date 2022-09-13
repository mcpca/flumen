import torch
from flow_model.data import TrajectoryGenerator
import matplotlib.pyplot as plt
import numpy as np

from os.path import dirname
from argparse import ArgumentParser

from meta import Meta


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file")
    ap.add_argument('--print_info',
                    action='store_true',
                    help="Print training metadata and quit")

    return ap.parse_args()


def main():
    args = parse_args()

    meta: Meta = torch.load(args.path, map_location=torch.device('cpu'))

    print(meta)

    if args.print_info:
        return

    meta.set_root(dirname(__file__))

    model = meta.load_model()

    fig, ax = plt.subplots()
    ax.semilogy(meta.train_loss, label='Train loss')
    ax.semilogy(meta.val_loss, label='Validation loss')
    ax.semilogy(meta.test_loss, label='Test loss')
    ax.legend()
    plt.show()

    model.eval()

    trajectory_generator: TrajectoryGenerator = meta.generator
    delta = trajectory_generator._delta

    with torch.no_grad():
        while True:
            fig, ax = plt.subplots()

            x0, t, y, u = trajectory_generator.get_example(
                time_horizon=10 * meta.args.time_horizon, n_samples=1000)

            x0_feed, t_feed, u_feed = pack_model_inputs(x0, t, u, delta)

            y_pred = meta.predict(model, t_feed, x0_feed, u_feed)

            print(np.mean(np.square(y - np.flip(y_pred, 0))))

            ax.plot(t_feed, y_pred, 'k', label='Prediction')
            ax.plot(t, y, 'b--', label='True state')

            ax.legend()

            plt.show()
            plt.close(fig)


def pack_model_inputs(x0, t, u, delta):
    t = torch.Tensor(t.reshape((-1, 1))).flip(0)
    x0 = torch.Tensor(x0.reshape((1, -1))).repeat(t.shape[0], 1)
    rnn_inputs = torch.empty((t.shape[0], u.size, 2))
    lengths = torch.empty((t.shape[0], ), dtype=torch.long)

    for idx, (t_, u_) in enumerate(zip(t, rnn_inputs)):
        control_seq = torch.from_numpy(u)
        deltas = torch.ones_like(control_seq)

        seq_len = 1 + int(np.floor(t_ / delta))
        lengths[idx] = seq_len
        deltas[seq_len - 1] = ((t_ - delta * (seq_len - 1)) / delta).item()
        deltas[seq_len:] = 0.

        u_[:] = torch.hstack((control_seq, deltas))

    u_packed = torch.nn.utils.rnn.pack_padded_sequence(rnn_inputs,
                                                       lengths,
                                                       batch_first=True,
                                                       enforce_sorted=True)

    return x0, t, u_packed




if __name__ == '__main__':
    main()
