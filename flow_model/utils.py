import torch
import numpy as np
from argparse import ArgumentParser, ArgumentTypeError


def print_gpu_info():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"CUDA is available, {n_gpus} devices can be used.")
        current_dev = torch.cuda.current_device()

        for id in range(n_gpus):
            msg = f"Device {id}: {torch.cuda.get_device_name(id)}"

            if id == current_dev:
                msg += " [Current]"

            print(msg)


def get_arg_parser():
    ap = ArgumentParser()

    ap.add_argument('--control_rnn_size',
                    type=positive_int,
                    help="Size of the RNN hidden state",
                    default=6)

    ap.add_argument('--control_rnn_depth',
                    type=positive_int,
                    help="Depth of the RNN",
                    default=1)

    ap.add_argument('--encoder_size',
                    type=positive_int,
                    help="Size (multiplier) of the encoder layers",
                    default=1)

    ap.add_argument('--encoder_depth',
                    type=positive_int,
                    help="Depth of the encoder",
                    default=2)

    ap.add_argument('--decoder_size',
                    type=positive_int,
                    help="Size (multiplier) of the decoder layers",
                    default=1)

    ap.add_argument('--decoder_depth',
                    type=positive_int,
                    help="Depth of the decoder",
                    default=2)

    ap.add_argument('--batch_size',
                    type=positive_int,
                    help="Batch size for training and validation",
                    default=1024)

    ap.add_argument('--lr',
                    type=positive_float,
                    help="Initial learning rate",
                    default=1e-3)

    ap.add_argument('--n_epochs',
                    type=positive_int,
                    help="Max number of epochs",
                    default=10000)

    ap.add_argument('--es_patience',
                    type=positive_int,
                    help="Early stopping -- patience (epochs)",
                    default=30)

    ap.add_argument('--es_delta',
                    type=nonnegative_float,
                    help="Early stopping -- minimum loss change",
                    default=0.)

    ap.add_argument('--sched_patience',
                    type=positive_int,
                    help="LR Scheduler -- Patience epochs",
                    default=10)

    ap.add_argument('--sched_cooldown',
                    type=positive_int,
                    help="LR scheduler -- Cooldown epochs",
                    default=0)

    ap.add_argument('--sched_factor',
                    type=positive_int,
                    help="LR Scheduler -- Reduction factor",
                    default=5)

    ap.add_argument('--use_batch_norm',
                    action='store_true',
                    help="Use batch normalisation in encoder and decoder.")

    ap.add_argument(
        '--whiten_data',
        action='store_true',
        help='Apply whitening normalization to the data before training.')

    ap.add_argument('--experiment_id',
                    type=str,
                    help="Human-readable experiment identifier. "
                    "Nothing is written to disk if this is not provided.",
                    default=None)

    ap.add_argument('--write_dir',
                    type=str,
                    help="Directory to which the model will be written.",
                    default='./outputs')

    return ap


def positive_int(value):
    value = int(value)

    if value <= 0:
        raise ArgumentTypeError(f"{value} is not a positive integer")

    return value


def positive_float(value):
    value = float(value)

    if value <= 0:
        raise ArgumentTypeError(f"{value} is not a positive float")

    return value


def nonnegative_float(value):
    value = float(value)

    if value < 0:
        raise ArgumentTypeError(f"{value} is not a nonnegative float")

    return value


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

    return x0, t, u_packed, rnn_inputs[:, :-1, -1].unsqueeze(-1)
