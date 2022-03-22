import torch

import argparse
from argparse import ArgumentParser


def parse_args():
    ap = ArgumentParser()

    ap.add_argument('--control_delta',
                    type=positive_float,
                    help="Control sampling rate",
                    default=0.5)

    ap.add_argument('--time_horizon',
                    type=positive_float,
                    help="Time horizon",
                    default=10.)

    ap.add_argument('--n_trajectories',
                    type=positive_int,
                    help="Number of trajectories to sample",
                    default=100)

    ap.add_argument('--n_samples',
                    type=positive_int,
                    help="Number of state samples per trajectory",
                    default=50)

    ap.add_argument(
        '--train_val_split',
        type=percentage,
        help="Percentage of the generated data that is used for training",
        default=70)

    ap.add_argument('--batch_size',
                    type=positive_int,
                    help="Batch size for training and validation",
                    default=256)

    ap.add_argument('--control_rnn_size',
                    type=positive_int,
                    help="Size of the RNN hidden state",
                    default=6)

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

    ap.add_argument('--save_model',
                    type=str,
                    help="Path to write .pth model",
                    default=None)

    return ap.parse_args()


def positive_int(value):
    value = int(value)

    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")

    return value


def positive_float(value):
    value = float(value)

    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive float")

    return value


def nonnegative_float(value):
    value = float(value)

    if value < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a nonnegative float")

    return value


def percentage(value):
    value = int(value)

    if not (0 <= value <= 100):
        raise argparse.ArgumentTypeError(f"{value} is not a valid percentage")

    return value

def print_gpu_info():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"CUDA is available, {n_gpus} can be used.")
        current_dev = torch.cuda.current_device()

        for id in range(n_gpus):
            msg = f"Device {id}: {torch.cuda.get_device_name(id)}"

            if id == current_dev:
                msg += " [Current]"

            print(msg)
