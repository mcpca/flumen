import torch

from argparse import ArgumentParser, ArgumentTypeError


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

    ap.add_argument('--noise_std',
                    type=nonnegative_float,
                    help="Standard deviation of measurement noise",
                    default=0.0)

    ap.add_argument('--noise_seed',
                    type=positive_int,
                    help="Measurement noise seed",
                    default=None)

    ap.add_argument(
        '--data_split',
        nargs=2,
        type=percentage,
        help="Percentage of data used for validation and test sets",
        default=[20, 20])

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
                    default=5)

    ap.add_argument('--encoder_depth',
                    type=positive_int,
                    help="Depth of the encoder",
                    default=3)

    ap.add_argument('--decoder_size',
                    type=positive_int,
                    help="Size (multiplier) of the decoder layers",
                    default=5)

    ap.add_argument('--decoder_depth',
                    type=positive_int,
                    help="Depth of the decoder",
                    default=3)

    ap.add_argument('--batch_size',
                    type=positive_int,
                    help="Batch size for training and validation",
                    default=256)

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
                    default=2)

    ap.add_argument('--sched_factor',
                    type=positive_int,
                    help="LR Scheduler -- Reduction factor",
                    default=5)

    ap.add_argument('--experiment_id',
                    type=str,
                    help="Human-readable experiment identifier. "
                    "Nothing is written to disk if this is not provided.",
                    default=None)

    ap.add_argument('--write_dir',
                    type=str,
                    help="Directory to which the model will be written.",
                    default='./outputs')

    ap.add_argument('--save_data',
                    type=str,
                    help="Path to write .pth trajectory dataset",
                    default=None)

    ap.add_argument('--load_data',
                    type=str,
                    help="Path to load .pth trajectory dataset",
                    default=None)

    return ap.parse_args()


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


def percentage(value):
    value = int(value)

    if not (0 <= value <= 100):
        raise ArgumentTypeError(f"{value} is not a valid percentage")

    return value


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
