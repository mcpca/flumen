import torch
from torch.utils.data import DataLoader

from . import CausalFlowModel
from .train import EarlyStopping

from .experiment import Experiment, instantiate_model

from argparse import ArgumentParser, ArgumentTypeError

import numpy as np
from scipy.linalg import sqrtm, inv


def whiten_targets(data):
    mean = data[0].state.mean(axis=0)
    std = sqrtm(np.cov(data[0].state.T))
    istd = inv(std)

    for d in data:
        d.state[:] = ((d.state - mean) @ istd).type(torch.get_default_dtype())
        d.init_state[:] = ((d.init_state - mean) @ istd).type(
            torch.get_default_dtype())

    return mean, std, istd


def prepare_experiment(data, args):
    train_data, val_data, test_data = data.preprocess()
    train_mean, train_std, train_istd = whiten_targets(
        (train_data, val_data, test_data))

    experiment = Experiment(args,
                            data.dims(), (train_mean, train_std, train_istd),
                            save_root=args.write_dir)

    model: CausalFlowModel = instantiate_model(args, train_data.state_dim,
                                               train_data.control_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=args.sched_patience,
        cooldown=args.sched_cooldown,
        factor=1. / args.sched_factor)

    mse_loss = torch.nn.MSELoss().to(device)

    early_stop = EarlyStopping(es_patience=args.es_patience,
                               es_delta=args.es_delta)

    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    return (experiment, model, mse_loss, optimizer, sched, early_stop,
            train_dl, val_dl, test_dl, device, args.n_epochs)


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
