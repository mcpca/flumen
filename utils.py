import torch

from argparse import ArgumentParser, ArgumentTypeError

import sys, uuid, subprocess, time, datetime, os
from inspect import cleandoc
from shlex import quote


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

    ap.add_argument('--examples_per_traj',
                    type=positive_int,
                    help="Number of training examples per trajectory",
                    default=25)

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

    ap.add_argument('--control_rnn_depth',
                    type=positive_int,
                    help="Depth of the RNN",
                    default=1)

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
                    help="Subdirectory where the model will be saved",
                    default=None)

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


def timestamp_str(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime(
        '%Y%m%d_%H%M%S') if timestamp else "N/A"


def save_path(dir, timestamp, train_id):
    file_name = dir + '_' + timestamp_str(timestamp) + '_' + str(train_id.hex)

    path = os.path.join(os.path.dirname(__file__), 'outputs', dir)

    return path, file_name


class TrainedModel:

    def __init__(self, args, generator, train_data_mean, train_data_std):
        self.model = None

        self.train_id = uuid.uuid4()

        self.cmd = ' '.join((quote(arg) for arg in sys.argv))
        self.args = args

        self.generator = generator
        self.td_mean = train_data_mean
        self.td_std = train_data_std

        self.save_timestamp = None

        try:
            self.git_head = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except:
            self.git_head = None

        if args.save_model:
            self.save_path, self.file_name = save_path(args.save_model,
                                                       time.time(),
                                                       self.train_id)
            os.makedirs(self.save_path, exist_ok=True)
        else:
            self.save_path = None
            self.file_name = None

        if args.load_data:
            self.data_path = os.path.abspath(args.load_data)
        else:
            self.data_path = None

        self.n_epochs = 0
        self.train_loss = []
        self.val_loss = []

        self.train_loss_best = None
        self.val_loss_best = None

        self.train_time = 0

    def register_progress(self, train, val, best):
        self.n_epochs += 1
        self.train_loss.append(train)
        self.val_loss.append(val)

        if best:
            self.train_loss_best = train
            self.val_loss_best = val

    def save_model(self, model):
        if self.save_path:
            self.save_timestamp = time.time()
            torch.save(
                model.state_dict(),
                os.path.join(self.save_path, self.file_name + '_params.pth'))

    def save(self, train_time):
        self.train_time = train_time

        if self.save_path:
            self.save_timestamp = time.time()
            torch.save(self,
                       os.path.join(self.save_path, self.file_name + '.pth'))

    def predict(self, t, x0, u):
        y_pred = self.model(t, x0, u).numpy()
        y_pred[:] = self.td_mean + y_pred @ self.td_std

        return y_pred

    def args_str(self):
        out_str = ""

        for k, v in vars(self.args):
            out_str += f"{k}: {v}\n"

        return out_str

    def __str__(self):
        return cleandoc(f'''\
            --- Trained model   {self.file_name}
                Timestamp:      {timestamp_str(self.save_timestamp)}
                Git hash:       {self.git_head if self.git_head else 'N/A'}
                Command line:   {self.cmd}
                Data:           {self.data_path if self.data_path else 'N/A'}
                Train time:     {self.train_time:.2f}
                Loss:           t{self.train_loss_best:>3e} // v{self.val_loss_best:>3e}
        ''')
