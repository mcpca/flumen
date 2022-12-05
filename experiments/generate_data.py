from torch import save

from dynamics import Dynamics
from sequence_generators import SequenceGenerator

from flow_model_odedata.data import TrajectoryDataGenerator

from argparse import ArgumentParser, ArgumentTypeError


def percentage(value):
    value = int(value)

    if not (0 <= value <= 100):
        raise ArgumentTypeError(f"{value} is not a valid percentage")

    return value


def parse_args():
    ap = ArgumentParser()

    ap.add_argument('save_path',
                    type=str,
                    help="Path to which the data will be written")

    ap.add_argument('--control_delta',
                    type=float,
                    help="Control sampling rate",
                    default=0.5)

    ap.add_argument('--time_horizon',
                    type=float,
                    help="Time horizon",
                    default=10.)

    ap.add_argument('--n_trajectories',
                    type=int,
                    help="Number of trajectories to sample",
                    default=100)

    ap.add_argument('--n_samples',
                    type=int,
                    help="Number of state samples per trajectory",
                    default=50)

    ap.add_argument('--noise_std',
                    type=float,
                    help="Standard deviation of measurement noise",
                    default=0.0)

    ap.add_argument('--noise_seed',
                    type=int,
                    help="Measurement noise seed",
                    default=None)

    ap.add_argument(
        '--data_split',
        nargs=2,
        type=percentage,
        help="Percentage of data used for validation and test sets",
        default=[20, 20])

    return ap.parse_args()


def generate(args,
             dynamics: Dynamics,
             seq_gen: SequenceGenerator,
             initial_state_generator=None):
    data_generator = TrajectoryDataGenerator(
        dynamics,
        seq_gen,
        control_delta=args.control_delta,
        noise_std=args.noise_std,
        n_trajectories=args.n_trajectories,
        n_samples=args.n_samples,
        time_horizon=args.time_horizon,
        split=args.data_split,
        initial_state_generator=initial_state_generator)

    data = data_generator.generate()
    save(data, f'outputs/{args.save_path}')
