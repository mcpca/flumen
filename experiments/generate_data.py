from torch import save

from dynamics import Dynamics
from sequence_generators import SequenceGenerator
from trajectory_sampler import TrajectorySampler

from argparse import ArgumentParser, ArgumentTypeError

from flow_model import RawTrajectoryDataset, TrajectoryDataset


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
        n_trajectories=args.n_trajectories,
        n_samples=args.n_samples,
        time_horizon=args.time_horizon,
        split=args.data_split,
        noise_std=args.noise_std,
        initial_state_generator=initial_state_generator)

    data = data_generator.generate()
    save(data, f'outputs/{args.save_path}')


class TrajectoryDataGenerator:

    def __init__(self, dynamics: Dynamics,
                 control_generator: SequenceGenerator, control_delta,
                 noise_std, initial_state_generator, n_trajectories, n_samples,
                 time_horizon, split):
        if split[0] + split[1] >= 100:
            raise Exception("Invalid data split.")

        self.split = split
        self.control_delta = control_delta
        self.n_samples = n_samples
        self.time_horizon = time_horizon
        self.noise_std = noise_std

        self.sampler = TrajectorySampler(
            time_horizon,
            n_samples,
            dynamics,
            control_delta=control_delta,
            control_generator=control_generator,
            initial_state_generator=initial_state_generator)

        n_val_t = int(n_trajectories * (split[0] / 100.))
        n_test_t = int(n_trajectories * (split[1] / 100.))
        n_train_t = n_trajectories - n_val_t - n_test_t

        self.n_trajectories = (n_train_t, n_val_t, n_test_t)

    def generate(self):
        return TrajectoryDataWrapper(self)

    def _generate_raw(self):
        return tuple(
            RawTrajectoryDataset.generate(
                self.sampler, n_trajectories=n, noise_std=self.noise_std)
            for n in self.n_trajectories)


class TrajectoryDataWrapper:

    def __init__(self, generator):
        self.generator = generator
        (self.train_data, self.val_data,
         self.test_data) = generator._generate_raw()

    def dims(self):
        return (self.train_data.state_dim, self.train_data.control_dim)

    def preprocess(self):
        return (TrajectoryDataset(self.train_data),
                TrajectoryDataset(self.val_data),
                TrajectoryDataset(self.test_data))
