from flow_model import RawTrajectoryDataset, TrajectoryDataset
from .trajectory_generator import TrajectoryGenerator, Dynamics, SequenceGenerator


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

        self.trajectory_generator = TrajectoryGenerator(
            time_horizon,
            n_samples,
            dynamics,
            control_delta=control_delta,
            control_generator=control_generator,
            noise_std=noise_std,
            initial_state_generator=initial_state_generator)

        n_val_t = int(n_trajectories * (split[0] / 100.))
        n_test_t = int(n_trajectories * (split[1] / 100.))
        n_train_t = n_trajectories - n_val_t - n_test_t

        self.n_trajectories = (n_train_t, n_val_t, n_test_t)

    def generate(self):
        return TrajectoryDataWrapper(self)

    def _generate_raw(self):
        return tuple(
            RawTrajectoryDataset.generate(self.trajectory_generator,
                                          n_trajectories=n)
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
