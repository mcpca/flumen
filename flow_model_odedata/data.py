import torch

from flow_model import RawTrajectoryDataset, TrajectoryDataset
from .trajectory_generator import TrajectoryGenerator, Dynamics, SequenceGenerator

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
                                          n_trajectories=n,
                                          n_samples=self.n_samples,
                                          time_horizon=self.time_horizon)
            for n in self.n_trajectories)


class TrajectoryDataWrapper:

    def __init__(self, generator):
        self.generator = generator
        (self.train_data, self.val_data,
         self.test_data) = generator._generate_raw()

    def preprocess(self):
        return (TrajectoryDataset(self.train_data),
                TrajectoryDataset(self.val_data),
                TrajectoryDataset(self.test_data))
