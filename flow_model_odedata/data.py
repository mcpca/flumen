import torch

from torch.utils.data import DataLoader

from flow_model import TrajectoryDataset
from .trajectory_generator import TrajectoryGenerator, Dynamics, SequenceGenerator

import numpy as np
from scipy.linalg import sqrtm, inv


class ODEExperimentData:

    def __init__(self, dynamics: Dynamics,
                 control_generator: SequenceGenerator, control_delta,
                 n_trajectories, n_samples, time_horizon, split):
        if split[0] + split[1] >= 100:
            raise Exception("Invalid data split.")

        trajectory_generator = TrajectoryGenerator(
            dynamics,
            control_delta=control_delta,
            control_generator=control_generator)

        n_val_t = int(n_trajectories * (split[0] / 100.))
        n_test_t = int(n_trajectories * (split[1] / 100.))
        n_train_t = n_trajectories - n_val_t - n_test_t

        self.n_trajectories = (n_train_t, n_val_t, n_test_t)

        self.data = (TrajectoryDataset(trajectory_generator,
                                       n_trajectories=n,
                                       n_samples=n_samples,
                                       time_horizon=time_horizon)
                     for n in self.n_trajectories)

    def whiten_targets(self):
        mean = self.train_mean = self.data[0].state.mean(axis=0)
        self.train_std = sqrtm(np.cov(self.data[0].state.T))
        istd = self.train_std_inv = inv(self.train_std)

        for d in self.data:
            d.state[:] = ((d.state - mean) @ istd).type(
                torch.get_default_dtype())
            d.init_state[:] = ((d.init_state - mean) @ istd).type(
                torch.get_default_dtype())

    def add_noise(self, noise_std, rng):
        for data in self.data:
            traj_noise = rng.normal(loc=0.0,
                                    scale=noise_std,
                                    size=(len(data), data.state_dim))

            init_state_noise = rng.normal(loc=0.0,
                                          scale=noise_std,
                                          size=(len(data), data.state_dim))

            data.state[:] += traj_noise
            data.init_state[:] += init_state_noise

    def get_metadata(self):
        return {
            "n_train_trajectories": self.n_trajectories[0],
            "n_val_trajectories": self.n_trajectories[1],
            "n_test_trajectories": self.n_trajectories[2],
            "train_data_mean": self.train_data_mean,
            "train_data_std": self.train_data_std,
            "train_data_std_inv": self.train_data_std_inv,
            "trajectory_generator": self.trajectory_generator,
        }

    def make_dataloaders(self, batch_size):
        shuffle = [True, False, False]
        return (DataLoader(data, batch_size=batch_size, shuffle=s)
                for (data, s) in zip(self.data, shuffle))
