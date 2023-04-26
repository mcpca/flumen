import numpy as np
import torch
from torch.utils.data import Dataset


class RawTrajectoryDataset(Dataset):

    def __init__(self,
                 data,
                 state_dim,
                 control_dim,
                 delta,
                 noise_std=0.,
                 **kwargs):
        self.__dict__.update(kwargs)

        n_traj = len(data)
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.delta = delta

        self.init_state = torch.empty(
            (n_traj, self.state_dim)).type(torch.get_default_dtype())
        self.init_state_noise = torch.empty(
            (n_traj, self.state_dim)).type(torch.get_default_dtype())
        self.time = []
        self.state = []
        self.state_noise = []
        self.control_seq = []

        for k, sample in enumerate(data):
            self.init_state[k] = torch.from_numpy(sample["init_state"].reshape(
                (1, self.state_dim)))
            self.init_state_noise[k] = 0.
            self.time.append(
                torch.from_numpy(sample["time"]).type(
                    torch.get_default_dtype()).reshape((-1, 1)))

            self.state.append(
                torch.from_numpy(sample["state"]).type(
                    torch.get_default_dtype()).reshape((-1, self.state_dim)))

            self.state_noise.append(
                torch.normal(mean=0.,
                             std=noise_std,
                             size=self.state[-1].size()))

            self.control_seq.append(
                torch.from_numpy(sample["control"]).type(
                    torch.get_default_dtype()).reshape((-1, self.control_dim)))

    @classmethod
    def generate(cls, generator, time_horizon, n_trajectories, n_samples,
                 noise_std):

        def get_example():
            x0, t, y, u = generator.get_example(time_horizon, n_samples)
            return {
                "init_state": x0,
                "time": t,
                "state": y,
                "control": u,
            }

        data = [get_example() for _ in range(n_trajectories)]

        return cls(data,
                   *generator.dims(),
                   delta=generator._delta,
                   generator=generator,
                   noise_std=noise_std)

    def __len__(self):
        return (self.init_state)

    def __getitem__(self, index):
        return (self.init_state[index], self.init_state_noise[index],
                self.time[index], self.state[index], self.state_noise[index],
                self.control_seq[index])


class TrajectoryDataset(Dataset):

    def __init__(self, raw_data: RawTrajectoryDataset):
        self.state_dim = raw_data.state_dim
        self.control_dim = raw_data.control_dim
        self.delta = raw_data.delta
        self.len = sum(len(s) for (_, _, _, s, _, _) in raw_data)

        self.init_state = torch.empty((self.len, self.state_dim))
        self.state = torch.empty((self.len, self.state_dim))
        self.control_len = torch.empty((self.len, ), dtype=torch.int64)
        self.control_delta = torch.empty((self.len, ))

        max_control_len = max(len(u) for (_, _, _, _, _, u) in raw_data)

        self.u = torch.zeros((self.len, max_control_len, self.control_dim))

        k_tr = 0

        for (x0, x0_n, t, y, y_n, u) in raw_data:
            y += y_n
            self.init_state[k_tr:k_tr + len(y)] = x0 + x0_n
            self.u[k_tr:k_tr + len(y), :len(u)] = u[:]

            for k_s, y_s in enumerate(y):
                self.state[k_tr + k_s] = y_s
                control_len = 1 + int(np.floor((t[k_s] - t[0]) / self.delta))
                self.control_len[k_tr + k_s] = control_len
                self.control_delta[k_tr +
                                   k_s] = (t[k_s] - self.delta *
                                           (control_len - 1)) / self.delta

            k_tr += len(y)

    @staticmethod
    def process_example(end_idx, t, u, delta):
        start_idx = 0
        init_time = 0.
        n_steps = 2

        u_start_idx = int(np.floor((t[start_idx] - init_time) / delta))
        u_end_idx = int(np.floor((t[end_idx] - init_time) / delta))
        u_sz = 1 + u_end_idx - u_start_idx
        seq_sz = 1 + n_steps * (u_end_idx - u_start_idx)
        total_sz = 1 + n_steps * u.shape[0]

        u_seq = torch.zeros((total_sz, u.shape[1]))
        # u_seq[0:seq_sz] = u[u_start_idx:(u_end_idx + 1)]

        deltas = torch.ones_like(u_seq)
        t_u_end = init_time + delta * u_end_idx
        t_u_start = init_time + delta * u_start_idx

        if seq_sz > 1:
            # deltas[0] = (1. - (t[start_idx] - t_u_start) / delta).item()
            # u_seq[0] = u[u_start_idx]

            for k in range(u_sz - 1):
                deltas[2 * k] = min(1.1 * torch.rand((1, )), 1)
                deltas[2 * k + 1] = 1 - deltas[2 * k]
                u_seq[2 * k] = u[k]
                u_seq[2 * k + 1] = u[k]

            deltas[seq_sz - 1] = ((t[end_idx] - t_u_end) / delta).item()
            u_seq[seq_sz - 1] = u[u_end_idx]
        else:
            deltas[0] = ((t[end_idx] - t[start_idx]) / delta).item()

        deltas[seq_sz:] = 0.  # zero out elements after last control

        rnn_input = torch.hstack((u_seq, deltas))

        return rnn_input, seq_sz

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.init_state[index], self.state[index], self.u[index],
                self.control_len[index], self.control_delta[index])
