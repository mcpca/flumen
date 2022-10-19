import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):

    def __init__(self, generator, n_trajectories, n_samples, time_horizon):
        self.generator = generator

        self.n_trajectories = n_trajectories
        self.n_samples = n_samples
        self.time_horizon = time_horizon

        self.state_dim, self.control_dim = generator._dyn.dims()

        n_examples = n_trajectories * n_samples

        self.init_state = torch.empty(
            (n_examples, self.state_dim)).type(torch.get_default_dtype())
        self.state = torch.empty(
            (n_examples, self.state_dim)).type(torch.get_default_dtype())
        self.time = torch.empty(
            (n_examples, 1)).type(torch.get_default_dtype())
        rnn_input_data = []
        seq_len_data = []

        for k_tr in range(n_trajectories):
            x0, t, y, u = generator.get_example(time_horizon, n_samples)
            u = torch.from_numpy(u)

            self.init_state[k_tr * n_samples:(k_tr + 1) *
                            n_samples] = torch.from_numpy(x0)

            for k_s in range(n_samples):
                self.state[k_tr * n_samples + k_s] = torch.from_numpy(y[k_s])
                self.time[k_tr * n_samples +
                          k_s] = torch.from_numpy(t[k_s] -
                                                  self.generator._init_time)

                rnn_input, rnn_input_len = self.process_example(0, k_s, t, u)

                seq_len_data.append(rnn_input_len)
                rnn_input_data.append(rnn_input)

        self.rnn_input = torch.stack(rnn_input_data).type(
            torch.get_default_dtype())
        self.seq_lens = torch.tensor(seq_len_data, dtype=torch.long)
        self.len = self.seq_lens.shape[0]

    def process_example(self, start_idx, end_idx, t, u):
        init_time = self.generator._init_time
        delta = self.generator._delta

        u_start_idx = int(np.floor((t[start_idx] - init_time) / delta))
        u_end_idx = int(np.floor((t[end_idx] - init_time) / delta))
        u_sz = 1 + u_end_idx - u_start_idx

        u_seq = torch.zeros_like(u)
        u_seq[0:u_sz] = u[u_start_idx:(u_end_idx + 1)]

        deltas = torch.ones_like(u_seq)
        t_u_end = init_time + delta * u_end_idx
        t_u_start = init_time + delta * u_start_idx

        if u_sz > 1:
            deltas[0] = (1. - (t[start_idx] - t_u_start) / delta).item()
            deltas[u_sz - 1] = ((t[end_idx] - t_u_end) / delta).item()
        else:
            deltas[0] = ((t[end_idx] - t[start_idx]) / delta).item()

        deltas[u_sz:] = 0.

        rnn_input = torch.hstack((u_seq, deltas))

        return rnn_input, u_sz

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.init_state[index], self.time[index], self.state[index],
                self.rnn_input[index], self.seq_lens[index])
