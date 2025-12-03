import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from math import floor


class RawTrajectoryDataset(Dataset):
    n_traj: int
    state_dim: int
    control_dim: int
    output_dim: int
    mask: tuple[int, ...]
    init_state: Tensor
    init_state_noise: Tensor
    time: list[Tensor]
    state: list[Tensor]
    state_noise: list[Tensor]
    control_seq: list[Tensor]

    def __init__(
        self,
        data: list[dict],
        state_dim: int,
        control_dim: int,
        output_dim: int,
        delta: float,
        output_mask: tuple[int, ...],
        noise_std: float = 0.0,
    ):
        self.n_traj = len(data)
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        self.delta = delta
        self.mask = output_mask

        self.init_state = torch.empty((self.n_traj, self.state_dim)).type(
            torch.get_default_dtype()
        )
        self.init_state_noise = torch.empty((self.n_traj, self.state_dim)).type(
            torch.get_default_dtype()
        )

        self.time = []
        self.state = []
        self.state_noise = []
        self.control_seq = []

        for k, sample in enumerate(data):
            self.init_state[k] = torch.from_numpy(
                sample["init_state"].reshape((1, self.state_dim))
            )

            self.init_state_noise[k] = 0.0

            self.time.append(
                torch.from_numpy(sample["time"])
                .type(torch.get_default_dtype())
                .reshape((-1, 1))
            )

            self.state.append(
                torch.from_numpy(sample["state"])
                .type(torch.get_default_dtype())
                .reshape((-1, self.state_dim))
            )

            self.state_noise.append(
                torch.normal(
                    mean=0.0, std=noise_std, size=self.state[-1].size()
                )
            )

            self.control_seq.append(
                torch.from_numpy(sample["control"])
                .type(torch.get_default_dtype())
                .reshape((-1, self.control_dim))
            )

    def __len__(self):
        return self.n_traj

    def __getitem__(self, index):
        return (
            self.init_state[index] + self.init_state_noise[index],
            self.time[index],
            self.state[index] + self.state_noise[index],
            self.control_seq[index],
        )


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        raw_data: RawTrajectoryDataset,
        max_seq_len: int = -1,
        n_samples: int = 1,
    ):
        self.state_dim = raw_data.state_dim
        self.control_dim = raw_data.control_dim
        self.output_dim = raw_data.output_dim
        self.delta = raw_data.delta

        mask = tuple(bool(v) for v in raw_data.mask)

        init_state = []
        state = []
        rnn_input_data = []
        tau_data = []
        seq_len_data = []

        rng = np.random.default_rng()

        for x0, t, y, u in raw_data:
            if max_seq_len == -1:
                for k_s, y_s in enumerate(y):
                    rnn_input, tau, rnn_input_len = make_rnn_inputs(
                        0, k_s, t, u, self.delta
                    )

                    s = y_s.view(1, -1)[:, mask].reshape(-1)

                    init_state.append(x0)
                    state.append(s)
                    seq_len_data.append(rnn_input_len)
                    rnn_input_data.append(rnn_input)
                    tau_data.append(tau)

            else:
                for k_s, y_s in enumerate(y):
                    # find index of last relevant state sample
                    times = t - t[k_s] - max_seq_len * self.delta
                    times[times > 0] = 0.0
                    k_l = times.argmax().item()

                    if k_l == k_s:
                        end_idxs = (0,)
                    else:
                        end_idxs = rng.choice(
                            k_l - k_s,
                            size=min(n_samples, k_l - k_s),
                            replace=False,
                        )

                    for k_e in end_idxs:
                        rnn_input, tau, rnn_input_len = make_rnn_inputs(
                            k_s, k_s + k_e, t, u, self.delta
                        )

                        init_state.append(y_s)
                        state.append(y[k_s + k_e, mask])
                        seq_len_data.append(rnn_input_len)
                        rnn_input_data.append(rnn_input)
                        tau_data.append(tau)

        self.init_state = torch.stack(init_state).type(
            torch.get_default_dtype()
        )

        self.state = torch.stack(state).type(torch.get_default_dtype())

        self.rnn_input = torch.stack(rnn_input_data).type(
            torch.get_default_dtype()
        )

        self.tau = torch.stack(tau_data).type(torch.get_default_dtype())

        self.seq_lens = torch.tensor(seq_len_data, dtype=torch.long)

        self.len = len(init_state)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (
            self.init_state[index],
            self.state[index],
            self.rnn_input[index],
            self.tau[index],
            self.seq_lens[index],
        )


def make_rnn_inputs(
    start_idx: int, end_idx: int, t: Tensor, u: Tensor, delta: float
) -> tuple[Tensor, Tensor, int]:
    init_time = 0.0

    u_start_idx = floor((t[start_idx] - init_time) / delta)
    u_end_idx = floor((t[end_idx] - init_time) / delta)
    u_sz = 1 + u_end_idx - u_start_idx

    u_seq = torch.zeros_like(u)
    u_seq[0:u_sz] = u[u_start_idx : (u_end_idx + 1)]

    tau_seq = torch.ones((u_seq.shape[0], 1))
    t_u_end = init_time + delta * u_end_idx
    t_u_start = init_time + delta * u_start_idx

    if u_sz > 1:
        tau_seq[0] = (1.0 - (t[start_idx] - t_u_start) / delta).item()
        tau_seq[u_sz - 1] = ((t[end_idx] - t_u_end) / delta).item()
    else:
        tau_seq[0] = ((t[end_idx] - t[start_idx]) / delta).item()

    tau_seq[u_sz:] = 0.0

    rnn_input = torch.hstack((u_seq, tau_seq))

    return rnn_input, tau_seq, u_sz
