import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm, inv
from pyDOE import lhs
import torch
from torch.utils.data import Dataset


class SequenceGenerator:

    def __init__(self, rng: np.random.Generator = None):
        self._rng = rng if rng else np.random.default_rng()

    def sample(self, time_range, delta):
        return self._sample_impl(time_range, delta)


class TrajectoryGenerator:

    def __init__(self,
                 dynamics,
                 control_delta,
                 control_generator: SequenceGenerator,
                 method='RK45'):
        self._n = dynamics.n
        self._ode_method = method
        self._dyn = dynamics
        self._rng = np.random.default_rng()
        self._delta = control_delta  # control sampling time
        self._seq_gen = control_generator

        self._init_time = 0.

    def get_example(self, time_horizon, n_samples):
        y0 = self._rng.standard_normal(size=self._n)

        control_seq = self._seq_gen.sample(time_range=(self._init_time,
                                                       time_horizon),
                                           delta=self._delta)

        def f(t, y):
            n_control = int(np.floor((t - self._init_time) / self._delta))
            u = control_seq[n_control]  # get u(t)

            return self._dyn(y, u)

        t_samples = self._init_time + (time_horizon - self._init_time) * lhs(
            1, n_samples).ravel()
        t_samples = np.append(t_samples, [self._init_time])
        t_samples = np.sort(t_samples)

        traj = solve_ivp(
            f,
            (self._init_time, time_horizon),
            y0,
            t_eval=t_samples,
            method=self._ode_method,
            rtol=1e-9,
        )

        y = traj.y.T
        t = traj.t.reshape(-1, 1)

        return y0, t, y, control_seq


class TrajectoryDataset(Dataset):

    def __init__(self, generator: TrajectoryGenerator, n_trajectories,
                 n_samples, time_horizon, examples_per_traj):

        init_time = generator._init_time
        delta = generator._delta

        n_examples = n_trajectories * examples_per_traj

        self.init_state = torch.empty(
            (n_examples, generator._n)).type(torch.get_default_dtype())
        self.state = torch.empty(
            (n_examples, generator._n)).type(torch.get_default_dtype())
        self.time = torch.empty(
            (n_examples, 1)).type(torch.get_default_dtype())
        rnn_input_data = []
        seq_len_data = []

        rng = np.random.default_rng()

        for k_tr in range(n_trajectories):
            x0, t, y, u = generator.get_example(time_horizon, n_samples)
            u = torch.from_numpy(u)

            self.init_state[k_tr * examples_per_traj:(k_tr + 1) *
                            examples_per_traj] = torch.from_numpy(x0)

            # generate random indexes from which samples will be taken
            if examples_per_traj < n_samples:
                end_state_idxs = rng.integers(n_samples,
                                              size=(examples_per_traj, ),
                                              dtype=np.uint64)
            else:
                end_state_idxs = np.arange(n_samples, dtype=np.uint64)

            for k_ei, end_idx in enumerate(end_state_idxs):
                self.state[k_tr * examples_per_traj + k_ei] = torch.from_numpy(
                    y[end_idx])
                self.time[k_tr * examples_per_traj +
                          k_ei] = torch.from_numpy(t[end_idx] - init_time)

                rnn_input, rnn_input_len = self.process_example(
                    0, end_idx, t, u, init_time, delta)

                seq_len_data.append(rnn_input_len)
                rnn_input_data.append(rnn_input)

        self.rnn_input = torch.stack(rnn_input_data).type(
            torch.get_default_dtype())
        self.seq_lens = torch.tensor(seq_len_data, dtype=torch.long)
        self.len = self.seq_lens.shape[0]

    @staticmethod
    def process_example(start_idx, end_idx, t, u, init_time, delta):
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

    def whiten_targets(self):
        mean = self.state.mean(axis=0)
        std = sqrtm(np.cov(self.state.T))

        istd = inv(std)

        self.state[:] = (self.state - mean) @ istd
        self.init_state[:] = (self.init_state - mean) @ istd

        return mean, std

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.init_state[index], self.time[index], self.state[index],
                self.rnn_input[index], self.seq_lens[index])


def pack_model_inputs(x0, t, u, delta, mean, std):
    t = torch.Tensor(t.reshape((-1, 1))).flip(0)
    x0 = torch.Tensor(x0.reshape((1, -1))).repeat(t.shape[0], 1)
    rnn_inputs = torch.empty((t.shape[0], u.size, 2))
    lengths = torch.empty((t.shape[0], ), dtype=torch.long)

    for idx, (t_, u_) in enumerate(zip(t, rnn_inputs)):
        control_seq = torch.from_numpy(u)
        deltas = torch.ones_like(control_seq)

        seq_len = 1 + int(np.floor(t_ / delta))
        lengths[idx] = seq_len
        deltas[seq_len - 1] = ((t_ - delta * (seq_len - 1)) / delta).item()
        deltas[seq_len:] = 0.

        u_[:] = torch.hstack((control_seq, deltas))

    u_packed = torch.nn.utils.rnn.pack_padded_sequence(rnn_inputs,
                                                       lengths,
                                                       batch_first=True,
                                                       enforce_sorted=True)

    return x0, t, u_packed


class GaussianSequence(SequenceGenerator):

    def __init__(self, mean=0., std=1., rng=None):
        super(GaussianSequence, self).__init__(rng)

        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))
        control_seq = self._rng.normal(loc=self._mean,
                                       scale=self._std,
                                       size=(n_control_vals, 1))

        return control_seq


class GaussianSqWave(SequenceGenerator):

    def __init__(self, period, mean=0., std=1., rng=None):
        super(GaussianSqWave, self).__init__(rng)

        self._period = period
        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))

        n_amplitude_vals = int(np.ceil(n_control_vals / self._period))

        amp_seq = self._rng.normal(loc=self._mean,
                                   scale=self._std,
                                   size=(n_amplitude_vals, 1))

        control_seq = np.repeat(amp_seq, self._period, axis=0)[:n_control_vals]

        return control_seq


class LogNormalSqWave(SequenceGenerator):

    def __init__(self, period, mean=0., std=1., rng=None):
        super(LogNormalSqWave, self).__init__(rng)

        self._period = period
        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))

        n_amplitude_vals = int(np.ceil(n_control_vals / self._period))

        amp_seq = self._rng.lognormal(mean=self._mean,
                                      sigma=self._std,
                                      size=(n_amplitude_vals, 1))

        control_seq = np.repeat(amp_seq, self._period, axis=0)[:n_control_vals]

        return control_seq


class RandomWalkSequence(SequenceGenerator):

    def __init__(self, mean=0., std=1., rng=None):
        super(RandomWalkSequence, self).__init__(rng)

        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))

        control_seq = np.cumsum(self._rng.normal(loc=self._mean,
                                                 scale=self._std,
                                                 size=(n_control_vals, 1)),
                                axis=1)

        return control_seq


class SinusoidalSequence(SequenceGenerator):

    def __init__(self, rng=None):
        super(SinusoidalSequence, self).__init__(rng)

        self._f_mean = 1.0
        self._f_std = 1.0
        self._amp_mean = 1.0
        self._amp_std = 1.0

    def _sample_impl(self, time_range, delta):
        amplitude = self._rng.lognormal(mean=self._amp_mean,
                                        sigma=self._amp_std)
        frequency = self._rng.lognormal(mean=self._f_mean, sigma=self._f_std)

        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))
        time = np.linspace(time_range[0], time_range[1], n_control_vals)

        return (amplitude * np.sin(frequency * time)).reshape((-1, 1))
