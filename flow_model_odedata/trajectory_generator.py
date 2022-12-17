import torch
from scipy.integrate import solve_ivp
from pyDOE import lhs
import numpy as np


def pack_model_inputs(x0, t, u, delta):
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


class Dynamics:

    def __init__(self, state_dim, control_dim):
        self.n = state_dim
        self.m = control_dim

    def __call__(self, x, u):
        return self._dx(x, u)

    def dims(self):
        return (self.n, self.m)


class InitialStateGenerator:

    def __init__(self, rng: np.random.Generator = None):
        self._rng = rng if rng else np.random.default_rng()

    def sample(self):
        return self._sample_impl()


class GaussianInitialState(InitialStateGenerator):

    def __init__(self, n, rng: np.random.Generator = None):
        self.rng = rng if rng else np.random.default_rng()
        self.n = n

    def _sample_impl(self):
        return self.rng.standard_normal(size=self.n)


class SequenceGenerator:

    def __init__(self, rng: np.random.Generator = None):
        self._rng = rng if rng else np.random.default_rng()

    def sample(self, time_range, delta):
        return self._sample_impl(time_range, delta)


class TrajectoryGenerator:

    def __init__(self,
                 time_horizon,
                 n_samples,
                 dynamics: Dynamics,
                 control_delta,
                 control_generator: SequenceGenerator,
                 noise_std,
                 initial_state_generator: InitialStateGenerator = None,
                 method='RK45'):
        self._n = dynamics.n
        self._ode_method = method
        self._dyn = dynamics
        self._delta = control_delta  # control sampling time
        self._seq_gen = control_generator

        self.time_horizon = time_horizon
        self.n_samples = n_samples

        self.state_generator = (initial_state_generator
                                if initial_state_generator else
                                GaussianInitialState(self._n))

        self._rng = np.random.default_rng()
        self._noise_std = noise_std

        self._init_time = 0.

    def dims(self):
        return self._dyn.dims()

    def get_example(self):
        y0 = self.state_generator.sample()

        control_seq = self._seq_gen.sample(time_range=(self._init_time,
                                                       self.time_horizon),
                                           delta=self._delta)

        def f(t, y):
            n_control = int(np.floor((t - self._init_time) / self._delta))
            u = control_seq[n_control]  # get u(t)

            return self._dyn(y, u)

        t_samples = self._init_time + (self.time_horizon - self._init_time
                                       ) * lhs(1, self.n_samples).ravel()
        t_samples = np.append(t_samples, [self._init_time])
        t_samples = np.sort(t_samples)

        traj = solve_ivp(
            f,
            (self._init_time, self.time_horizon),
            y0,
            t_eval=t_samples,
            method=self._ode_method,
            rtol=1e-9,
        )

        y = traj.y.T
        y += self._rng.normal(loc=0.0, scale=self._noise_std, size=y.shape)
        t = traj.t.reshape(-1, 1)

        return y0, t, y, control_seq
