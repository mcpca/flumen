import torch
from scipy.integrate import solve_ivp
from pyDOE import lhs
import numpy as np

from dynamics import Dynamics
from sequence_generators import SequenceGenerator
from initial_state import InitialStateGenerator, GaussianInitialState


class TrajectorySampler:

    def __init__(self,
                 time_horizon,
                 n_samples,
                 dynamics: Dynamics,
                 control_delta,
                 control_generator: SequenceGenerator,
                 method,
                 initial_state_generator: InitialStateGenerator = None):
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

        self._init_time = 0.

    def dims(self):
        return self._dyn.dims()

    def get_example(self, time_horizon=None, n_samples=None):
        y0 = self.state_generator.sample()

        time_horizon = time_horizon if time_horizon else self.time_horizon
        n_samples = n_samples if n_samples else self.n_samples

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
        )

        y = traj.y.T
        t = traj.t.reshape(-1, 1)

        return y0, t, y, control_seq
