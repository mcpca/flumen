import numpy as np
from scipy.integrate import solve_ivp
from pyDOE import lhs
from torch.utils.data import Dataset


class TrajectoryGenerator:
    def __init__(self,
                 state_dimension,
                 dynamics,
                 control_delta,
                 method='RK45'):
        self._n = state_dimension
        self._ode_method = method
        self._dyn = dynamics
        self._rng = np.random.default_rng()
        self._delta = control_delta  # control sampling time

        self._init_time = 0.

    def get_example(self, time_horizon, n_samples):
        y0 = self._rng.standard_normal(size=self._n)

        n_control_vals = int(1 + np.floor((time_horizon - self._init_time) /
                                          self._delta))
        control_seq = self._rng.standard_normal(size=(n_control_vals, 1))

        def f(t, y):
            n_control = int(np.floor((t - self._init_time) / self._delta))
            u = control_seq[n_control]  # get u(t)

            return self._dyn(y, u)

        t_samples = np.sort(self._init_time +
                            (time_horizon - self._init_time) *
                            lhs(1, n_samples).ravel())

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


class TrajectoryDataset(Dataset):
    def __init__(self, generator: TrajectoryGenerator, n_trajectories,
                 n_samples, time_horizon):

        self._raw_data = [
            generator.get_example(time_horizon, n_samples)
            for _ in range(n_trajectories)
        ]

    def __len__(self):
        return len(self._raw_data)

    def __getitem__(self, index):
        init_state, time, trajectory, control = self._raw_data[index]

        return init_state, time, trajectory, control
