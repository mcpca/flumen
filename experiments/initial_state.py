import numpy as np


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


class UniformInitialState(InitialStateGenerator):

    def __init__(self, n, rng: np.random.Generator = None):
        self.rng = rng if rng else np.random.default_rng()
        self.n = n

    def _sample_impl(self):
        return self.rng.uniform(size=self.n)


class HHFSInitialState(InitialStateGenerator):

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng if rng else np.random.default_rng()

    def _sample_impl(self):
        x0 = self.rng.uniform(size=(4, ))
        x0[0] = 2. * x0[0] - 1.

        return x0


class HHRSAInitialState(InitialStateGenerator):

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng if rng else np.random.default_rng()

    def _sample_impl(self):
        x0 = self.rng.uniform(size=(5, ))
        x0[0] = 2. * x0[0] - 1.

        return x0


class HHFFEInitialState(InitialStateGenerator):

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng if rng else np.random.default_rng()

    def _sample_impl(self):
        x0 = self.rng.uniform(size=(10, ))
        x0[0] = 2. * x0[0] - 1.
        x0[5] = 2. * x0[5] - 1.

        return x0


class HHFBEInitialState(InitialStateGenerator):

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng if rng else np.random.default_rng()

    def _sample_impl(self):
        x0 = self.rng.uniform(size=(11, ))
        x0[0] = 2. * x0[0] - 1.
        x0[5] = 2. * x0[5] - 1.

        return x0


class HHIBInitialState(InitialStateGenerator):

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng if rng else np.random.default_rng()

    def _sample_impl(self):
        x0 = self.rng.uniform(size=(7, ))
        x0[0] = 2. * x0[0] - 1.

        return x0


class GreenshieldsInitialState(InitialStateGenerator):

    def __init__(self, n_cells, n_sections, rng: np.random.Generator = None):
        self.rng = rng if rng else np.random.default_rng()

        self.n_cells = n_cells
        self.n_sec = n_sections
        self.sec_size = self.n_cells // self.n_sec

    def _sample_impl(self):
        x0_vals = self.rng.uniform(0., 0.5, size=(self.n_sec, ))
        x0 = np.empty((self.n_cells,))
        x0[0:self.sec_size * self.n_sec] = np.repeat(x0_vals, self.sec_size)
        x0[self.sec_size * self.n_sec:-1] = x0[self.sec_size * self.n_sec-1]

        return x0
