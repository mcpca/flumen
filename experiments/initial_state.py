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


class HHFSInitialState(InitialStateGenerator):

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng if rng else np.random.default_rng()

    def _sample_impl(self):
        v0 = self.rng.normal(scale=1./np.sqrt(100))
        n0 = self.rng.uniform()
        m0 = self.rng.uniform()
        h0 = self.rng.uniform()

        return np.array((v0, n0, m0, h0))
