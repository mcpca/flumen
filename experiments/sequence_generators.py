import numpy as np


class SequenceGenerator:

    def __init__(self, rng: np.random.Generator = None):
        self._rng = rng if rng else np.random.default_rng()

    def sample(self, time_range, delta):
        return self._sample_impl(time_range, delta)


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


class UniformSqWave(SequenceGenerator):

    def __init__(self, period, min=0., max=1., rng=None):
        super(UniformSqWave, self).__init__(rng)

        self._period = period
        self._min = min
        self._max = max

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))

        n_amplitude_vals = int(np.ceil(n_control_vals / self._period))

        amp_seq = self._rng.uniform(low=self._min,
                                    high=self._max,
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

    def __init__(self, max_freq=1.0, rng=None):
        super(SinusoidalSequence, self).__init__(rng)

        self._amp_mean = 1.0
        self._amp_std = 1.0
        self._mf = max_freq

    def _sample_impl(self, time_range, delta):
        amplitude = self._rng.lognormal(mean=self._amp_mean,
                                        sigma=self._amp_std)
        frequency = self._rng.uniform(0, self._mf)

        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))
        time = np.linspace(time_range[0], time_range[1], n_control_vals)

        return (amplitude * np.sin(2 * np.pi * frequency * time)).reshape(
            (-1, 1))
