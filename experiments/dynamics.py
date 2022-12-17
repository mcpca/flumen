import numpy as np


class Dynamics:

    def __init__(self, state_dim, control_dim):
        self.n = state_dim
        self.m = control_dim

    def __call__(self, x, u):
        return self._dx(x, u)

    def dims(self):
        return (self.n, self.m)


class LinearSys(Dynamics):

    def __init__(self, a, b):
        super().__init__(a.shape[0], b.shape[1])

        self.a = a
        self.b = b

    def _dx(self, x, u):
        return self.a @ x + self.b @ u


class VanDerPol(Dynamics):

    def __init__(self, damping):
        super().__init__(2, 1)
        self.damping = damping

    def _dx(self, x, u):
        p, v = x

        dp = v
        dv = -p + self.damping * (1 - p**2) * v + u

        return (dp, dv)


class FitzHughNagumo(Dynamics):

    def __init__(self, tau, a, b):
        super().__init__(2, 1)
        self.tau = tau
        self.a = a
        self.b = b

    def _dx(self, x, u):
        v, w = x

        dv = 50 * (v - v**3 - w + u)
        dw = (v - self.a - self.b * w) / self.tau

        return (dv, dw)


class Pendulum(Dynamics):

    def __init__(self, damping, freq=2 * np.pi):
        super().__init__(2, 1)
        self.damping = damping
        self.freq2 = freq**2

    def _dx(self, x, u):
        p, v = x

        dp = v
        dv = -self.freq2 * np.sin(p) - self.damping * v + u

        return (dp, dv)
