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
        dv = -p + self.damping * (1 - p**2) * v + u.item()

        return (dp, dv)


class FitzHughNagumo(Dynamics):

    def __init__(self, tau, a, b):
        super().__init__(2, 1)
        self.tau = tau
        self.a = a
        self.b = b

    def _dx(self, x, u):
        v, w = x

        dv = 50 * (v - v**3 - w + u.item())
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
        dv = -self.freq2 * np.sin(p) - self.damping * v + u.item()

        return (dp, dv)


class HodgkinHuxleyFS(Dynamics):

    def __init__(self):
        super().__init__(4, 1)

        self.time_scale = 100

        # Parameters follow
        #   A. G. Giannari and A. Astolfi, ‘Model design for networks of
        #   heterogeneous Hodgkin–Huxley neurons’,
        #   Neurocomputing, vol. 496, pp. 147–157, Jul. 2022,
        #   doi: 10.1016/j.neucom.2022.04.115.
        self.c_m = 0.5
        self.v_k = -90.
        self.v_na = 50.
        self.v_l = -70.
        self.v_t = -56.2
        self.g_k = 10.
        self.g_na = 56.
        self.g_l = 1.5e-2

    def _dx(self, x, u):
        v, n, m, h = x

        # denormalise first state variable
        v *= 100.

        dv = (u.item() - self.g_k * n**4 *
              (v - self.v_k) - self.g_na * m**3 * h *
              (v - self.v_na) - self.g_l * (v - self.v_l)) / (100. * self.c_m)

        a_n = -0.032 * (v - self.v_t -
                        15.) / (np.exp(-(v - self.v_t - 15.) / 5.) - 1)
        b_n = 0.5 * np.exp(-(v - self.v_t - 10.) / 40.)
        dn = a_n * (1. - n) - b_n * n

        a_m = -0.32 * (v - self.v_t -
                       13.) / (np.exp(-(v - self.v_t - 13.) / 4.) - 1)
        b_m = 0.28 * (v - self.v_t - 40.) / (np.exp(
            (v - self.v_t - 40.) / 5.) - 1)
        dm = a_m * (1. - m) - b_m * m

        a_h = 0.128 * np.exp(-(v - self.v_t - 17.) / 18.)
        b_h = 4. / (1. + np.exp(-(v - self.v_t - 40.) / 5.))
        dh = a_h * (1. - h) - b_h * h

        return tuple(self.time_scale * dx for dx in (dv, dn, dm, dh))
