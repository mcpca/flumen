class Dynamics:
    def __init__(self, state_dim, control_dim):
        self.n = state_dim
        self.m = control_dim

    def __call__(self, x, u):
        return self._dx(x, u)


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
        dv = -p + (self.damping**2) * (1 - p**2) * v

        return (dp, dv)
