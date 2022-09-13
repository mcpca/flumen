class Dynamics:
    def __init__(self, state_dim, control_dim):
        self.n = state_dim
        self.m = control_dim

    def __call__(self, x, u):
        return self._dx(x, u)

    def dims(self):
        return (self.n, self.m)
