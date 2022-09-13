import numpy as np

from flow_model.data import RandomWalkSequence
from dynamics import Dynamics

from utils import parse_args, print_gpu_info
from sim_and_train import sim_and_train


class LinearSys(Dynamics):
    def __init__(self, a, b):
        super().__init__(a.shape[0], b.shape[1])

        self.a = a
        self.b = b

    def _dx(self, x, u):
        return self.a @ x + self.b @ u


def main():
    args = parse_args()

    a_matrix = np.array([[-0.01, 1], [0, -1]])
    b_matrix = np.array([0, 1]).reshape((-1, 1))
    dynamics = LinearSys(a_matrix, b_matrix)
    control_generator = RandomWalkSequence()

    sim_and_train(args, dynamics, control_generator)


if __name__ == '__main__':
    print_gpu_info()
    main()
