import numpy as np
from trajectory import RandomWalkSequence
from utils import parse_args, print_gpu_info
from dynamics import LinearSys
from sim_and_train import sim_and_train


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
