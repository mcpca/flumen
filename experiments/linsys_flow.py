import numpy as np
from sequence_generators import RandomWalkSequence
from dynamics import LinearSys

from generate_data import parse_args, generate


def main():
    args = parse_args()

    a_matrix = np.array([[-0.01, 1], [0, -1]])
    b_matrix = np.array([0, 1]).reshape((-1, 1))
    dynamics = LinearSys(a_matrix, b_matrix)
    control_generator = RandomWalkSequence()

    generate(args, dynamics, control_generator)


if __name__ == '__main__':
    main()
