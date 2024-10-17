import numpy as np

from semble import TrajectorySampler
from semble.dynamics import LinearSys
from semble.sequence_generators import RandomWalkSequence

from generate_data import parse_args, generate


def main():
    args = parse_args()

    a_matrix = np.array([[-0.01, 1], [0, -1]])
    b_matrix = np.array([0, 1]).reshape((-1, 1))
    dynamics = LinearSys(a_matrix, b_matrix)
    control_generator = RandomWalkSequence()

    sampler = TrajectorySampler(dynamics, args.control_delta,
                                control_generator)

    generate(args, sampler)


if __name__ == '__main__':
    main()
