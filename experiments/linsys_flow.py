import numpy as np
from sequence_generators import RandomWalkSequence
from flow_model_odedata import run_experiment
from flow_model_odedata.utils import parse_args, print_gpu_info
from dynamics import LinearSys


def main():
    args = parse_args()

    a_matrix = np.array([[-0.01, 1], [0, -1]])
    b_matrix = np.array([0, 1]).reshape((-1, 1))
    dynamics = LinearSys(a_matrix, b_matrix)
    control_generator = RandomWalkSequence()

    run_experiment(args, dynamics, control_generator)


if __name__ == '__main__':
    print_gpu_info()
    main()
