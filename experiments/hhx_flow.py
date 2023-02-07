from dynamics import HodgkinHuxleyFS
from sequence_generators import UniformSqWave
from initial_state import HHFSInitialState

from generate_data import parse_args, generate

from math import log


def main():
    args = parse_args()

    dynamics = HodgkinHuxleyFS()
    control_generator = UniformSqWave(period=5)
    initial_state = HHFSInitialState()

    generate(args, dynamics, control_generator,
             initial_state_generator=initial_state,
             method='BDF')


if __name__ == '__main__':
    main()
