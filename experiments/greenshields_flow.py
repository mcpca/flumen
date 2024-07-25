from dynamics import GreenshieldsTraffic
from sequence_generators import UniformSqWave
from initial_state import GreenshieldsInitialState

from generate_data import parse_args, generate

N = 100 # number of cells


def main():
    args = parse_args()

    dynamics = GreenshieldsTraffic(N, 1.0)
    control_generator = UniformSqWave(period=1, min=0., max=0.5)
    initial_state = GreenshieldsInitialState(N)

    generate(args, dynamics, control_generator, initial_state)


if __name__ == '__main__':
    main()
