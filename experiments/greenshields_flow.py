from semble import TrajectorySampler
from semble.dynamics import GreenshieldsTraffic
from semble.sequence_generators import UniformSqWave
from semble.initial_state import GreenshieldsInitialState

from generate_data import parse_args, generate


def main():
    args = parse_args()

    dynamics = GreenshieldsTraffic(n=100, v0=1.0)
    control_generator = UniformSqWave(period=1, min=0., max=0.5)
    initial_state = GreenshieldsInitialState(n_cells=100, n_sections=4)

    sampler = TrajectorySampler(dynamics,
                                args.control_delta,
                                control_generator,
                                initial_state_generator=initial_state)

    generate(args, sampler)


if __name__ == '__main__':
    main()
