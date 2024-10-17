from semble import TrajectorySampler
from semble.dynamics import Pendulum
from semble.sequence_generators import SinusoidalSequence

from generate_data import parse_args, generate


def main():
    args = parse_args()

    dynamics = Pendulum(damping=0.01)
    control_generator = SinusoidalSequence()

    sampler = TrajectorySampler(dynamics, args.control_delta,
                                control_generator)

    generate(args, sampler)


if __name__ == '__main__':
    main()
