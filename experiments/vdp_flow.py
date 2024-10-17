from semble import TrajectorySampler
from semble.dynamics import VanDerPol
from semble.sequence_generators import GaussianSqWave

from generate_data import parse_args, generate


def main():
    args = parse_args()

    dynamics = VanDerPol(1.0)
    control_generator = GaussianSqWave(period=5)
    sampler = TrajectorySampler(dynamics, args.control_delta,
                                control_generator)

    generate(args, sampler)


if __name__ == '__main__':
    main()
