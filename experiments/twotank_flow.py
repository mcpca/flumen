from semble import TrajectorySampler
from semble.dynamics import TwoTank
from semble.sequence_generators import GaussianSqWave, Product

from generate_data import parse_args, generate


def main():
    args = parse_args()

    twotank = TwoTank()
    pump = GaussianSqWave(period=5)
    valve = GaussianSqWave(period=20)

    sampler = TrajectorySampler(twotank, args.control_delta,
                                Product([pump, valve]))

    generate(args, sampler)


if __name__ == '__main__':
    main()
