from semble import TrajectorySampler
from semble.sequence_generators import LogNormalSqWave
from semble.dynamics import FitzHughNagumo

from generate_data import parse_args, generate

from math import log


def main():
    args = parse_args()

    dynamics = FitzHughNagumo(tau=0.8, a=-0.3, b=1.4)
    control_generator = LogNormalSqWave(mean=log(0.2), std=0.5, period=8)
    sampler = TrajectorySampler(dynamics, args.control_delta,
                                control_generator)

    generate(args, sampler)


if __name__ == '__main__':
    main()
