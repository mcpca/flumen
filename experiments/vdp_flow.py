from dynamics import VanDerPol
from sequence_generators import GaussianSqWave

from generate_data import parse_args, generate


def main():
    args = parse_args()

    dynamics = VanDerPol(1.0)
    control_generator = GaussianSqWave(period=5)

    generate(args, dynamics, control_generator)


if __name__ == '__main__':
    main()
