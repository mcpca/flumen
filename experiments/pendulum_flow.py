from dynamics import Pendulum
from sequence_generators import SinusoidalSequence

from generate_data import parse_args, generate


def main():
    args = parse_args()

    dynamics = Pendulum(damping=0.01)
    control_generator = SinusoidalSequence()

    generate(args, dynamics, control_generator)


if __name__ == '__main__':
    main()
