from dynamics import TwoTank
from sequence_generators import GaussianSqWave, Product
from initial_state import UniformInitialState

from generate_data import parse_args, generate


def main():
    args = parse_args()

    dynamics = TwoTank()
    pump = GaussianSqWave(period=5)
    valve = GaussianSqWave(period=20)
    init_state = UniformInitialState(2)

    generate(args, dynamics, Product([pump, valve]),
             initial_state_generator=init_state, method='BDF')


if __name__ == '__main__':
    main()
