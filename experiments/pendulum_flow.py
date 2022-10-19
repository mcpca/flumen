from flow_model_odedata import run_experiment
from flow_model_odedata.utils import parse_args, print_gpu_info
from dynamics import Pendulum
from sequence_generators import SinusoidalSequence


def main():
    args = parse_args()

    dynamics = Pendulum(damping=0.01)
    control_generator = SinusoidalSequence()

    run_experiment(args, dynamics, control_generator)


if __name__ == '__main__':
    print_gpu_info()
    main()
