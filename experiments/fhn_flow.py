from sequence_generators import LogNormalSqWave
from flow_model_odedata import run_experiment
from flow_model_odedata.utils import parse_args, print_gpu_info
from dynamics import FitzHughNagumo
from math import log

def main():
    args = parse_args()

    dynamics = FitzHughNagumo(tau=0.8, a=-0.3, b=1.4)
    control_generator = LogNormalSqWave(mean=log(0.2), std=0.1, period=5)

    run_experiment(args, dynamics, control_generator)


if __name__ == '__main__':
    print_gpu_info()
    main()
