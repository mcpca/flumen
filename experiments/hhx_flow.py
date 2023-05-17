import torch
from dynamics import HodgkinHuxleyFS
from sequence_generators import UniformSqWave
from initial_state import HHFSInitialState

from generate_data import parse_args, generate

from math import log


# Downsample trajectories, focusing on the peaks
def rejection_sampling(data):
    for (k, y) in enumerate(data.state):
        p = y[:, 0].flatten()
        p_min = p.min()
        p = 0.01 + (p - p_min) / (p.max() - p_min)

        likelihood_ratio = p / torch.mean(p)
        lr_bound = torch.max(likelihood_ratio)

        u = torch.rand((len(likelihood_ratio), ))
        keep_idxs = (u <= (likelihood_ratio / lr_bound))

        data.state[k] = y[keep_idxs, :]
        data.state_noise[k] = data.state_noise[k][keep_idxs, :]
        data.time[k] = data.time[k][keep_idxs, :]


def main():
    args = parse_args()

    dynamics = HodgkinHuxleyFS()
    control_generator = UniformSqWave(period=5)
    initial_state = HHFSInitialState()

    generate(args,
             dynamics,
             control_generator,
             postprocess=[
                 rejection_sampling,
             ],
             initial_state_generator=initial_state,
             method='BDF')


if __name__ == '__main__':
    main()
