import torch

from semble import TrajectorySampler
from semble.dynamics import HodgkinHuxleyFS
from semble.sequence_generators import UniformSqWave

from generate_data import parse_args, generate

from numpy import ceil
from scipy.signal import find_peaks


# Downsample trajectories, focusing on the peaks
def rejection_sampling(data):
    for (k, y) in enumerate(data.state):
        p = y[:, 0].flatten()
        p_min = p.min()
        p = 1e-4 + ((p - p_min) / (p.max() - p_min))

        likelihood_ratio = p / torch.mean(p)
        lr_bound = torch.max(likelihood_ratio)

        u = torch.rand((len(likelihood_ratio), ))
        keep_idxs = (u <= (likelihood_ratio / lr_bound))
        keep_idxs[0] = True

        peaks, _ = find_peaks(y[:, 0])
        keep_idxs[peaks] = True

        data.state[k] = y[keep_idxs, :]
        data.state_noise[k] = data.state_noise[k][keep_idxs, :]
        data.time[k] = data.time[k][keep_idxs, :]


def main():
    args = parse_args()

    dynamics = HodgkinHuxleyFS()

    # Inputs are piecewise constant signals with 100 ms period
    # and uniformly distributed amplitudes between 0 and 1 uA.
    period = int(
        ceil(100. / (dynamics.time_scale * args.control_delta)).item())
    control_generator = UniformSqWave(period=period)

    sampler = TrajectorySampler(dynamics, args.control_delta,
                                control_generator)

    generate(args, sampler, postprocess=[
        rejection_sampling,
    ])


if __name__ == '__main__':
    main()
