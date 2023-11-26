import torch
from dynamics import HodgkinHuxleyFBE
from sequence_generators import UniformSqWave
from initial_state import HHFBEInitialState

from generate_data import parse_args, generate

from numpy import ceil
from scipy.signal import find_peaks


# Downsample trajectories, focusing on the peaks
def rejection_sampling(data):
    for (k, y) in enumerate(data.state):
        p_1 = y[:, 0].flatten()
        p_min = p_1.min()
        p_1 = ((p_1 - p_min) / (p_1.max() - p_min))

        p_2 = y[:, 5].flatten()
        p_min = p_2.min()
        p_2 = ((p_2 - p_min) / (p_2.max() - p_min))

        p = torch.maximum(p_1, p_2)

        likelihood_ratio = p / torch.mean(p)
        lr_bound = torch.max(likelihood_ratio)

        u = torch.rand((len(likelihood_ratio), ))
        keep_idxs = (u <= (likelihood_ratio / lr_bound))
        keep_idxs[0] = True

        for y_spiking in (y[:, 0], y[:, 5]):
            peaks, _ = find_peaks(y_spiking)
            keep_idxs[peaks] = True

        data.state[k] = y[keep_idxs, :]
        data.state_noise[k] = data.state_noise[k][keep_idxs, :]
        data.time[k] = data.time[k][keep_idxs, :]


def main():
    args = parse_args()

    dynamics = HodgkinHuxleyFBE()

    # Inputs are piecewise constant signals with 100 ms period
    # and uniformly distributed amplitudes between 0 and 1 uA.
    period = int(ceil(100. / (dynamics.time_scale * args.control_delta)).item())
    control_generator = UniformSqWave(period=period)

    initial_state = HHFBEInitialState()

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
