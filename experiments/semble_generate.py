import torch

torch.set_default_dtype(torch.float32)

import pickle, yaml
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError

from scipy.signal import find_peaks

from semble import TrajectorySampler, TSamplerSpec, make_trajectory_sampler
from flumen import RawTrajectoryDataset


def main():
    args = parse_args()

    with open(args.settings, "r") as f:
        settings: TSamplerSpec = yaml.load(f, Loader=yaml.FullLoader)

    sampler = make_trajectory_sampler(settings)
    postprocess = get_postprocess(settings["dynamics"]["name"])

    train_data, val_data, test_data = generate(
        args, sampler, postprocess=postprocess
    )

    data = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "settings": settings,
        "args": vars(args),
    }

    output_dir = Path("./data/")
    output_dir.mkdir(exist_ok=True)

    # Write to disk
    with open(output_dir.joinpath(args.output_name + ".pkl"), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_postprocess(dynamics: str):
    if dynamics.startswith("HodgkinHuxley"):
        if dynamics.endswith(("FS", "RSA", "IB")):
            return [
                rejection_sampling_single_neuron,
            ]
        elif dynamics.endswith(("FFE", "FBE")):
            return [
                rejection_sampling_two_neuron,
            ]
    return []


def rejection_sampling_single_neuron(data):
    for k, y in enumerate(data.state):
        p = y[:, 0].flatten()
        p_min = p.min()
        p = 1e-4 + ((p - p_min) / (p.max() - p_min))

        likelihood_ratio = p / torch.mean(p)
        lr_bound = torch.max(likelihood_ratio)

        u = torch.rand((len(likelihood_ratio),))
        keep_idxs = u <= (likelihood_ratio / lr_bound)
        keep_idxs[0] = True

        peaks, _ = find_peaks(y[:, 0])
        keep_idxs[peaks] = True

        data.state[k] = y[keep_idxs, :]
        data.state_noise[k] = data.state_noise[k][keep_idxs, :]
        data.time[k] = data.time[k][keep_idxs, :]


def rejection_sampling_two_neuron(data):
    for k, y in enumerate(data.state):
        p_1 = y[:, 0].flatten()
        p_min = p_1.min()
        p_1 = (p_1 - p_min) / (p_1.max() - p_min)

        p_2 = y[:, 5].flatten()
        p_min = p_2.min()
        p_2 = (p_2 - p_min) / (p_2.max() - p_min)

        p = torch.maximum(p_1, p_2)

        likelihood_ratio = p / torch.mean(p)
        lr_bound = torch.max(likelihood_ratio)

        u = torch.rand((len(likelihood_ratio),))
        keep_idxs = u <= (likelihood_ratio / lr_bound)
        keep_idxs[0] = True

        for y_spiking in (y[:, 0], y[:, 5]):
            peaks, _ = find_peaks(y_spiking)
            keep_idxs[peaks] = True

        data.state[k] = y[keep_idxs, :]
        data.state_noise[k] = data.state_noise[k][keep_idxs, :]
        data.time[k] = data.time[k][keep_idxs, :]


def percentage(value):
    value = int(value)

    if not (0 <= value <= 100):
        raise ArgumentTypeError(f"{value} is not a valid percentage")

    return value


def parse_args():
    ap = ArgumentParser()

    ap.add_argument(
        "settings",
        type=str,
        help="Path to a YAML file containing the parameters"
        " defining the trajectory sampler.",
    )

    ap.add_argument(
        "output_name", type=str, help="File name for writing the data to disk."
    )

    ap.add_argument(
        "--time_horizon", type=float, help="Time horizon", default=10.0
    )

    ap.add_argument(
        "--n_trajectories",
        type=int,
        help="Number of trajectories to sample",
        default=100,
    )

    ap.add_argument(
        "--n_samples",
        type=int,
        help="Number of state samples per trajectory",
        default=50,
    )

    ap.add_argument(
        "--noise_std",
        type=float,
        help="Standard deviation of measurement noise",
        default=0.0,
    )

    ap.add_argument(
        "--noise_seed", type=int, help="Measurement noise seed", default=None
    )

    ap.add_argument(
        "--data_split",
        nargs=2,
        type=percentage,
        help="Percentage of data used for validation and test sets",
        default=[20, 20],
    )

    return ap.parse_args()


def generate(args, trajectory_sampler: TrajectorySampler, postprocess=[]):
    if args.data_split[0] + args.data_split[1] >= 100:
        raise Exception("Invalid data split.")

    n_val = int(args.n_trajectories * (args.data_split[0] / 100.0))
    n_test = int(args.n_trajectories * (args.data_split[1] / 100.0))
    n_train = args.n_trajectories - n_val - n_test

    def get_example():
        x0, t, y, u = trajectory_sampler.get_example(
            args.time_horizon, args.n_samples
        )
        return {
            "init_state": x0,
            "time": t,
            "state": y,
            "control": u,
        }

    train_data = [get_example() for _ in range(n_train)]
    trajectory_sampler.reset_rngs()

    val_data = [get_example() for _ in range(n_val)]
    trajectory_sampler.reset_rngs()

    test_data = [get_example() for _ in range(n_test)]

    train_data = RawTrajectoryDataset(
        train_data,
        *trajectory_sampler.dims(),
        delta=trajectory_sampler._delta,
        output_mask=trajectory_sampler._dyn.mask,
        noise_std=args.noise_std,
    )

    val_data = RawTrajectoryDataset(
        val_data,
        *trajectory_sampler.dims(),
        delta=trajectory_sampler._delta,
        output_mask=trajectory_sampler._dyn.mask,
        noise_std=args.noise_std,
    )

    test_data = RawTrajectoryDataset(
        test_data,
        *trajectory_sampler.dims(),
        delta=trajectory_sampler._delta,
        output_mask=trajectory_sampler._dyn.mask,
        noise_std=args.noise_std,
    )

    for d in (train_data, val_data, test_data):
        for p in postprocess:
            p(d)

    return train_data, val_data, test_data


if __name__ == "__main__":
    main()
