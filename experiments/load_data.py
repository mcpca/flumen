import torch

torch.set_default_dtype(torch.float32)

from flow_model import (prepare_experiment, get_arg_parser, train,
                        print_gpu_info)


def main():
    ap = get_arg_parser()

    ap.add_argument('load_path',
                    type=str,
                    help="Path to load .pth trajectory dataset")

    ap.add_argument('--reset_noise',
                    action='store_true',
                    help="Regenerate the measurement noise.")

    ap.add_argument('--noise_std',
                    type=float,
                    default=None,
                    help="If reset_noise is set, set standard deviation ' \
                            'of the measurement noise to this value.")

    args = ap.parse_args()

    data = torch.load(args.load_path)

    if args.reset_noise:
        data.reset_state_noise(args.noise_std)
        if args.noise_std is not None:
            data.generator.noise_std = args.noise_std

    experiment, train_args = prepare_experiment(data, args)

    experiment.generator = data.generator

    train_time = train(experiment, *train_args)
    print(f"Training took {train_time:.2f} seconds.")


if __name__ == '__main__':
    print_gpu_info()
    main()
