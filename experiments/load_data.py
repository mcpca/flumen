import torch

torch.set_default_dtype(torch.float32)

from flow_model import (prepare_experiment, get_arg_parser, train,
                        print_gpu_info)


def main():
    ap = get_arg_parser()

    ap.add_argument('load_path',
                    type=str,
                    help="Path to load .pth trajectory dataset")

    ap.add_argument(
        '--reset_noise_var',
        type=float,
        default=None,
        help="Regenerate the measurement noise and set variance to this value."
    )

    args = ap.parse_args()

    data = torch.load(args.load_path)

    if args.reset_noise_var:
        data.reset_state_noise(args.reset_noise_var)
        data.generator.noise_std = args.reset_noise_var

    experiment, train_args = prepare_experiment(data, args)

    experiment.generator = data.generator

    train_time = train(experiment, *train_args)
    print(f"Training took {train_time:.2f} seconds.")


if __name__ == '__main__':
    print_gpu_info()
    main()
