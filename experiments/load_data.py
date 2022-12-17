import torch

torch.set_default_dtype(torch.float32)

from flow_model import (prepare_experiment, get_arg_parser, train,
                        print_gpu_info)


def main():
    ap = get_arg_parser()

    ap.add_argument('load_path',
                    type=str,
                    help="Path to load .pth trajectory dataset")

    args = ap.parse_args()

    data = torch.load(args.load_path)

    experiment, train_args = prepare_experiment(data, args)

    experiment.generator = data.generator

    train_time = train(experiment, *train_args)
    print(f"Training took {train_time:.2f} seconds.")


if __name__ == '__main__':
    print_gpu_info()
    main()
