from torch import load
from flow_model_odedata import (prepare_experiment, print_gpu_info,
                                get_arg_parser, training_loop)


def main():
    ap = get_arg_parser()

    ap.add_argument('load_path',
                    type=str,
                    help="Path to load .pth trajectory dataset")

    args = ap.parse_args()

    data = load(args.load_path)

    train_args = prepare_experiment(data, args)

    train_time = training_loop(*train_args)
    print(f"Training took {train_time:.2f} seconds.")


if __name__ == '__main__':
    print_gpu_info()
    main()
