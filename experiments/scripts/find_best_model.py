import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from glob import glob
from flow_model import Experiment


def main():
    val_mse = {}

    dir = sys.argv[-1]
    file_prefix = os.path.split(dir)[-1]

    for fname in glob('*.pth', root_dir=dir):
        load_path = os.path.join(dir, fname)
        experiment: Experiment = torch.load(
            load_path, map_location=torch.device('cpu'))
        val_mse[load_path] = experiment.val_loss_best

    best_path = min(val_mse, key=val_mse.get)
    print(best_path)


if __name__ == '__main__':
    main()
