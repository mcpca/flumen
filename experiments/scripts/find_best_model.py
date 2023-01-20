import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import re
import torch
from flow_model import Experiment


def main():
    val_mse = {}

    dir = sys.argv[-1]
    file_prefix = os.path.split(dir)[-1]
    file_matcher = re.compile(file_prefix +
                              '_[0-9]{8}_[0-9]{6}_[0-9a-f]{32}.pth')

    for fname in os.listdir(dir):
        if not file_matcher.match(fname):
            continue
        load_path = os.path.join(dir, fname)
        experiment: Experiment = torch.load(
            load_path, map_location=torch.device('cpu'))
        val_mse[load_path] = experiment.val_loss_best

    best_path = min(val_mse, key=val_mse.get)
    print(best_path)


if __name__ == '__main__':
    main()
