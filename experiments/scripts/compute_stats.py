import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from argparse import ArgumentParser
import csv
from glob import glob
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

import numpy as np
from scipy.stats import sem

from flow_model import CausalFlowModel, TrajectoryDataset, Experiment


def parse_args():
    ap = ArgumentParser()

    ap.add_argument(
        'dir',
        nargs='+',
        type=str,
        help="One or more directories containing the models to be tested")

    ap.add_argument('--test_set',
                    nargs='+',
                    help="Additional test datasets.",
                    default=[])

    ap.add_argument('--no_write',
                    action='store_true',
                    help="Don't write a CSV file.")

    return ap.parse_args()


def main():
    args = parse_args()

    metrics = [
        TrainTime(),
        TrainError(),
        ValError(),
        TestError(),
        MeasurementNoise(),
        GetParam('lr'),
        GetParam('control_rnn_size'),
        GetParam('encoder_size'),
        GetParam('encoder_depth'),
    ]

    for path in args.test_set:
        metrics.append(TestOnData(path))

    rows = []

    for dir in args.dir:
        file_prefix = os.path.split(dir)[-1]

        for fname in glob('*.pth', root_dir=dir):
            load_path = os.path.join(dir, fname)
            experiment: Experiment = torch.load(
                load_path, map_location=torch.device('cpu'))

            rows.append({'id': experiment.train_id.hex})

            for metric in metrics:
                rows[-1][str(metric)] = metric(experiment)

        print(f"-- {dir}")

        for metric in metrics:
            metric_name = str(metric)
            values = np.array([row[metric_name] for row in rows])

            print(
                f"{metric_name:16} mean={np.mean(values):.2e}, sem={sem(values):.2e}"
            )

    if args.no_write:
        return

    for dir in args.dir:
        ofname = os.path.join(dir, file_prefix + '_stats.csv')

        with open(ofname, 'w', newline='') as ofile:
            writer = csv.DictWriter(ofile,
                                    fieldnames=['id'] +
                                    [str(stat) for stat in metrics])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


class TestOnData:

    def __init__(self, path):
        self.name = path
        self.data: TrajectoryDataset = torch.load(path)
        self.loss = torch.nn.MSELoss()

    def __call__(self, experiment: Experiment):
        model: CausalFlowModel = experiment.load_model()

        data = deepcopy(self.data)
        data.init_state[:] = (data.init_state -
                              experiment.td_mean) @ experiment.td_std_inv
        data.state[:] = (data.state -
                         experiment.td_mean) @ experiment.td_std_inv
        loader = DataLoader(data, batch_size=1024, shuffle=False)

        rv = 0.

        with torch.no_grad():
            for x0, t, y, u, lengths in loader:
                sort_idxs = torch.argsort(lengths, descending=True)

                x0 = x0[sort_idxs]
                t = t[sort_idxs]
                y = y[sort_idxs]
                u = u[sort_idxs]
                lengths = lengths[sort_idxs]

                u = torch.nn.utils.rnn.pack_padded_sequence(
                    u, lengths, batch_first=True, enforce_sorted=True)

                y_pred = model(t, x0, u)
                rv += self.loss(y, y_pred).item()

        return rv / len(self.data)

    def __str__(self):
        return f'test_mse_{self.name}'


class TestError:

    def __init__(self):
        pass

    def __call__(self, experiment: Experiment):
        return experiment.test_loss_best

    def __str__(self):
        return 'test_mse'


class MeasurementNoise:

    def __init__(self):
        pass

    def __call__(self, experiment: Experiment):
        return experiment.generator.noise_std

    def __str__(self):
        return 'noise_std'


class ValError:

    def __init__(self):
        pass

    def __call__(self, experiment: Experiment):
        return experiment.val_loss_best

    def __str__(self):
        return 'val_mse'


class TrainError:

    def __init__(self):
        pass

    def __call__(self, experiment: Experiment):
        return experiment.train_loss_best

    def __str__(self):
        return 'train_mse'


class TrainTime:

    def __init__(self):
        pass

    def __call__(self, experiment: Experiment):
        return experiment.train_time

    def __str__(self):
        return 'train_time'


class GetParam:

    def __init__(self, param: str):
        self.param = param

    def __call__(self, experiment: Experiment):
        return vars(experiment.args)[self.param]

    def __str__(self):
        return self.param


if __name__ == "__main__":
    main()
