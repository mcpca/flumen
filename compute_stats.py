from argparse import ArgumentParser
import os, csv, re

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from scipy.stats import sem
from scipy.linalg import inv

from flow_model import CausalFlowModel
from meta import Meta


def parse_args():
    ap = ArgumentParser()

    ap.add_argument('dir',
                    type=str,
                    help="Directory containing the models to be tested")

    ap.add_argument('--test_set', type=str, help="Test dataset.", default=None)

    ap.add_argument('--no_write', help="Don't write a CSV file.")

    return ap.parse_args()

    return


def main():
    args = parse_args()

    metrics = [TrainTime(), TrainError(), ValError()]

    if args.test_set:
        test_data: Dataset = torch.load(args.test_set)
        test_dl = DataLoader(test_data, batch_size=1024, shuffle=False)
        metrics.append(TestError(test_dl))

    file_prefix = os.path.split(args.dir)[-1]

    file_matcher = re.compile(file_prefix +
                              '_[0-9]{8}_[0-9]{6}_[0-9a-f]{32}.pth')

    rows = []

    for fname in os.listdir(args.dir):
        if not file_matcher.match(fname):
            continue

        load_path = os.path.join(args.dir, fname)
        meta: Meta = torch.load(load_path, map_location=torch.device('cpu'))
        meta.set_root(os.path.dirname(__file__))

        rows.append({'id': meta.train_id.hex})

        for metric in metrics:
            rows[-1][str(metric)] = metric(meta)

    for metric in metrics:
        metric_name = str(metric)
        values = np.array([row[metric_name] for row in rows])

        print(
            f"{metric_name:16} mean={np.mean(values):.2e}, sem={sem(values):.2e}"
        )

    if args.no_write:
        return

    ofname = os.path.join(args.dir, file_prefix + '_stats.csv')

    with open(ofname, 'w', newline='') as ofile:
        writer = csv.DictWriter(ofile,
                                fieldnames=['id'] +
                                [str(stat) for stat in metrics])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestError:

    def __init__(self, test_dl: DataLoader):
        self.loss = torch.nn.MSELoss()
        self.data = test_dl

    def __call__(self, meta: Meta):
        model: CausalFlowModel = meta.load_model()

        rv = 0.

        with torch.no_grad():
            for (x0, t, y, u, lengths) in self.data:
                sort_idxs = torch.argsort(lengths, descending=True)

                x0 = x0[sort_idxs]
                t = t[sort_idxs]
                y = y[sort_idxs]
                u = u[sort_idxs]
                lengths = lengths[sort_idxs]

                weight = inv(meta.td_std)
                x0[:] = (x0 - meta.td_mean) @ weight
                y[:] = (y - meta.td_mean) @ weight

                u = torch.nn.utils.rnn.pack_padded_sequence(
                    u, lengths, batch_first=True, enforce_sorted=True)

                y_pred = model(t, x0, u)
                rv += self.loss(y, y_pred)

        return rv / len(self.data)

    def __str__(self):
        return 'test_mse'


class ValError:

    def __init__(self):
        pass

    def __call__(self, meta: Meta):
        return meta.val_loss_best

    def __str__(self):
        return 'val_mse'


class TrainError:

    def __init__(self):
        pass

    def __call__(self, meta: Meta):
        return meta.train_loss_best

    def __str__(self):
        return 'train_mse'


class TrainTime:

    def __init__(self):
        pass

    def __call__(self, meta: Meta):
        return meta.train_time

    def __str__(self):
        return 'train_time'


if __name__ == "__main__":
    main()
