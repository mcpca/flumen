from argparse import ArgumentParser
import os, csv, re

import torch
from torch.utils.data import Dataset, DataLoader

from meta import Meta


def parse_args():
    ap = ArgumentParser()

    ap.add_argument('dir',
                    type=str,
                    help="Directory containing the models to be tested")
    ap.add_argument('test_data', type=str, help="Test dataset.")

    return ap.parse_args()

    return


def main():
    args = parse_args()

    # test_data: Dataset = torch.load(args.test_data)
    # test_dl = DataLoader(test_data, batch_size=1024, shuffle=True)

    stats = (TrainTime(), TrainError(), ValError())

    file_prefix = os.path.split(args.dir)[-1]

    file_matcher = re.compile(file_prefix +
                              '_[0-9]{8}_[0-9]{6}_[0-9a-f]{32}.pth')

    rows = []

    for fname in os.listdir(args.dir):
        if not file_matcher.match(fname):
            continue

        load_path = os.path.join(args.dir, fname)
        meta: Meta = torch.load(load_path, map_location=torch.device('cpu'))
        meta.args.save_model = file_prefix  # DEBUG -- remove me later
        meta.set_root(os.path.dirname(__file__))

        rows.append({'id': meta.train_id.hex})

        for stat in stats:
            rows[-1][str(stat)] = stat(meta)

    ofname = os.path.join(args.dir, file_prefix + '_stats.csv')

    with open(ofname, 'w', newline='') as ofile:
        writer = csv.DictWriter(ofile,
                                fieldnames=['id'] +
                                [str(stat) for stat in stats])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestError:

    def __init__(self, test_dl: DataLoader):
        self.loss = torch.nn.MSELoss
        self.data = test_dl

    def __call__(self, meta: Meta):
        model = meta.load_model()

        def eval_fn(t, x, u):
            return meta.predict(model, t, x, u)

        rv = 0.

        with torch.no_grad():
            for (x0, t, y, u, lengths) in self.data:
                sort_idxs = torch.argsort(lengths, descending=True)

                x0 = x0[sort_idxs]
                t = t[sort_idxs]
                y = y[sort_idxs]
                u = u[sort_idxs]
                lengths = lengths[sort_idxs]

                u = torch.nn.utils.rnn.pack_padded_sequence(
                    u, lengths, batch_first=True, enforce_sorted=True)

                y_pred = eval_fn(t, x0, u)
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
