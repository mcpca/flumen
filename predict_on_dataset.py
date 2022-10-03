import torch
from torch.utils.data import DataLoader
from train import validate
from meta import Meta

import pandas as pd
import numpy as np

from argparse import ArgumentParser
from os.path import dirname

PREFIX='mse_time_horizon_'


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('model', type=str, help="Path to model .pth file")
    ap.add_argument('data', type=str, help="Path to data .pth file")

    return ap.parse_args()


def main():
    args = parse_args()

    meta: Meta = torch.load(args.model, map_location=torch.device('cpu'))
    print(meta)
    data = torch.load(args.data, map_location=torch.device('cpu'))

    compute_loss_vals(meta, data)


def compute_loss_vals(meta, data):
    meta.set_root(dirname(__file__))
    model = meta.load_model()
    model.eval()

    data.state[:] = ((data.state[:] - meta.td_mean) @ meta.td_std_inv).type(torch.get_default_dtype())
    data.init_state[:] = ((data.init_state[:] - meta.td_mean) @ meta.td_std_inv).type(torch.get_default_dtype())

    dloader = DataLoader(data, shuffle=False, batch_size=1024)
    loss_fn = torch.nn.MSELoss()
    loss = validate(dloader, loss_fn, model, device=torch.device('cpu'))
    print(f"{loss:.3e}")


if __name__ == '__main__':
    main()

