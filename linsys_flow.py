import torch

torch.set_default_dtype(torch.float32)

from torch import nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from trajectory import (TrajectoryGenerator, TrajectoryDataset,
                        RandomWalkSequence)
from flow_model import CausalFlowModel
from train import EarlyStopping, train, validate
from utils import parse_args
from dynamics import LinearSys


def main():
    args = parse_args()

    a_matrix = np.array([[-0.01, 1], [0, -1]])
    b_matrix = np.array([0, 1]).reshape((-1, 1))
    dynamics = LinearSys(a_matrix, b_matrix)

    trajectory_generator = TrajectoryGenerator(
        dynamics,
        control_generator=RandomWalkSequence(),
        control_delta=args.control_delta)

    traj_data = TrajectoryDataset(trajectory_generator,
                                  n_trajectories=args.n_trajectories,
                                  n_samples=args.n_samples,
                                  time_horizon=args.time_horizon)

    norm_center, norm_weight = traj_data.whiten_targets()

    train_len = int((float(args.train_val_split) / 100.) * len(traj_data))
    train_data, val_data = random_split(traj_data,
                                        lengths=(train_len,
                                                 len(traj_data) - train_len))

    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = CausalFlowModel(state_dim=dynamics.n,
                            control_dim=dynamics.m,
                            control_rnn_size=args.control_rnn_size,
                            delta=args.control_delta,
                            norm_center=norm_center,
                            norm_weight=norm_weight,
                            generator=trajectory_generator)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2)
    mse_loss = nn.MSELoss()

    early_stop = EarlyStopping(es_patience=args.es_patience,
                               es_delta=args.es_delta)

    print('Epoch :: Loss (Train) :: Loss (Val) :: Best (Val)')
    print('=================================================')

    for epoch in range(args.n_epochs):
        loss = 0.

        model.train()
        for example in train_dl:
            loss += train(example, mse_loss, model, optimizer, epoch)

        loss /= len(train_dl)

        model.eval()
        val_loss = validate(val_dl, mse_loss, model)
        sched.step(val_loss)
        early_stop.step(val_loss)

        print(
            f"{epoch + 1:>5d} :: {loss:>7e} :: {val_loss:>7e} :: {early_stop.best_val_loss:>7e}"
        )

        if early_stop.best_model and args.save_model:
            torch.save(model, f'outputs/{args.save_model}')

        if early_stop.early_stop:
            break


if __name__ == '__main__':
    main()
