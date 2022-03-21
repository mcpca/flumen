import torch

torch.set_default_dtype(torch.float32)

from torch import nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from trajectory import (TrajectoryGenerator, TrajectoryDataset,
                        pack_model_inputs, RandomWalkSequence)
from flow_model import CausalFlowModel
from utils import parse_args

A_ = np.array([[-0.01, 1], [0, -1]])
B_ = np.array([0, 1]).reshape((-1, 1))


def dynamics(x, u):
    return A_ @ x + B_ @ u


def main():
    args = parse_args()

    trajectory_generator = TrajectoryGenerator(
        A_.shape[0],
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

    model = CausalFlowModel(state_dim=A_.shape[0],
                            control_dim=B_.shape[1],
                            control_rnn_size=args.control_rnn_size,
                            delta=args.control_delta)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    mse_loss = nn.MSELoss()

    print('Epoch :: Loss (Train) :: Loss (Val)')
    print('===================================')

    for epoch in range(args.n_epochs):
        try:
            loss = 0.

            model.train()
            for example in train_dl:
                loss += train(example, mse_loss, model, optimizer, epoch)

            model.eval()
            val_loss = validate(val_dl, mse_loss, model)
            sched.step(val_loss)

            loss /= len(train_dl)
            print(f"{epoch + 1:>5d} :: {loss:>7e} :: {val_loss:>7e}")

        except KeyboardInterrupt:
            break

    model.eval()

    with torch.no_grad():
        while True:
            fig, ax = plt.subplots()

            x0, t, y, u = trajectory_generator.get_example(time_horizon=10.,
                                                           n_samples=100)

            x0, t, u = pack_model_inputs(x0, t, u, args.control_delta)

            y_pred = model(t, x0, u).numpy()
            y_pred[:] = norm_center + y_pred @ norm_weight

            ax.plot(t, y_pred, 'k', label='Prediction')
            ax.plot(t, y, 'b--', label='True state')

            ax.legend()

            plt.show()
            plt.close(fig)


def validate(data, loss_fn, model):
    vl = 0.

    with torch.no_grad():
        for (x0, t, y, u) in data:
            y_pred = model(t, x0, u)
            vl += loss_fn(y, y_pred)

    return vl / len(data)


def train(example, loss_fn, model, optimizer, epoch):
    model.train()
    x0, t, y, u = example

    optimizer.zero_grad()

    y_pred = model(t, x0, u)
    loss = loss_fn(y, y_pred)

    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    main()
