import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from trajectory import TrajectoryGenerator, TrajectoryDataset
from flow_model import CausalFlowModel

torch.set_default_dtype(torch.float64)

# A = np.array([[-4., 2. * np.pi], [-2. * np.pi, 0.]], dtype=np.float32)
# A_ = np.array([[-1, 0, 1], [2, 0, 2 * np.pi], [0, -2 * np.pi, 0]])
A_ = np.array([[-0.01, 1], [0, -1]])
B_ = np.array([0, 1]).reshape((-1, 1))


def dynamics(x, u):
    return A_ @ x + B_ @ u


def main():
    trajectory_generator = TrajectoryGenerator(A_.shape[0], dynamics, 0.2)

    traj_data = TrajectoryDataset(trajectory_generator,
                                  n_trajectories=100,
                                  n_samples=20,
                                  time_horizon=10.)

    train_len = int(0.4 * len(traj_data))
    train_data, test_data = random_split(
        traj_data, (train_len, len(traj_data) - train_len))

    train_dl = DataLoader(train_data, shuffle=True)
    test_dl = DataLoader(test_data, shuffle=True)

    for x0, t, y, u in test_dl:
        fig, ax = plt.subplots()

        t = t.reshape((-1, 1))
        y = y.reshape(-1, A_.shape[1])

        ax.plot(t, y[:, 0], 'r--')
        ax.plot(t, y[:, 1], 'r--')

        plt.show()
        plt.close(fig)
        break

    model = CausalFlowModel(state_dim=A_.shape[0],
                            control_dim=B_.shape[1],
                            control_rnn_size=20)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 100

    print('Epoch :: Loss\n=================')

    for epoch in range(n_epochs):
        loss = 0.

        for example in train_dl:
            loss += train(example, model, optimizer, epoch)

        loss /= len(train_dl)
        print(f"{epoch + 1:>5d} :: {loss:>7f}")

    model.eval()

    with torch.no_grad():
        for x0, t, y, u in test_dl:
            fig, ax = plt.subplots()

            t = t.reshape(-1, 1)
            u = u.reshape(-1, B_.shape[1])
            x0 = x0.repeat(t.shape[0], 1)
            y = y.reshape(-1, A_.shape[1])
            y_pred = model(t, x0, u).numpy()

            ax.plot(y_pred[:, 0], 'k')
            ax.plot(y_pred[:, 1], 'k')
            ax.plot(y[:, 0], 'r--')
            ax.plot(y[:, 1], 'r--')

            plt.show()
            plt.close(fig)


def train(example, model, optimizer, epoch):
    model.train()
    x0, t, y, u = example

    t = t.reshape(-1, 1)
    u = u.reshape(-1, B_.shape[1])
    x0 = x0.repeat(t.shape[0], 1)
    y = y.reshape(-1, A_.shape[1])

    mse = nn.MSELoss()

    optimizer.zero_grad()

    y_pred = model(t, x0, u)
    loss = mse(y, y_pred)

    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    main()
