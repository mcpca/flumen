import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from trajectory import TrajectoryGenerator, TrajectoryDataset, pack_model_inputs
from flow_model import CausalFlowModel

torch.set_default_dtype(torch.float64)

# A = np.array([[-4., 2. * np.pi], [-2. * np.pi, 0.]], dtype=np.float32)
# A_ = np.array([[-1, 0, 1], [2, 0, 2 * np.pi], [0, -2 * np.pi, 0]])
A_ = np.array([[-0.01, 1], [0, -1]])
B_ = np.array([0, 1]).reshape((-1, 1))


def dynamics(x, u):
    return A_ @ x + B_ @ u


def main():
    delta = 0.5
    trajectory_generator = TrajectoryGenerator(A_.shape[0],
                                               dynamics,
                                               control_delta=delta)

    traj_data = TrajectoryDataset(trajectory_generator,
                                  n_trajectories=100,
                                  n_samples=100,
                                  time_horizon=10.)

    batch_size = 32
    train_dl = DataLoader(traj_data, batch_size=batch_size, shuffle=True)

    model = CausalFlowModel(state_dim=A_.shape[0],
                            control_dim=B_.shape[1],
                            control_rnn_size=12,
                            delta=delta)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    n_epochs = 240

    print('Epoch :: Loss\n=================')

    for epoch in range(n_epochs):
        try:
            loss = 0.

            for example in train_dl:
                loss += train(example, model, optimizer, epoch)

            loss /= len(train_dl)
            print(f"{epoch + 1:>5d} :: {loss:>7f}")

        except KeyboardInterrupt:
            break

    model.eval()

    with torch.no_grad():
        while True:
            fig, ax = plt.subplots()

            x0, t, y, u = trajectory_generator.get_example(time_horizon=10.,
                                                           n_samples=100)

            x0, t, u = pack_model_inputs(x0, t, u, delta)

            y_pred = model(t, x0, u)

            ax.plot(t, y_pred[:, 0], 'k', label='Prediction')
            ax.plot(t, y_pred[:, 1], 'k')
            ax.plot(t, y[:, 0], 'r--', label='State 1')
            ax.plot(t, y[:, 1], 'b--', label='State 2')

            ax.legend()

            plt.show()
            plt.close(fig)


def train(example, model, optimizer, epoch):
    model.train()
    x0, t, y, u = example

    mse = nn.MSELoss()

    optimizer.zero_grad()

    y_pred = model(t, x0, u)
    loss = mse(y, y_pred)

    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    main()
