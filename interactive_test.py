import torch
from utils import TrainedModel
from trajectory import TrajectoryGenerator, pack_model_inputs
import matplotlib.pyplot as plt

import sys


def main():
    model: TrainedModel = torch.load(sys.argv[1],
                                     map_location=torch.device('cpu'))

    print(model)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].semilogy(model.train_loss)
    ax[0].set_ylabel('Train loss')
    ax[1].semilogy(model.val_loss)
    ax[1].set_ylabel('Validation loss')

    plt.show()

    model.model.eval()

    trajectory_generator: TrajectoryGenerator = model.generator
    delta = trajectory_generator._delta

    weight = model.td_std
    center = model.td_mean

    with torch.no_grad():
        while True:
            fig, ax = plt.subplots()

            x0, t, y, u = trajectory_generator.get_example(time_horizon=100.,
                                                           n_samples=10000)

            x0_feed, t_feed, u_feed = pack_model_inputs(
                x0, t, u, delta, center, weight)

            y_pred = model.predict(t_feed, x0_feed, u_feed)
            ax.plot(t_feed, y_pred, 'k', label='Prediction')
            ax.plot(t, y, 'b--', label='True state')

            ax.legend()

            plt.show()
            plt.close(fig)


if __name__ == '__main__':
    main()
