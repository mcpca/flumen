import torch
from meta import Meta
from trajectory import TrajectoryGenerator, pack_model_inputs
import matplotlib.pyplot as plt

import sys
from os.path import dirname


def main():
    meta: Meta = torch.load(sys.argv[1], map_location=torch.device('cpu'))
    meta.set_root(dirname(__file__))

    print(meta)

    model = meta.load_model()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].semilogy(meta.train_loss)
    ax[0].set_ylabel('Train loss')
    ax[1].semilogy(meta.val_loss)
    ax[1].set_ylabel('Validation loss')

    plt.show()

    model.eval()

    trajectory_generator: TrajectoryGenerator = meta.generator
    delta = trajectory_generator._delta

    weight = meta.td_std
    center = meta.td_mean

    with torch.no_grad():
        while True:
            fig, ax = plt.subplots()

            x0, t, y, u = trajectory_generator.get_example(time_horizon=100.,
                                                           n_samples=10000)

            x0_feed, t_feed, u_feed = pack_model_inputs(
                x0, t, u, delta, center, weight)

            y_pred = meta.predict(model, t_feed, x0_feed, u_feed)
            ax.plot(t_feed, y_pred, 'k', label='Prediction')
            ax.plot(t, y, 'b--', label='True state')

            ax.legend()

            plt.show()
            plt.close(fig)


if __name__ == '__main__':
    main()
