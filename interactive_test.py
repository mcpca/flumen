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

    fig, ax = plt.subplots()
    ax.semilogy(meta.train_loss, label='Train loss')
    ax.semilogy(meta.val_loss, label='Validation loss')
    ax.semilogy(meta.test_loss, label='Test loss')
    ax.legend()
    plt.show()

    model.eval()

    trajectory_generator: TrajectoryGenerator = meta.generator
    delta = trajectory_generator._delta

    with torch.no_grad():
        while True:
            fig, ax = plt.subplots()

            x0, t, y, u = trajectory_generator.get_example(time_horizon=100.,
                                                           n_samples=10000)

            x0_feed, t_feed, u_feed = pack_model_inputs(x0, t, u, delta)

            y_pred = meta.predict(model, t_feed, x0_feed, u_feed)
            ax.plot(t_feed, y_pred, 'k', label='Prediction')
            ax.plot(t, y, 'b--', label='True state')

            ax.legend()

            plt.show()
            plt.close(fig)


if __name__ == '__main__':
    main()
