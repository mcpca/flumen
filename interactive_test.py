import torch
from utils import TrainedModel
from flow_model import CausalFlowModel
from trajectory import TrajectoryGenerator, pack_model_inputs
import matplotlib.pyplot as plt

import sys


def main():
    model: TrainedModel = torch.load(sys.argv[1],
                                     map_location=torch.device('cpu'))

    print(model)

    model = model.model
    model.eval()

    trajectory_generator: TrajectoryGenerator = model.generator
    delta = trajectory_generator._delta

    weight = model.weight
    center = model.center

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
