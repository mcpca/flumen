import torch
from flow_model import CausalFlowModel
from trajectory import pack_model_inputs
import matplotlib.pyplot as plt

import sys


def main():
    model: CausalFlowModel = torch.load(sys.argv[1],
                                        map_location=torch.device('cpu'))

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.eval()

    delta = model.delta.item()

    trajectory_generator = model.generator

    with torch.no_grad():
        while True:
            fig, ax = plt.subplots()

            x0, t, y, u = trajectory_generator.get_example(time_horizon=10.,
                                                           n_samples=100)

            x0, t, u = pack_model_inputs(x0, t, u, delta)

            y_pred = model.predict(t, x0, u)
            ax.plot(t, y_pred, 'k', label='Prediction')
            ax.plot(t, y, 'b--', label='True state')

            ax.legend()

            plt.show()
            plt.close(fig)


if __name__ == '__main__':
    main()
