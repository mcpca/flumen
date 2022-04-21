import torch
from torch import nn


class CausalFlowModel(nn.Module):
    def __init__(self, state_dim, control_dim, control_rnn_size, delta,
                 norm_center, norm_weight, generator):
        super(CausalFlowModel, self).__init__()

        self.generator = generator
        self.center = norm_center
        self.weight = norm_weight

        delta = torch.tensor((delta, ))
        self.register_buffer('delta', delta)

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.control_rnn_size = control_rnn_size

        self.u_rnn = torch.nn.LSTM(
            input_size=1 + control_dim,
            hidden_size=control_rnn_size,
            batch_first=True,
            num_layers=3,
            dropout=0,
        )

        x_dnn_osz = self.u_rnn.num_layers * 2 * control_rnn_size
        self.x_dnn = FFNet(in_size=state_dim,
                           out_size=x_dnn_osz,
                           hidden_size=(5 * x_dnn_osz, 5 * x_dnn_osz,
                                        5 * x_dnn_osz))

        u_dnn_isz = control_rnn_size
        self.u_dnn = FFNet(in_size=u_dnn_isz,
                           out_size=state_dim,
                           hidden_size=(5 * u_dnn_isz, 5 * u_dnn_isz,
                                        5 * u_dnn_isz))

    def forward(self, t, x, u):
        hidden_states = self.x_dnn(x)

        h0, c0 = hidden_states.split(self.u_rnn.num_layers *
                                     self.control_rnn_size,
                                     dim=1)

        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.stack(c0.split(self.control_rnn_size, dim=1))

        _, (hf, _) = self.u_rnn(u, (h0, c0))

        encoded_controls = hf[-1, :, :]
        output = self.u_dnn(encoded_controls)

        return output

    def predict(self, t, x0, u):
        y_pred = self.__call__(t, x0, u).numpy()
        y_pred[:] = self.center + y_pred @ self.weight

        return y_pred


class FFNet(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, activation=nn.Tanh):
        super(FFNet, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))
        self.layers.append(nn.BatchNorm1d(hidden_size[0]))
        self.layers.append(activation())

        for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
            self.layers.append(nn.Linear(isz, osz))
            self.layers.append(nn.BatchNorm1d(osz))
            self.layers.append(activation())

        self.layers.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)

        return input
