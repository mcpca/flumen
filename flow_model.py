import torch
from torch import nn


class CausalFlowModel(nn.Module):
    def __init__(self, state_dim, control_dim, control_rnn_size, delta,
                 norm_center, norm_weight, generator):
        super(CausalFlowModel, self).__init__()

        self.generator = generator
        self.center = norm_center
        self.weight = norm_weight

        self.delta = torch.tensor((delta, ))

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

        self.u_dnn = FFNet(in_size=1 + control_rnn_size,
                           out_size=state_dim,
                           hidden_size=(2 * control_rnn_size,
                                        2 * control_rnn_size))

        self.x_dnn = FFNet(in_size=1 + state_dim,
                           out_size=state_dim,
                           hidden_size=(2 * state_dim, 2 * state_dim))

        self.output_transform = nn.Tanh()
        comb_isz = self.x_dnn.out_size + self.u_dnn.out_size
        self.combinator = FFNet(in_size=comb_isz,
                                out_size=state_dim,
                                hidden_size=(2 * comb_isz, 2 * comb_isz))

    def forward(self, t, x, u):
        batch_size = t.shape[0]
        seq_size = u.shape[1]
        u_rnn_in = torch.empty((batch_size, seq_size, self.u_rnn.input_size))

        # control index corresponding to each time
        t_u = torch.floor(t / self.delta).long().squeeze()
        t_rel = (t - self.delta * t_u.unsqueeze(-1)) / self.delta

        for k, v in enumerate(u_rnn_in):
            v[:] = torch.hstack((t[k].repeat(seq_size, 1), u[k]))

        encoded_controls, _ = self.u_rnn(u_rnn_in)
        encoded_controls = encoded_controls[range(len(t)), t_u, :]

        control_part = self.u_dnn(torch.hstack((t_rel, encoded_controls)))
        state_part = self.x_dnn(torch.hstack((t, x)))
        stacked_outputs = self.output_transform(
            torch.hstack((control_part, state_part)))

        return self.combinator(stacked_outputs)

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
