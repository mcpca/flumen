import torch
from torch import nn


class CausalFlowModel(nn.Module):

    def __init__(self,
                 state_dim,
                 control_dim,
                 control_rnn_size,
                 num_layers=1):
        super(CausalFlowModel, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.control_rnn_size = control_rnn_size

        self.u_rnn = torch.nn.LSTM(
            input_size=1 + control_dim,
            hidden_size=control_rnn_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=0,
        )

        x_dnn_osz = self.u_rnn.num_layers * 2 * control_rnn_size
        self.x_dnn = FFNet(in_size=state_dim,
                           out_size=x_dnn_osz,
                           hidden_size=3 * (5 * x_dnn_osz, ))

        u_dnn_isz = control_rnn_size
        self.u_dnn = FFNet(in_size=u_dnn_isz,
                           out_size=state_dim,
                           hidden_size=3 * (5 * u_dnn_isz, ))

    def forward(self, t, x, u):
        hidden_states = self.x_dnn(x)

        h0, c0 = hidden_states.split(self.u_rnn.num_layers *
                                     self.control_rnn_size,
                                     dim=1)

        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.stack(c0.split(self.control_rnn_size, dim=1))

        rnn_out_seq_packed, _ = self.u_rnn(u, (h0, c0))
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
                                                      batch_first=True)

        u_raw, u_lens = torch.nn.utils.rnn.pad_packed_sequence(
            u, batch_first=True)
        deltas = u_raw[:, :, -1].unsqueeze(-1)

        h_shift = torch.roll(h, shifts=1, dims=1)
        h_shift[:, 0, :] = h0[-1]

        encoded_controls = (1 - deltas) * h_shift + deltas * h
        output = self.u_dnn(encoded_controls[range(encoded_controls.shape[0]),
                                             u_lens - 1, :])

        return output


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
