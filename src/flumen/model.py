import torch
from torch import nn


class CausalFlowModel(nn.Module):

    def __init__(self,
                 state_dim,
                 control_dim,
                 output_dim,
                 control_rnn_size,
                 control_rnn_depth,
                 encoder_size,
                 encoder_depth,
                 decoder_size,
                 decoder_depth,
                 use_batch_norm=False):
        super(CausalFlowModel, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim

        self.control_rnn_size = control_rnn_size

        self.u_rnn = torch.nn.LSTM(
            input_size=1 + control_dim,
            hidden_size=control_rnn_size,
            batch_first=True,
            num_layers=control_rnn_depth,
            dropout=0,
        )

        x_dnn_osz = control_rnn_depth * control_rnn_size
        self.x_dnn = FFNet(in_size=state_dim,
                           out_size=x_dnn_osz,
                           hidden_size=encoder_depth *
                           (encoder_size * x_dnn_osz, ),
                           use_batch_norm=use_batch_norm)

        u_dnn_isz = control_rnn_size
        self.u_dnn = FFNet(in_size=u_dnn_isz,
                           out_size=output_dim,
                           hidden_size=decoder_depth *
                           (decoder_size * u_dnn_isz, ),
                           use_batch_norm=use_batch_norm)

    def forward(self, x, rnn_input, tau):
        h0 = self.x_dnn(x)
        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)

        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0, c0))
        h, lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
                                                            batch_first=True)

        # get next to last state (possibly h0)
        h_prev = h[range(h.shape[0]), lengths - 2, :]
        h_prev = torch.where(lengths.unsqueeze(-1) > 1, h_prev, h0[-1])

        h_last = h[range(h.shape[0]), lengths - 1, :]

        tau = tau[range(h.shape[0]), lengths - 1, :]
        z = (1 - tau) * h_prev + tau * h_last
        output = self.u_dnn(z)

        return output

    def forward_trajectory(self, x, u, skips, tau):
        h0 = torch.stack(self.x_dnn(x).split(self.control_rnn_size, dim=1))
        h = torch.empty((1, skips[-1] + 1, h0.shape[-1]))
        c = torch.empty_like(h)

        h[:, 0] = h0
        c[:, 0] = torch.zeros_like(h0)

        rnn_input = torch.hstack((u, torch.ones_like(u)))

        for k in range(skips[-1]):
            _, (h[:, k + 1], c[:,
                               k + 1]) = self.u_rnn(rnn_input[k].unsqueeze(0),
                                                    (h[:, k], c[:, k]))

        rnn_input = torch.hstack((u[skips], tau)).unsqueeze(1)
        h_prev, c_prev = h[:, skips, :], c[:, skips, :]
        _, (h_last, _) = self.u_rnn(rnn_input, (h_prev, c_prev))

        z = (1 - tau) * h_prev + tau * h_last
        output = self.u_dnn(z).squeeze()

        return output


class FFNet(nn.Module):

    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 activation=nn.Tanh,
                 use_batch_norm=False):
        super(FFNet, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))

        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_size[0]))

        self.layers.append(activation())

        for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
            self.layers.append(nn.Linear(isz, osz))

            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(osz))

            self.layers.append(activation())

        self.layers.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)

        return input
