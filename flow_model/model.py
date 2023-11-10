import torch
from torch import nn


class CausalFlowModel(nn.Module):

    def __init__(self,
                 state_dim,
                 control_dim,
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
        self.control_rnn_size = control_rnn_size

        self.u_rnn = FlowNet(input_size=1 + control_dim,
                             hidden_size=control_rnn_size)

        x_dnn_osz = control_rnn_depth * control_rnn_size
        self.x_dnn = FFNet(in_size=state_dim,
                           out_size=x_dnn_osz,
                           hidden_size=encoder_depth *
                           (encoder_size * x_dnn_osz, ),
                           use_batch_norm=use_batch_norm)

        u_dnn_isz = control_rnn_size
        self.u_dnn = FFNet(in_size=u_dnn_isz,
                           out_size=state_dim,
                           hidden_size=decoder_depth *
                           (decoder_size * u_dnn_isz, ),
                           use_batch_norm=use_batch_norm)

    def forward(self, x, rnn_input, deltas):
        h = self.x_dnn(x)
        c = torch.zeros_like(h)

        for u in rnn_input:
            h, c = self.u_rnn(u, h, c)

        return self.u_dnn(h)


class FlowNet(nn.Module):
    '''Based on https://github.com/seba-1511/lstms.pth'''

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super(FlowNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

    def forward(self, u, h, c):
        deltas = u[:, -1].unsqueeze(-1)

        # Linear mappings
        preact = self.i2h(u) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
        h_d = torch.mul(o_t, c_t.tanh())

        h_t = h + deltas * h_d

        return h_t, c_t


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
