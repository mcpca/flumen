import torch
from torch import nn


class CausalFlowModel(nn.Module):

    def __init__(self, state_dim, control_dim, control_rnn_size, delta):
        super(CausalFlowModel, self).__init__()

        self.delta = delta

        self.u_rnn = RecurrentNet(in_size=control_dim,
                                  out_size=state_dim,
                                  hidden_size=control_rnn_size)

        self.u_dnn = SimpleNet(in_size=1 + self.u_rnn.out_size,
                               out_size=state_dim,
                               hidden_size=(20, 20))

        self.x_dnn = SimpleNet(in_size=1 + state_dim,
                               out_size=state_dim,
                               hidden_size=(20, 20))

        self.output_transform = nn.Sigmoid()
        self.combinator = nn.Linear(2 * state_dim, state_dim)

    def forward(self, t, x, u):
        state_part = self.x_dnn(torch.hstack((t, x)))

        self.u_rnn.init_hidden_state()
        u_rnn_out = torch.zeros((u.shape[0], self.u_rnn.out_size))

        for k, u_val in enumerate(u):
            u_rnn_out[k, :] = self.u_rnn(u_val.reshape((1, -1)))

        # find control index coresponding to each time
        t_u = torch.floor(t / self.delta).long().squeeze()
        encoded_controls = u_rnn_out[t_u, :]

        control_part = self.u_dnn(torch.hstack((t, encoded_controls)))

        stacked_outputs = self.output_transform(
            torch.hstack((state_part, control_part)))

        return self.combinator(stacked_outputs)


class RecurrentNet(nn.Module):

    def __init__(self, in_size, out_size, hidden_size):
        super(RecurrentNet, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self._hsz = hidden_size
        self.hidden = None

        self.act = nn.Tanh()
        self.i2h = nn.Linear(in_size + self._hsz, self._hsz)
        self.h2o = nn.Linear(in_size + self._hsz, out_size)

        self.init_hidden_state()

    def init_hidden_state(self):
        self.hidden = torch.zeros((1, self._hsz))
        return self.hidden

    def forward(self, input):
        joint_input = torch.hstack((input, self.hidden))

        self.hidden = self.act(self.i2h(joint_input))
        output = self.act(self.h2o(joint_input))

        return output


class SimpleNet(nn.Module):

    def __init__(self, in_size, out_size, hidden_size, activation=nn.Sigmoid):
        super(SimpleNet, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.stack = nn.Sequential(
            nn.Linear(in_size, hidden_size[0]),
            activation(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            activation(),
            nn.Linear(hidden_size[1], out_size),
        )

    def forward(self, input):
        return self.stack(input)
