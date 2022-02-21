import torch
from torch import nn


class CausalFlowModel(nn.Module):
    def __init__(self, state_dim, control_dim, control_rnn_size):
        super(CausalFlowModel, self).__init__()

        self.u_rnn = RecurrentNet(in_size=control_dim,
                                  out_size=state_dim,
                                  hidden_size=control_rnn_size)

        self.u_dnn = SimpleNet(in_size=1 + self.u_rnn.out_size,
                               out_size=state_dim,
                               hidden_size=(10, 10))

        self.x_dnn = SimpleNet(in_size=1 + state_dim,
                               out_size=state_dim,
                               hidden_size=(10, 10))

        self.output_transform = nn.Tanh()
        self.combinator = nn.Linear(2 * state_dim, state_dim)

    def forward(self, t, x, u):
        state_part = self.x_dnn(torch.hstack((t, x)))

        u_rnn_out = self.u_rnn.init_hidden_state()

        for u_val in u:
            u_rnn_out = self.u_rnn(u_val.reshape((1, -1)))

        u_rnn_out = u_rnn_out.repeat(t.shape[0], 1)

        control_part = self.u_dnn(torch.hstack((t, u_rnn_out)))

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
    def __init__(self, in_size, out_size, hidden_size):
        super(SimpleNet, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.stack = nn.Sequential(
            nn.Linear(in_size, hidden_size[0]),
            nn.Tanh(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Tanh(),
            nn.Linear(hidden_size[1], out_size),
        )

    def forward(self, input):
        return self.stack(input)
