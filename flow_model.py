import torch
from torch import nn


class CausalFlowModel(nn.Module):
    def __init__(self, state_dim, control_dim, control_rnn_size, delta):
        super(CausalFlowModel, self).__init__()

        self.delta = delta

        self.u_rnn = torch.nn.LSTM(
            input_size=control_dim + 1,
            hidden_size=control_rnn_size,
            batch_first=True,
            num_layers=2,
        )

        self.u_dnn = SimpleNet(in_size=1 + control_rnn_size,
                               out_size=state_dim,
                               hidden_size=(10, 10))

        self.x_dnn = SimpleNet(in_size=1 + state_dim,
                               out_size=state_dim,
                               hidden_size=(10, 10))

        self.output_transform = nn.Sigmoid()
        self.combinator = nn.Linear(self.x_dnn.out_size + self.u_dnn.out_size,
                                    state_dim)

    def forward(self, t, x, u):
        state_part = self.x_dnn(torch.hstack((t, x)))

        batch_size = t.shape[0]
        seq_size = u.shape[1]
        u_rnn_in = torch.empty((batch_size, seq_size, self.u_rnn.input_size))

        # control index corresponding to each time
        t_u = torch.floor(t / self.delta).long().squeeze()
        t_rel = t - self.delta * t_u.unsqueeze(-1)

        for k, v in enumerate(u_rnn_in):
            v[:] = torch.hstack((t[k].repeat(seq_size, 1), u[k]))

        _, (encoded_controls, _) = self.u_rnn(u_rnn_in)

        control_part = self.u_dnn(torch.hstack((t_rel, encoded_controls[-1])))

        stacked_outputs = self.output_transform(
            torch.hstack((state_part, control_part)))

        return x + torch.tanh(t) * self.combinator(stacked_outputs)


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
