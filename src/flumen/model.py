import torch
from torch import nn


class CausalFlowModel(nn.Module):
    def __init__(
        self,
        state_dim,
        control_dim,
        output_dim,
        control_rnn_size,
        control_rnn_depth,
        encoder_size,
        encoder_depth,
        decoder_size,
        decoder_depth,
        use_batch_norm=False,
    ):
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
        self.x_dnn = FFNet(
            in_size=state_dim,
            out_size=x_dnn_osz,
            hidden_size=encoder_depth * (encoder_size * x_dnn_osz,),
            use_batch_norm=use_batch_norm,
        )

        u_dnn_isz = control_rnn_size
        self.u_dnn = FFNet(
            in_size=u_dnn_isz,
            out_size=output_dim,
            hidden_size=decoder_depth * (decoder_size * u_dnn_isz,),
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x, rnn_input, tau):
        h0 = self.x_dnn(x)
        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)

        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0, c0))
        h, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            rnn_out_seq_packed, batch_first=True
        )

        # get next to last state (possibly h0)
        h_prev = h[range(h.shape[0]), lengths - 2, :]
        h_prev = torch.where(lengths.unsqueeze(-1) > 1, h_prev, h0[-1])
        # get last state
        h_last = h[range(h.shape[0]), lengths - 1, :]

        z = (1 - tau) * h_prev + tau * h_last
        output = self.u_dnn(z)

        return output

    def forward_trajectory(self, x, u, skips, tau):
        h0 = torch.stack(torch.split(self.x_dnn(x), self.control_rnn_size, dim=1))

        lstm_depth = h0.shape[0]
        batch_size = h0.shape[1]
        hsz = h0.shape[-1]

        h = [h0]
        c = [torch.zeros_like(h0)]

        rnn_input = torch.cat(
            (u, torch.ones((batch_size, u.shape[1], 1), device=u.device)), dim=-1
        )

        for k in range(skips[-1]):
            _, (h_next, c_next) = self.u_rnn(rnn_input[:, k].unsqueeze(1), (h[k], c[k]))
            h.append(h_next)
            c.append(c_next)

        tau = tau.unsqueeze(0).expand(batch_size, -1, -1)
        rnn_input = torch.cat((u[:, skips], tau), dim=-1).view(-1, 1 + u.shape[-1])

        h = torch.stack(h, dim=2)
        c = torch.stack(c, dim=2)

        h_prev, c_prev = h[:, :, skips], c[:, :, skips]
        h_prev = h_prev.view(lstm_depth, -1, hsz)
        c_prev = c_prev.view(lstm_depth, -1, hsz)

        _, (h_last, _) = self.u_rnn(rnn_input.unsqueeze(1), (h_prev, c_prev))

        h_prev = h_prev[-1].view(batch_size, -1, hsz)
        h_last = h_last[-1].view(batch_size, -1, hsz)

        z = (1 - tau) * h_prev + tau * h_last
        output = self.u_dnn(z).squeeze()

        return output


class FFNet(nn.Module):
    def __init__(
        self, in_size, out_size, hidden_size, activation=nn.Tanh, use_batch_norm=False
    ):
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
