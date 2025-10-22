import torch


def prep_inputs(x0, y, rnn_input, lengths, device):
    tau = rnn_input[range(rnn_input.shape[0]), lengths - 1, -1].unsqueeze(-1)

    rnn_input_padded = torch.nn.utils.rnn.pack_padded_sequence(
        rnn_input, lengths, batch_first=True, enforce_sorted=False)

    x0 = x0.to(device)
    y = y.to(device)
    rnn_input_padded = rnn_input_padded.to(device)
    tau = tau.to(device)

    return x0, y, rnn_input_padded, tau


def validate(data, loss_fn, model, device):
    vl = 0.

    with torch.no_grad():
        for example in data:
            x0, y, rnn_input, tau = prep_inputs(*example, device)

            y_pred = model(x0, rnn_input, tau)
            vl += loss_fn(y, y_pred).item()

    return model.state_dim * vl / len(data)


def train_step(example, loss_fn, model, optimizer, device):
    x0, y, rnn_input, tau = prep_inputs(*example, device)

    optimizer.zero_grad()

    y_pred = model(x0, rnn_input, tau)
    loss = model.state_dim * loss_fn(y, y_pred)

    loss.backward()
    optimizer.step()

    return loss.item()


class EarlyStopping:

    def __init__(self, es_patience, es_delta=0.):
        self.patience = es_patience
        self.delta = es_delta

        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model = False

    def step(self, val_loss):
        self.best_model = False

        if self.best_val_loss - val_loss > self.delta:
            self.best_val_loss = val_loss
            self.best_model = True
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
