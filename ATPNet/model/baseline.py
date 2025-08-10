import torch
from torch import nn

from dltools.hp_utils import PropertiesToolDecorator
from model.layers.tcn import TemporalConvNet


@PropertiesToolDecorator
class AR(nn.Module):
    def __init__(self, horizon, n_seq):
        super(AR, self).__init__()
        self.horizon = horizon
        self.n_seq = n_seq
        self.ar = nn.Linear(self.n_seq, 1)

    def forward(self, x, y):
        y = y.permute(0, 2, 1)  # [N, 1, T]
        for h in range(self.horizon):
            y_pt = self.ar(y[:, :, -self.n_seq:])
            y = torch.cat([y, y_pt], dim=2)
        y_pred = y[:, :, -self.horizon:].permute(0, 2, 1)
        return y_pred


@PropertiesToolDecorator
class MLP(nn.Module):
    def __init__(self, horizon, n_seq, n_hidden):
        super(MLP, self).__init__()
        self.n_seq = n_seq
        self.mlp = nn.Sequential(
            nn.Linear(n_seq, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, horizon)
        )

    def forward(self, x, y):
        return self.mlp(y.squeeze(2)).unsqueeze(2)


@PropertiesToolDecorator
class RNN(nn.Module):
    def __init__(self, horizon, n_seq, n_hidden,
                 n_layers=1, dropout=0.3, cell='LSTM',
                 on_exogenous=False, nx=None, whole_output=False):
        super(RNN, self).__init__()

        # super(RNN, self).__init__()
        self.horizon = horizon
        self.n_seq = n_seq
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.on_exogenous = on_exogenous
        self.nx = nx

        self.whole_output = whole_output

        if cell == 'LSTM':
            self.rnn_layer = nn.LSTM(input_size=nx + 1 if on_exogenous else 1,
                                     hidden_size=n_hidden,
                                     num_layers=n_layers, dropout=dropout)
        elif cell == 'GRU':
            self.rnn_layer = nn.GRU(input_size=nx + 1 if on_exogenous else 1,
                                    hidden_size=n_hidden,
                                    num_layers=n_layers, dropout=dropout)
        else:
            raise ValueError('ERROR: RNN cell is not defined')

        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=n_hidden * n_seq if whole_output else n_hidden, out_features=horizon)
        self.linear_trans = nn.Linear(3, 1)

    def forward(self, x, y):
        if self.on_exogenous:
            input = torch.cat((x, y), dim=2)
        else:
            input = y

        input = input.permute(1, 0, 2)  # shape = [t, n, v]
        output, _ = self.rnn_layer(input)

        if self.whole_output:
            fc_input = torch.flatten(output.permute(1, 0, 2), start_dim=1)
        else:
            fc_input = output[-1]

        y_pred = self.fc(fc_input)  # y_pred = [batch_size, H]
        y_pred = y_pred.unsqueeze(dim=2)

        return y_pred


@PropertiesToolDecorator
class Seq2Seq(nn.Module):
    def __init__(self, horizon, n_seq, n_hidden,
                 n_layers=1, dropout=0.3, cell='LSTM',
                 on_exogenous=False, nx=None):
        super().__init__()

        # super(RNN, self).__init__()
        self.horizon = horizon
        self.n_seq = n_seq
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.on_exogenous = on_exogenous
        self.nx = nx

        if cell == 'LSTM':
            self.encoder = nn.LSTM(input_size=nx + 1 if on_exogenous else 1,
                                   hidden_size=n_hidden,
                                   num_layers=n_layers, dropout=dropout)
            self.decoder = nn.LSTM(input_size=1, hidden_size=n_hidden,
                                   num_layers=n_layers, dropout=dropout)
        elif cell == 'GRU':
            self.encoder = nn.GRU(input_size=nx + 1 if on_exogenous else 1,
                                  hidden_size=n_hidden,
                                  num_layers=n_layers, dropout=dropout)
            self.decoder = nn.GRU(input_size=1, hidden_size=n_hidden,
                                  num_layers=n_layers, dropout=dropout)
        else:
            raise ValueError('ERROR: RNN cell is not defined')

        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=n_hidden, out_features=1)

    def forward(self, x, y):
        if self.on_exogenous:
            input = torch.cat((x, y), dim=2)
        else:
            input = y

        input = input.permute(1, 0, 2)  # shape = [t, n, v]
        output, final_state = self.encoder(input)

        y_pred = [self.fc(output[-1]).unsqueeze(dim=0)]
        for h in range(self.horizon - 1):
            y_pred_t, final_state = self.decoder(y_pred[-1], final_state)
            y_pred.append(self.fc(y_pred_t))
        y_pred = torch.cat(y_pred, dim=0).permute(1, 0, 2)

        return y_pred  # [batch_size, H, 1]


@PropertiesToolDecorator
class TCN(nn.Module):
    def __init__(self, horizon, n_seq, n_channels=[64] * 3, kernel_size=3,
                 dropout=0.3, on_exogenous=False, nx=None):
        super(TCN, self).__init__()
        self.horizon = horizon
        self.n_seq = n_seq

        self.on_exogenous = on_exogenous
        self.nx = nx

        self.tcn = TemporalConvNet(nx + 1 if on_exogenous else 1, n_channels, kernel_size, dropout)
        self.fc_h = nn.Linear(self.n_seq, self.horizon)
        self.fc_v = nn.Linear(n_channels[-1], 1)

    def forward(self, x, y):
        if self.on_exogenous:
            x = torch.cat((x, y), dim=2)
        else:
            x = y
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.fc_h(x).transpose(1, 2)
        y_pred = self.fc_v(x)
        return y_pred