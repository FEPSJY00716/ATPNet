import torch.nn.functional as F

from model.long_memory import *
from .layers.attention import Attention
from dltools.hp_utils import PropertiesToolDecorator
from dltools.trainer import Step


class AEStep(Step):
    def __init__(self, cl=False, cl_step=1, criterion='mse_loss', scaler_y=None):
        self.cl = cl
        self.cl_step = cl_step
        self.criterion = criterion
        self.iter = 0
        self.task_level = 1
        self.scaler_y = scaler_y

    def loss_function(self, preds, labels):
        if self.criterion == 'l1_loss':
            loss = F.l1_loss(preds, labels)
        elif self.criterion == 'mse_loss':
            loss = F.mse_loss(preds, labels)
        return loss

    def extract_input(self, samples):
        x_hist, y_hist, y_gt, t = samples
        x = x_hist, y_hist, t
        y = y_gt
        return x, y

    def forward(self, model, samples):
        x, y_true = self.extract_input(samples)
        y_pred, x_encoded, x_decoded = model(*x if type(x) == tuple else x)
        return y_pred, y_true, x_encoded, x_decoded, x[1]

    def _normal_y(self, y):
        # print(x_true.shape)
        mean = torch.mean(y.squeeze(2), dim=1).view(-1, 1, 1)
        std = torch.std(y.squeeze(2), dim=1).view(-1, 1, 1)
        # print(mean.shape)
        return (y - mean) / std

    def train(self, model, samples):
        y_pred, y_gt, y_encoded, y_decoded, y_hist = self.forward(model, samples)  # output.shape = (batch_size, seq_out_len, n_nodes)

        # print(y_encoded[0, :, 0].shape, y_decoded[0, :, 0].shape, y_hist[0, :, 0].shape)
        # plot = TimeSeriesPlotter(plt)
        # plot._plotseries_compare(tss=[y_encoded[0, :, 0].cpu().detach().numpy(),
        #                               y_decoded[0, :, 0].cpu().detach().numpy(),
        #                               y_hist[0, :, 0].cpu().detach().numpy()],
        #                          labels=['encoded', 'decoded', 'hist'])
        # plot._plotseries_compare(tss=[y_pred[0, :, 0].cpu().detach().numpy(),
        #                               y_gt[0, :, 0].cpu().detach().numpy()],
        #                          labels=['pred', 'gt'])

        # if self.scaler_y is not None:
        #     y_pred = self.scaler_y.inverse_transform(y_pred)
        #     y_gt = self.scaler_y.inverse_transform(y_gt)

        if y_pred.size(1) != y_gt.size(1):
            raise RuntimeError("The shape of pred and true not matching")

        # print(y_pred.shape, y_gt.shape, x_decoded.shape, x_true.shape)
        loss1 = self.loss_function(y_pred, y_gt)
        loss2 = self.loss_function(y_decoded, y_hist)
        #
        #
        loss3 = self.loss_function(torch.mean(y_encoded.squeeze(2), dim=1),
                                   torch.tensor(0, dtype=y_encoded.dtype, device=y_encoded.device))
        loss4 = self.loss_function(torch.std(y_encoded.squeeze(2), dim=1),
                                   torch.tensor(1, dtype=y_encoded.dtype, device=y_encoded.device))

        # # print(y_norm)
        # loss5 = self.loss_function(y_encoded, self._normal_y(y_hist))
        gain = torch.tensor(1, dtype=y_gt.dtype, device=y_gt.device)

        # loss = gain * (loss1 + loss2)
        # print(loss1, loss2, loss3, loss4)
        # loss = loss1 + loss2
        loss = loss1
        # print(gain * (loss1 + loss2), loss3)
        # # print(loss1.item(), loss2.item(), loss3.item())


        self.iter += 1

        # 评估指标
        metric = {}
        return loss, metric

    def eval(self, model, samples):
        y_pred, y_gt, _, _, _ = self.forward(model, samples)

        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred)
            y_gt = self.scaler_y.inverse_transform(y_gt)

        loss = self.loss_function(y_pred, y_gt)

        # 评估指标
        metric = {}
        return loss, metric

    def test(self, model, samples):
        y_pred, y_gt, _, _, _ = self.forward(model, samples)

        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred)
            y_gt = self.scaler_y.inverse_transform(y_gt)

        return y_pred, y_gt



class ShortMemory(nn.Module):
    def __init__(self, horizon, n_seq, n_seq_cxt,
                 n_var_cxt, n_var_time, n_hidden=64,
                 on_context=False, on_time=False):
        super(ShortMemory, self).__init__()
        self.horizon = horizon
        self.n_seq = n_seq
        self.n_seq_cxt = n_seq_cxt
        self.n_var_cxt = n_var_cxt
        self.n_var_time = n_var_time
        self.n_hidden = n_hidden

        self.attn_temporal = Attention(self.n_var_cxt, 2 * self.n_hidden, self.n_var_cxt)
        # self.fc_lstm = nn.Linear(self.n_var_cxt + 1 + self.n_var_time, 1)
        self.fc_lstm = nn.Sequential(
            # nn.Linear(self.n_var_cxt + 1 + self.n_var_time, 1),
            # nn.Linear(1 + self.n_var_time, 1),
            nn.Linear(self.n_var_cxt, 1),
            # nn.BatchNorm1d(1)
        )
        self.lstm = nn.LSTM(input_size=2, hidden_size=self.n_hidden)
        self.fc_time = nn.Sequential(
            nn.Linear(self.n_var_time, 16),
            # nn.BatchNorm1d(self.horizon),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.fc_final = nn.Linear(self.n_hidden, self.horizon)

        self.h_ln = nn.LayerNorm(self.n_hidden)
        self.s_ln = nn.LayerNorm(self.n_hidden)

        self.fc_encoder = nn.Sequential(
            nn.Linear(self.n_var_time + 1, 16),
            # nn.BatchNorm1d(self.horizon),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.n_var_time + 1, 16),
            # nn.BatchNorm1d(self.horizon),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x, y, t):
        device = y.device
        dtype = y.dtype
        n_batch = y.size(0)

        attn_weight = torch.zeros((n_batch, self.n_seq, self.n_seq_cxt), dtype=dtype, device=device)
        h_n = torch.zeros((1, n_batch, self.n_hidden), dtype=dtype, device=device)
        s_n = torch.zeros((1, n_batch, self.n_hidden), dtype=dtype, device=device)
        y_encoded = torch.zeros((n_batch, self.n_seq, 1), dtype=dtype, device=device)
        y_decoded = torch.zeros((n_batch, self.n_seq, 1), dtype=dtype, device=device)

        for ti in range(self.n_seq):
            Q = torch.cat((h_n, s_n), dim=2).permute(1, 0, 2)
            K = x
            attn_wt = self.attn_temporal(Q, K)
            context = torch.bmm(attn_wt.unsqueeze(dim=1), x)  # shape = (batch_size, 1, n_var_long)
            #
            # # fc_input = torch.cat((context.squeeze(1), y[:, ti, :], t[:, ti, :]), dim=1)
            # # fc_input = torch.cat((y[:, ti, :], t[:, ti, :]), dim=1)
            # fc_input = torch.cat((context.squeeze(1), y[:, ti, :]), dim=1)
            context = self.fc_lstm(context).squeeze(1)  # shape = (batch_size, 1)
            lstm_input = torch.cat((context, y[:, ti, :]), dim=1).unsqueeze(dim=0)  # shape= (1, batch_size, 2)
            # lstm_input = y[:, ti, :].unsqueeze(dim=0)
            # lstm_input = y[:, ti, :].unsqueeze(dim=0) - self.fc_time(t[:, ti, :])
            # y_encoded[:, ti, :] = lstm_input
            # y_decoded[:, ti, :] = y_encoded[:, ti, :].unsqueeze(dim=0) + self.fc_time(t[:, ti, :])
            # lstm_input = self.fc_encoder(torch.cat((y[:, ti, :], t[:, ti, :]), dim=1))
            # y_encoded[:, ti, :] = lstm_input
            # y_decoded[:, ti, :] = self.fc_decoder(torch.cat((lstm_input, t[:, ti, :]), dim=1))
            #
            # lstm_input = lstm_input.unsqueeze(dim=0)


            _, final_state = self.lstm(lstm_input, (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            # h_n = self.h_ln(h_n)
            # s_n = self.s_ln(s_n)

            # attn_weight[:, ti, :] = attn_wt

        y_pred = self.fc_final(h_n.squeeze(dim=0))  # shape = (batch_size, H)
        # print(self.fc_time(t[:, -self.horizon:, :]).shape)
        # y_pred = y_pred.unsqueeze(dim=2) + self.fc_time(t[:, -self.horizon:, :])
        # print(t[0, -self.horizon:, :])
        # print(self.fc_time(t[:, -self.horizon:, :]).shape)
        y_pred = y_pred.unsqueeze(dim=2) + self.fc_time(t[:, -self.horizon:, :])
        # y_pred = self.fc_decoder(torch.cat((y_pred.unsqueeze(dim=2), t[:, -self.horizon:, :]), dim=2))
        # y_pred = y_pred.unsqueeze(dim=2)

        return y_pred, y_encoded, y_decoded


@PropertiesToolDecorator
class GYModel(nn.Module):
    def __init__(self, T, H,
                 n_features,
                 n_encoder_hidden,
                 n_decoder_hidden):
        super(GYModel, self).__init__()
        self.T = T
        self.H = H
        self.n_features = n_features
        self.n_encoder_hidden = n_encoder_hidden
        self.n_decoder_hidden = n_decoder_hidden

        # self.encoder = RNNLongMemory(T, n_features, n_encoder_hidden, on_attn=True, on_norm=False)
        # self.encoder = TCNLongMemory(T, n_features, [n_encoder_hidden]*3)
        # self.encoder = LinearLongMemory(T)
        self.encoder = SelfAttnLongMemory(n_features, n_encoder_hidden)
        self.decoder = ShortMemory(H, T, T, n_encoder_hidden, 3, n_decoder_hidden)
        # self.decoder = ShortMemory(H, T, T, n_encoder_hidden, 3, n_decoder_hidden)

    def forward(self, x, y, t=None):
        """
        forward of DARNN
        :param X: shape = (batch_size, T, n_features)
        :param y_hist: shape = (batch_size, T)
        :return: y_pred
        """
        # first phase attention stage
        # X_encoded, _ = self.encoder(x)  # attn_weight1.shape = (batch_size, T, n_feature)
        X_encoded = self.encoder(x)  # attn_weight1.shape = (batch_size, T, n_feature)
        # X_encoded.shape = (batch_size, T, n_encoded)
        # temporal attention of decoder stage
        y_pred, y_encoded, y_decoded = self.decoder(X_encoded, y, t)
        # y_pred = y_pred + self.linear_trans(T[:, -self.H:, :])

        # return y_pred, y_encoded, y_decoded
        return y_pred

