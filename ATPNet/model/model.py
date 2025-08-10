import torch
from torch import nn
import torch.nn.functional as F


class dilated_inception(nn.Module):
    # kernel_set = [2, 3, 6, 7]
    dilation_set = [1, 2, 4, 6]

    def __init__(self, cin, cout, kernel_size=3):
        super(dilated_inception, self).__init__()
        # print(cin, cout)
        self.tconv = nn.ModuleList()
        # self.kernel_set = [7]
        cout = int(cout / len(self.dilation_set))  # 每个核输出 cout/dilation_set 个通道
        for dilation_factor in self.dilation_set:
            # Inception 多尺度卷积
            self.tconv.append(nn.Conv2d(cin, cout, (1, kernel_size), dilation=(1, dilation_factor)))  # 加入膨胀因子

    def forward(self, input):
        x = []
        # print(input.shape)  # (batch_size, residual_channels, n_notes, time_step)
        for i in range(len(self.dilation_set)):
            x.append(self.tconv[i](input))  # shape = [batch_size, cout/n_kernel, n_notes, n_seq_out_i]
            # print(x[i].shape)
            # pass
        for i in range(len(self.dilation_set)):
            x[i] = x[i][..., -x[-1].size(3):]  # 对每个核函数计算的尺度进行规整
        x = torch.cat(x, dim=1)  # 对结果进行拼接  shape = [batch_size, cout, n_notes, n_seq_out_{-1}]
        return x


class TCN_Block(nn.Module):
    def __init__(self, conv_channels, residual_channels, kernel_size, in_seq_len, dropout):
        super(TCN_Block, self).__init__()
        # Fig.5 堆砌时序卷积模块
        self.filter_conv = dilated_inception(residual_channels, conv_channels, kernel_size=kernel_size)
        self.gate_conv = dilated_inception(residual_channels, conv_channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)

        # 计算输出的时序长度
        dilation = self.filter_conv.dilation_set[-1]  # 选取多尺度卷积中最大的扩张因子大小
        self.out_seq_len = in_seq_len - dilation * (kernel_size - 1)

        # 将TCN输出的通道，转换到residual通道上
        self.residual_conv = nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1))

    def forward(self, x):
        residual = x  # 残差连接，用于网络的传播

        # 带有门控的卷积模块
        filter = self.filter_conv(x)
        filter = torch.tanh(filter)
        gate = self.gate_conv(x)
        gate = torch.sigmoid(gate)
        x = filter * gate  # x.shape=(batch_size, conv_channels, n_node, conv_seq_leng)
        # x = filter  # x.shape=(batch_size, conv_channels, n_node, conv_seq_leng)
        # print(x.shape)
        # 对于获得的 x 进行dropout正则化
        x = self.dropout(x)
        x = self.residual_conv(x)  # 1by1卷积，目的在于转换通道
        x = x + residual[:, :, :, -x.size(3):]  # -x.size(3) 的目的在于，将网络开始时的 residual 的形状与x匹配
        return x


class TCN(nn.Module):
    def __init__(self, seq_length, layers=3, kernel_size=3, dropout=0.3,
                 in_dim=1, conv_channels=32, residual_channels=32, end_channels=16):
        super(TCN, self).__init__()
        self.dropout = dropout
        self.seq_length = seq_length

        self.dropout = nn.Dropout(dropout)
        self.satch_blocks = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,  # 保证残差连接，通道数相同
                                    kernel_size=(1, 1))

        # 计算模型最高层的感受野
        dilation = 6
        if dilation > 1:
            self.receptive_field = dilation * layers * (kernel_size - 1) + 1
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        if self.seq_length > self.receptive_field:
            new_seq_length = self.seq_length
        else:
            new_seq_length = self.receptive_field

        # 循环 MTGNN 模型需要的模块
        for i in range(1, layers + 1):
            satch_block = TCN_Block(conv_channels, residual_channels, kernel_size, new_seq_length, dropout)
            self.satch_blocks.append(satch_block)
            new_seq_length = satch_block.out_seq_len
            # new_dilation *= dilation_exponential

        self.last_seq_length = new_seq_length

        # 模型的 output 部分
        self.layers = layers
        self.end_conv = nn.Conv2d(in_channels=residual_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1, 1),
                                  bias=True)
        self.end_channels = end_channels

    def forward(self, u, v):
        """

        :param v: shape = (batch_size, n_seq, n_feature)
        :param u: shape = (batch_size, n_seq, n_feature)
        :return: y shape = (batch_size, h, n_feature)
        """
        bs = u.size(0)
        n_u = u.size(2)
        n_v = v.size(2)
        input = torch.cat((u, v), dim=2)  # (batch_size, n_seq, n_feature)
        input = input.unsqueeze(dim=1)  # (batch_size, 1, n_seq, n_feature)
        input = input.permute(0, 1, 3, 2)  # (batch_size, 1, n_feature, n_seq)
        target_idx = torch.BoolTensor(list(map(lambda i: False if i < n_u else True, torch.arange(n_u + n_v)))).to(input.device)
        # print(target_idx)

        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            # 如何时间序列的长度，不足以支撑最高层 layer 中单个神经元的感受野，那么我们就需要对时间序列进行 padding
            input = F.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        # Conv2d 的输入格式为 (N, C_in, H, W) 输出格式为 (N. C_out, H, W)
        x = self.start_conv(input)  # 初始的 1-by-1 卷积，主要的目的是适配通道数，shape = (batch_size, residual_channels, n_notes, time_step:187)
        for i in range(self.layers):  # 中间网络层数，其中每一次包括 TCN, GCN 模块
            x = self.satch_blocks[i](x)
        x = self.end_conv(F.relu(x))
        x = self.dropout(x)


        x = x.permute(0, 3, 1, 2)  # (batch_size, n_seq, end_channel, n_feature)
        # print(x[:, :, :, ~target_idx].shape, self.last_seq_length, bs, n_v, n_u)
        u = x[:, :, :, ~target_idx].contiguous().view(bs, self.last_seq_length, -1)
        v = x[:, :, :, target_idx].contiguous().view(bs, self.last_seq_length, -1)
        return u, v


class MPTCN(nn.Module):
    def __init__(self, T, H, n_features):
        super(MPTCN, self).__init__()
        self.Tl = T
        self.H = H
        self.n_features = n_features

        self.tcn = TCN(T, layers=3, conv_channels=64, residual_channels=64, end_channels=32)
        self.Ts = self.tcn.last_seq_length
        x_features = self.tcn.end_channels * n_features
        y_features = self.tcn.end_channels * 1
        # self.linear_trans = nn.Conv1d(in_channels=self.tcn.last_seq_length, out_channels=H, kernel_size=x_features+y_features)
        self.linear_trans = nn.Conv1d(in_channels=self.tcn.last_seq_length, out_channels=H, kernel_size=y_features)

    def forward(self, X: torch.FloatTensor, y_hist: torch.FloatTensor):
        """
        forward of DARNN
        :param X: shape = (batch_size, T, n_features)
        :param y_hist: shape = (batch_size, T)
        :return: y_pred
        """
        x, y = self.tcn(X, y_hist)
        # y_pred = self.linear_trans(torch.cat((x, y), dim=2))
        y_pred = self.linear_trans(y)

        return y_pred


if __name__ == '__main__':
    import pandas as pd
    from dltools.dataload import TsDataMouleFromNumpy

    dm = TsDataMouleFromNumpy('../data/AQ2021_WD.npz', time_window=84, horizon=3,
                              data_splite=(-1, 204, 0))
    n_features = 9
    model = MPTCN(dm.time_window, dm.horizon, n_features, 128)
    # print(model.tcn.last_seq_length, model.tcn.receptive_field)

    for i in range(1):
        print(i)
        for x_hist, y_hist, y_gt in dm.train_dataloader():
            y_pred = model(x_hist, y_hist)
            print(x_hist.shape, y_hist.shape, y_gt.shape, y_pred.shape)
