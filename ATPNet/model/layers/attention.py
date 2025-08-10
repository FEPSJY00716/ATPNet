import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class Attention(Module):
    def __init__(self, hidden, Q_size, K_size):
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(Q_size + K_size, hidden)
        self.linear2 = nn.Linear(hidden, 1)

    def forward(self, Q: torch.FloatTensor, K: torch.FloatTensor):
        """
        Calculate the 1-D Attention
        :param Q: query. shape=(batch_size, 1, 2*hidden_num)
        :param K: key. in the spacial attention, shape=(batch_size, n_features, T);
                    in the temporal attention, shape=(batch_size, T, n_encoder_hidden)
        :return:
        """
        target_num = K.size(1)
        # print("=======================")
        # print(Q.shape)
        # print(Q.repeat(1, target_num, 1).shape)
        # print(K.shape)
        input = torch.cat((Q.repeat(1, target_num, 1), K), dim=2)  # shape=(batch_size, n_target, 2 * n_hidden + T)
        l1 = torch.tanh(self.linear1(input))  # shape = (batch_size, n_target, hidden)
        E = self.linear2(l1)  # shape = (batch_size, n_target, 1)
        A = torch.softmax(E.squeeze(dim=2), dim=1)  # shape = (batch_size, n_target)
        return A