import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

# the Dataset with exogenous variables
class TsDataset(Dataset):
    def __init__(self, X_raw: np.ndarray, Y_raw: np.ndarray, time_window, horizon, one_step_model=False):
        super(TsDataset, self).__init__()
        # 数据时序处理
        self.X_hist = torch.Tensor([X_raw[i:i + time_window] for i in range(len(X_raw) - time_window - horizon + 1)])
        self.Y_hist = torch.Tensor([Y_raw[i:i + time_window] for i in range(len(X_raw) - time_window - horizon + 1)])
        if one_step_model:
            self.Y_gt = torch.Tensor([Y_raw[i + horizon - 1:i + horizon]
                                      for i in range(time_window, len(X_raw) - horizon + 1)])
        else:
            self.Y_gt = torch.Tensor([Y_raw[i:i + horizon] for i in range(time_window, len(X_raw) - horizon + 1)])
        # print('The size of dataset is X_hist:{}, Y_hist:{}, Y_gt:{}'.format(
        #     self.X_hist.size(), self.Y_hist.size(), self.Y_gt.size()))
        # self.Y_gt = torch.FloatTensor([Y_raw[i + H - 1:i + H].values for i in range(T, len(X_raw) - H + 1)]).to(device)
        # print(Y_hist[-1, :, :], '\n', Y_gt[-1, :, :])
        # print(self.X_hist.dtype)

    def __len__(self):
        return len(self.X_hist)

    def __getitem__(self, item):
        return self.X_hist[item], self.Y_hist[item], self.Y_gt[item]


class TsDataset2(Dataset):
    def __init__(self, X_raw: np.ndarray, Y_raw: np.ndarray, T: np.ndarray, time_window, horizon, one_step_model=False):
        super(TsDataset2, self).__init__()
        # 数据时序处理
        self.X_hist = torch.Tensor([X_raw[i:i + time_window] for i in range(len(X_raw) - time_window - horizon + 1)])
        self.Y_hist = torch.Tensor([Y_raw[i:i + time_window] for i in range(len(X_raw) - time_window - horizon + 1)])
        self.T = torch.Tensor([T[i:i + time_window + horizon] for i in range(len(X_raw) - time_window - horizon + 1)])
        if one_step_model:
            self.Y_gt = torch.Tensor([Y_raw[i + horizon - 1:i + horizon]
                                      for i in range(time_window, len(X_raw) - horizon + 1)])
        else:
            self.Y_gt = torch.Tensor([Y_raw[i:i + horizon] for i in range(time_window, len(X_raw) - horizon + 1)])
        # print('The size of dataset is X_hist:{}, Y_hist:{}, Y_gt:{}'.format(
        #     self.X_hist.size(), self.Y_hist.size(), self.Y_gt.size()))
        # self.Y_gt = torch.FloatTensor([Y_raw[i + H - 1:i + H].values for i in range(T, len(X_raw) - H + 1)]).to(device)
        # print(Y_hist[-1, :, :], '\n', Y_gt[-1, :, :])
        # print(self.X_hist.dtype)

    def __len__(self):
        return len(self.X_hist)

    def __getitem__(self, item):
        return self.X_hist[item], self.Y_hist[item], self.Y_gt[item], self.T[item]


class TsDataMouleFromNumpy:
    def __init__(self, np_path, time_window, horizon,
                 one_step_model=False,
                 normalize="std",
                 data_splite=(0.8, 0.1, 0.1),
                 batch_size=128,
                 pin_memory=False,
                 num_workers=0,
                 time_flag=False,
                 verbose=True):
        self.np_path = np_path
        self.time_window = time_window
        self.horizon = horizon
        self.batch_size = batch_size
        self.data_splite = data_splite
        self.one_step_model = one_step_model
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.time_flag = time_flag
        self.verbose = verbose

        if normalize == 'max':
            self.scaler_x = MaxMinScaler()
            self.scaler_y = MaxMinScaler()
        elif normalize == 'std':
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()

        self.setup()

    def setup(self):
        # load dataset
        data = np.load(self.np_path)

        # split feature and label
        # if self.targets is not None:
        X, Y = data['x'], data['y']
        self.X = self.scaler_x.fit_transform(X)
        self.Y = self.scaler_y.fit_transform(Y)

        if not self.time_flag:
            dataset = TsDataset(self.X, self.Y, self.time_window, self.horizon, self.one_step_model)
        else:
            self.T = data['t']
            dataset = TsDataset2(self.X, self.Y, self.T, self.time_window, self.horizon, self.one_step_model)

        n_data = len(dataset)
        if 0 <= self.data_splite[0] <= 1:
            train_index = int(n_data * self.data_splite[0])
            val_index = int(n_data * self.data_splite[1]) + train_index
            test_index = n_data
        else:
            split_mode = min(self.data_splite[0], 0) + min(self.data_splite[1], 0) + min(self.data_splite[2], 0)
            split_max = self.data_splite[0] + self.data_splite[1] + self.data_splite[2]
            if split_mode >= -1 and split_max <= n_data:
                train_index = self.data_splite[0] if self.data_splite[0] != -1 \
                    else n_data - self.data_splite[1] - self.data_splite[2]
                val_index = train_index + self.data_splite[1] if self.data_splite[1] != -1 \
                    else n_data - self.data_splite[0] - self.data_splite[2]
                test_index = val_index + self.data_splite[2] if self.data_splite[2] != -1 \
                    else n_data - self.data_splite[10] - self.data_splite[1]
            else:
                raise RuntimeError('Date split error!!!')

        self.train = Subset(dataset, range(train_index))
        self.val = Subset(dataset, range(train_index, val_index))
        self.test = Subset(dataset, range(val_index, test_index))
        if self.verbose:
            print("The split of train:val:test is {}:{}:{}".format(len(self.train), len(self.val), len(self.test)))

    def rescale(self, data, flag='y'):
        if flag == 'x':
            return self.scaler_x.inverse_transform(data)
        elif flag == 'y':
            return self.scaler_y.inverse_transform(data)

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle,
                          pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self, shuffle=False):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=shuffle,
                          pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self, shuffle=False):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=shuffle,
                          pin_memory=self.pin_memory, num_workers=self.num_workers)


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return self

    def transform(self, data: np.ndarray):
        data = (data - self.mean) / (self.std + np.finfo(float).eps)
        return data

    def fit_transform(self, data: np.ndarray):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray):
        std = self.std + np.finfo(float).eps
        mean = self.mean
        data = (data * std) + mean
        return data

    def inverse_transfrom_from_tensor(self, data: torch.Tensor):
        device = data.device
        std = torch.Tensor(self.std + np.finfo(float).eps).to(device)
        mean = torch.Tensor(self.mean).to(device)
        data = (data * std) + mean
        return data


class MaxMinScaler():
    """
    Standard the input
    """

    def __init__(self):
        self.max = None
        self.min = None

    def fit(self, data: np.ndarray):
        self.max = np.max(data, axis=0)
        self.min = np.min(data, axis=0)
        return self

    def transform(self, data: np.ndarray):
        data = (data - self.min) / (self.max - self.min + np.finfo(float).eps)
        return data

    def fit_transform(self, data: np.ndarray):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray):
        span = self.max - self.min + np.finfo(float).eps
        min = self.min
        data = data * span + min
        return data

    def inverse_transfrom_from_tensor(self, data: torch.Tensor):
        device = data.device
        span = torch.Tensor(self.max - self.min + np.finfo(float).eps).to(device)
        min = torch.Tensor(self.min).to(device)
        data = data * span + min
        return data