import torch
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from abc import ABC, abstractmethod
import random
from .hp_utils import PropertiesTool, PropertiesToolDecorator
from typing import Tuple, Any, Union, Dict, List
from .callbacks import Callback
from dltools.metrics import *
import torch.nn.functional as F
import numpy as np
import time


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    torch.set_default_tensor_type(torch.FloatTensor)


class Trainer(PropertiesTool):
    def __init__(self, epoch=1000, device='cpu', callbacks: List[Callback] = [],
                 op_method='adam', lr=0.001, clip=None,
                 w_decay=0, lr_decay=None, lr_decay_step=None,
                 verbose=True):
        super(Trainer, self).__init__()

        self.epoch = epoch
        self.device = device
        self.callbacks = callbacks
        self.callbacks.sort(key=lambda c: c.CALLBACK_PRIORITY)  # 按照优先级排序

        # Parameters of Optimizer and Grad
        self.lr = lr
        self.method = op_method  # 优化方法
        self.clip = clip  # 梯度剪切
        self.w_decay = w_decay  # 梯度剪切
        self.lr_decay = lr_decay  # decay of learning rate
        self.lr_decay_step = lr_decay_step  # 如果为None，这自动化的根据损失修改梯度，如果为list，则MultiStepLR

        self.verbose = verbose

    def _make_optimizer(self, params):
        if self.method == 'sgd':
            optimizer = optim.SGD(params, lr=self.lr, weight_decay=self.w_decay)
        elif self.method == 'adagrad':
            optimizer = optim.Adagrad(params, lr=self.lr, weight_decay=self.w_decay)
        elif self.method == 'adadelta':
            optimizer = optim.Adadelta(params, lr=self.lr, weight_decay=self.w_decay)
        elif self.method == 'adam':
            optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.w_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)
        return optimizer

    def _mount_hook(self, hook_name, *args):
        for callback in self.callbacks:
            hook = getattr(callback, hook_name)
            hook(*args)

    def fit(self, model, step, train_dataloader, val_dataloader, test_dataloader=None):
        print('-' * 89)
        self._mount_hook('on_train_start', self, model, step, train_dataloader, val_dataloader)

        model.to(self.device)
        self.optimizer = self._make_optimizer(model.parameters())

        # 学习率更新调度器
        used_lr_scheduler = 'None'
        if self.lr_decay is not None and self.lr_decay_step is None:
            used_lr_scheduler = 'ReduceLROnPlateau'
            scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=self.lr_decay)
        elif self.lr_decay is not None and self.lr_decay_step is not None:
            if type(self.lr_decay_step) is list:
                used_lr_scheduler = 'MultiStepLR'
                scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_decay_step, gamma=self.lr_decay)
            else:
                used_lr_scheduler = 'StepLR'
                scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay)

        last_outputs = {}
        last_idx_epoch = 0
        try:
            print('Begin training in {}, optimizer is {}, learning rate is {}, scheduler is {}'.format(
                self.device, self.method, self.lr, used_lr_scheduler))
            for idx_epoch in range(self.epoch):
                if self.verbose:
                    with tqdm(total=len(train_dataloader.dataset),
                              desc=f"[Epoch {idx_epoch + 1:3d}/{self.epoch}]",
                              ncols=100) as pbar:
                        train_loss, train_metrics = self._epoch_train(idx_epoch, model, pbar, step, train_dataloader)
                        val_loss, val_metrics = self._epoch_val(idx_epoch, model, step, val_dataloader)
                        if test_dataloader is not None:
                            test_loss, test_metrics = self._epoch_val(idx_epoch, model, step, test_dataloader)
                            pbar.set_postfix({'loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss})
                        else:
                            pbar.set_postfix({'loss': train_loss, 'val_loss': val_loss})
                else:
                    start = time.time()
                    train_loss, train_metrics = self._epoch_train(idx_epoch, model, None, step, train_dataloader)
                    val_loss, val_metrics = self._epoch_val(idx_epoch, model, step, val_dataloader)
                    end = time.time()
                    if idx_epoch % 100 == 0:
                        print("Epoch:{}, Cost:{:.4f}, Train_loss:{:.5f}, Val_loss:{:.5f}".format(
                            idx_epoch, end - start, train_loss, val_loss))

                # update last metrics
                last_outputs = {'train_loss': train_loss, 'val_loss': val_loss}
                last_outputs.update(train_metrics)
                last_outputs.update(val_metrics)
                last_idx_epoch = idx_epoch

                if self.lr_decay is not None and self.lr_decay_step is None:
                    scheduler.step(val_loss)  # 根据损失，更新学习率
                elif self.lr_decay is not None and self.lr_decay_step is not None:
                    scheduler.step()
            print('-' * 89)
            print('Exiting from training')
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early in epoch {}'.format(last_idx_epoch))

        self._mount_hook('on_train_end', self, model, last_idx_epoch, last_outputs)

    def _epoch_train(self, idx_epoch, model, pbar, step, train_dataloader):
        self._mount_hook('on_train_epoch_start', self, model, idx_epoch)
        train_loss = 0
        train_metrics = {}
        model.train()
        for idx_batch, samples in enumerate(train_dataloader):
            self._mount_hook('on_train_batch_start', self, model, samples, idx_batch)
            self._mount_hook('on_before_zero_grad', self, model, self.optimizer)

            self.optimizer.zero_grad()

            if type(samples) is list or type(samples) is tuple:
                samples = tuple(x.to(self.device) for x in samples)  # remove data to GPU
                batch_size = len(samples[0])
            else:
                samples.to(self.device)
                batch_size = len(samples)

            loss, metrics = step.train(model, samples)
            loss.backward()  # 反向传播

            if self.clip is not None:  # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)

            self.optimizer.step()  # 更新梯度

            # 计算评价指标
            train_loss += loss.item()
            train_metrics = {k: v.item() + int(train_metrics.get(k) or 0) for k, v in metrics.items()}

            if pbar is not None:
                # 跟新进度条
                pbar.set_postfix({'loss:': train_loss / (idx_batch + 1)})
                pbar.update(batch_size)

            self._mount_hook('on_train_batch_end', self, model, samples, idx_batch, loss.item())

        train_len = idx_batch + 1
        train_loss = train_loss / train_len
        train_metrics = {'train_' + k: v / train_len for k, v in train_metrics.items()}
        outputs = {'train_loss': train_loss}
        outputs.update(train_metrics)

        self._mount_hook('on_train_epoch_end', self, model, idx_epoch, outputs)

        return train_loss, train_metrics

    def _epoch_val(self, idx_epoch, model, step, val_dataloader):
        self._mount_hook('on_validation_epoch_start', self, model, idx_epoch)

        preds, trues = self._epoch_test(model, step, val_dataloader)
        val_loss = mse(preds, trues).item()
        val_metrics = {'val_rmse': rmse(preds, trues).item(),
                       'val_mae': mae(preds, trues).item(),
                       'val_mape': mape(preds, trues).item()}
        outputs = {'val_loss': val_loss}
        outputs.update(val_metrics)

        self._mount_hook('on_validation_epoch_end', self, model, idx_epoch, outputs)

        return val_loss, val_metrics

    def test(self, model, step, test_dataloader):
        self._mount_hook('on_test_start', self, model, step, test_dataloader)

        model.to(self.device)
        preds, trues = self._epoch_test(model, step, test_dataloader)

        if self.verbose:
            print(preds.shape, trues.shape)
            print('Exiting from test, and MSE:{:.4f}, RMSE:{:.4f}, MAE:{:.4f}, MAPE:{:.4f}'.format(
                mse(preds, trues).item(), rmse(preds, trues).item(),
                mae(preds, trues).item(), mape(preds, trues).item()))

            horizon = preds.size(1) - 1
            print('\t CORR is', end=' ')
            while horizon > 0:
                print('{}:{:.4f}'.format(
                    horizon+1, corr(preds[:, horizon, :], trues[:, horizon, :]).item()), end=' ')
                horizon = int(horizon / 2)

            horizon = preds.size(1) - 1
            print('\n\t RRSE is', end=' ')
            while horizon > 0:
                print('{}:{:.4f}'.format(
                    horizon+1, rrse(preds[:, horizon, :], trues[:, horizon, :]).item()), end=' ')
                horizon = int(horizon / 2)
            print()

        self._mount_hook('on_test_end', self, model, preds.cpu().numpy(), trues.cpu().numpy())

        return preds, trues

    def _epoch_test(self, model, step, test_dataloader):
        model.eval()
        pred_list = []
        true_list = []
        with torch.no_grad():
            for idx_batch, samples in enumerate(test_dataloader):
                if type(samples) is list or type(samples) is tuple:
                    samples = tuple(x.to(self.device) for x in samples)  # remove data to GPU
                else:
                    samples.to(self.device)

                batch_pred, batch_true = step.eval(model, samples)
                pred_list.append(batch_pred)
                true_list.append(batch_true)

            preds = torch.cat(pred_list, dim=0)
            trues = torch.cat(true_list, dim=0)
        return preds, trues


class Step(ABC):
    @abstractmethod
    def train(self, model, samples) -> [Any, Dict]:
        pass

    @abstractmethod
    def eval(self, model, samples):
        pass

    def extract_input(self, samples) -> [Union[Tuple, Any], Any]:
        x, y = samples
        return x, y

    def forward(self, model, samples):
        x, y_true = self.extract_input(samples)
        y_pred = model(*x if type(x) == tuple else x)
        return y_pred, y_true


class TimeSeriesStep(Step):
    def __init__(self, cl=False, cl_step=1, criterion='mse_loss', rescale=None):
        self.cl = cl
        self.cl_step = cl_step
        self.criterion = criterion
        self.iter = 0
        self.task_level = 1
        self.rescale = rescale

    def loss_function(self, preds, labels):
        if self.criterion == 'l1_loss':
            loss = F.l1_loss(preds, labels)
        elif self.criterion == 'mse_loss':
            loss = F.mse_loss(preds, labels)
        return loss

    def extract_input(self, samples) -> [Union[Tuple, Any], Any]:
        x_hist, y_hist, y_gt = samples
        x = x_hist, y_hist
        y = y_gt
        return x, y

    def train(self, model, samples):
        y_pred, y_gt = self.forward(model, samples)  # output.shape = (batch_size, seq_out_len, n_nodes)

        if self.rescale is not None:
            y_pred = self.rescale(y_pred)
            y_gt = self.rescale(y_gt)

        if y_pred.size(1) != y_gt.size(1):
            raise RuntimeError("The shape of pred and true not matching. pred:{}, true:{}".format(
                y_pred.size(1), y_gt.size(1)))

        if self.cl:
            seq_out_len = y_gt.size(1)
            if self.iter % self.cl_step == 0 and self.task_level < seq_out_len:
                self.task_level += 1  # 逐渐增加学习难度（防止过拟合）

            # 使用多步输出序列的前 task_level 个时间步计算损失（课程学习策略 curriculum learning，先易后难）
            loss = self.loss_function(y_pred[:, :self.task_level, :], y_gt[:, :self.task_level, :])
        else:
            loss = self.loss_function(y_pred, y_gt)

        self.iter += 1

        # 评估指标
        metric = {'mae': mape(y_pred, y_gt), 'rmse': rmse(y_pred, y_gt)}
        return loss, metric

    def eval(self, model, samples):
        y_pred, y_gt = self.forward(model, samples)

        if self.rescale is not None:
            y_pred = self.rescale(y_pred)
            y_gt = self.rescale(y_gt)

        return y_pred, y_gt
