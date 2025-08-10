from abc import ABC, abstractmethod
import torch
import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Union
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import shutil
import copy


class Callback(ABC):
    CALLBACK_PRIORITY = 10

    def on_train_start(self, trainer, model, step, train_dataloader, val_dataloader):
        pass

    def on_train_epoch_start(self, trainer, model, idx_epoch):
        pass

    def on_train_batch_start(self, trainer, model, batch, idx_batch):
        pass

    def on_train_batch_end(self, trainer, model, batch, idx_batch, batch_loss):
        pass

    def on_train_epoch_end(self, trainer, model, idx_epoch, outputs):
        pass

    def on_validation_epoch_start(self, trainer, model, idx_epoch):
        pass

    def on_validation_epoch_end(self, trainer, model, idx_epoch, outputs):
        pass

    def on_train_end(self, trainer, model, last_idx_epoch, last_outputs):
        pass

    def on_before_zero_grad(self, trainer, model, optimizer):
        pass

    def on_test_start(self, trainer, model, step, test_dataloader):
        pass

    def on_test_end(self, trainer, model, preds: np.ndarray, trues: np.ndarray):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, dirpath, monitor='val_loss', filename=None, prefix='model', mode='min',
                 period=1, save_best=True, save_last=True, save_weights_only=True, save_hp=False):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.best_score = None
        self.dirpath = dirpath
        self.filename = filename
        self.prefix = prefix
        self.mode = mode
        self.period = period
        self.save_best = save_best
        self.save_last = save_last
        self.save_weights_only = save_weights_only
        self.save_hp = save_hp
        self.best_model_path = ""
        self.last_model_path = ""
        self.best_ckpt = {}

        self._init_ckpt_path()

    def _init_ckpt_path(self):
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        if self.filename is None:
            self.filename = '{epoch}-{val_loss:.3f}'

        self.best_model_path_frame = os.path.join(self.dirpath, "{}-{}.ckpt".format(self.prefix, self.filename))
        self.last_model_path_frame = os.path.join(self.dirpath, "{}-{}.ckpt".format(self.prefix, self.filename))

    def _init_ckpt_dict(self, epoch, model, outputs, trainer):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        ckpt = {
            'model_name': model.__class__.__name__,
            # model.state_dict() is shallow copy, the parameter will change in training. So need deep copy here.
            'model': copy.deepcopy(model.state_dict()) if self.save_weights_only else copy.deepcopy(model),
            'epoch': epoch,
            'outputs': outputs,
        }

        if self.save_hp:
            ckpt.update({
                'model_hparams': model.hparams if hasattr(model, 'hparams') else None,
                'trainer_hparams': trainer.hparams if hasattr(trainer, 'hparams') else None
            })
        return ckpt

    def _compare_best(self, score):
        if self.mode == 'min':
            return True if self.best_score is None else score < self.best_score
        elif self.mode == 'max':
            return True if self.best_score is None else score > self.best_score

    def on_validation_epoch_end(self, trainer, model, idx_epoch, outputs: dict):
        if idx_epoch % self.period == 0:
            if self.monitor not in outputs:
                raise RuntimeError('The monitor of Checkpoint is not in outputs of Step')

            idx_epoch += 1
            score = outputs[self.monitor]

            if self._compare_best(score):
                self.best_ckpt = self._init_ckpt_dict(idx_epoch, model, outputs, trainer)

                outputs.update({'epoch': idx_epoch})
                self.best_model_path = self.best_model_path_frame.format(**outputs)
                self.best_score = score

            if not self.save_best:
                ckpt = self._init_ckpt_dict(idx_epoch, model, outputs, trainer)

                outputs.update({'epoch': idx_epoch})
                model_path = self.best_model_path_frame.format(**outputs)
                torch.save(ckpt, model_path)

    def on_train_end(self, trainer, model, last_idx_epoch, last_outputs):
        if self.best_model_path != '':
            # save best ckpt
            if self.save_best:
                torch.save(self.best_ckpt, self.best_model_path)
                print('Save best checkpoint \'{}\''.format(self.best_model_path))

            # save last ckpt
            if self.save_last:
                ckpt = self._init_ckpt_dict(last_idx_epoch, model, last_outputs, trainer)

                last_idx_epoch += 1
                last_outputs.update({'epoch': last_idx_epoch})
                self.last_model_path = self.last_model_path_frame.format(**last_outputs)

                torch.save(ckpt, self.last_model_path)
                print('Save last checkpoint \'{}\''.format(self.last_model_path))
        else:
            print('No checkpoint save')

    @classmethod
    def load_checkpoint(cls, checkpoint_path, model_content=None):
        ckpt = torch.load(checkpoint_path)
        if type(ckpt['model']) != OrderedDict:
            model = ckpt['model']
        elif (model_content is not None) and not hasattr(model_content, '__name__'):
            # model_content是一个实例化的对象，直接将权重载入模型
            model = model_content
            model.load_state_dict(ckpt['model'])
        elif hasattr(model_content, '__name__') and model_content.__name__ == ckpt['model_name'] \
                and 'model_hparams' in ckpt.keys() and ckpt['model_hparams'] is not None:
            # model_content是一个类，需要通过超参数载入模型
            model = model_content(**ckpt['model_hparams'])
            model.load_state_dict(ckpt['model'])
        else:
            raise RuntimeError('Load model failed from checkpoint')

        return model, ckpt['epoch'], ckpt['outputs']


class TensorBoardLogger(Callback):
    def __init__(
            self,
            save_dir: str,
            version: Optional[Union[int, str]] = None,
            log_graph: bool = False,
            log_hparams: bool = False
    ):
        super(TensorBoardLogger, self).__init__()
        self.CALLBACK_PRIORITY = 1

        self.save_dir = save_dir
        self.version = version
        self.log_graph = log_graph
        self.log_hparams = log_hparams
        self.writer = None

    def _init_log_dir(self):
        version = str(self.version or 0)
        while True:
            dirpath = os.path.join(self.save_dir, '{}'.format(version))
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            elif self.version is not None:
                shutil.rmtree(dirpath)
                print('Delete path {}'.format(dirpath))
                continue
            else:
                version = version + 1
                continue
            break
        self.version = version
        return dirpath

    def on_train_start(self, trainer, model, step, train_dataloader, val_dataloader):
        dirpath = self._init_log_dir()
        print('The path of Tensorboard Logger is \'{}\''.format(dirpath))
        self.writer = SummaryWriter(dirpath)
        if self.log_graph:
            samples = train_dataloader.__iter__().__next__()
            x, _ = step.extract_input(samples)
            self.writer.add_graph(model, x)

    def on_train_epoch_end(self, trainer, model, idx_epoch, outputs):
        for k, v in outputs.items():
            self.writer.add_scalar('train/{}'.format(k), v, global_step=idx_epoch+1)

    def on_validation_epoch_end(self, trainer, model, idx_epoch, outputs):
        for k, v in outputs.items():
            self.writer.add_scalar('val/{}'.format(k), v, global_step=idx_epoch+1)

    def on_train_end(self, trainer, model, last_idx_epoch, last_outputs):
        if self.log_hparams and last_outputs:
            check = lambda v: True if isinstance(v, (int, float, str, bool)) else False
            hparam_dict = trainer.hparams
            hparam_dict.update(model.hparams if hasattr(model, 'hparams') else {})
            hparam_dict = {'{}'.format(k): v for k, v in hparam_dict.items() if check(v)}
            metric_dict = {'hparam/{}'.format(k): v for k, v in last_outputs.items()}
            self.writer.add_hparams(hparam_dict, metric_dict)
            print('Saved hyper-parameters as follow: {}.'.format(hparam_dict))
            print('Saved metric as follow: {}.'.format(metric_dict))

        if self.writer is not None:
            # self.writer.flush()
            self.writer.close()

        print('Tensorboard logger already closed.')


class TimeSeriesPlotter(Callback):
    def __init__(self, plt, horizons=[-1], save_dir=None, prefix='default'):
        super(TimeSeriesPlotter, self).__init__()
        self.plt = plt
        self.horizons = horizons
        self.save_dir = save_dir
        self.prefix = prefix

    def _init_plot_path(self, filename):
        if self.save_dir is None:
            return None

        dirpath = self.save_dir
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = os.path.join(dirpath, "{}-{}.jpg".format(self.prefix, filename))
        return filepath

    def _plotseries(self, se, title=None, filepath=None):
        self.plt.figure()
        self.plt.plot(se)
        self.plt.title(title)

        if filepath is not None:
            self.plt.savefig(filepath)
        else:
            self.plt.show()

    def _plotseries_fit(self, preds, trues, labels=None, title=None, filepath=None, dpi=100):
        self.plt.ioff()
        self.plt.figure(dpi=dpi)
        y_true = np.concatenate(trues)
        self.plt.plot(range(len(y_true)), y_true, label="True")

        start = 0
        for i, (pred, true) in enumerate(zip(preds, trues)):
            if len(pred) != len(true):
                raise RuntimeError('length of time series not matching')
            end = start + len(pred)
            self.plt.plot(range(start, end), pred, label=None if labels is None else labels[i])
            self.plt.legend(loc='best')
            self.plt.title(title)
            start = end

        if filepath is not None:
            self.plt.savefig(filepath)
        else:
            self.plt.show()

    def _plotseries_compare(self, tss: list, labels: list, title=None, filepath=None):
        if len(np.unique(list(map(lambda ts: len(ts), tss)))) != 1:
            raise RuntimeError('length of time series not matching')
        self.plt.ioff()
        self.plt.figure()
        for ts, label in zip(tss, labels):
            self.plt.plot(range(len(ts)), ts, label=label)
        self.plt.title(title)
        self.plt.legend(loc='best')

        if filepath is not None:
            self.plt.savefig(filepath)
        else:
            self.plt.show()

    def on_test_end(self, trainer, model, preds: np.ndarray, trues: np.ndarray):
        if preds.shape != trues.shape:
            raise RuntimeError('The size of preds and trues not match!')
        n_samples, n_horizon, n_features = preds.shape
        # plot single horizon fit curve
        for i in range(n_features):
            for h in self.horizons:
                idx_horizon = h + 1 if h > 0 else n_horizon + h + 1
                idx_feature = i + 1

                title = 'The {}\'th horizon fit curve of {}\'th feature'.format(idx_horizon, idx_feature)
                filename = 'single-horizon-{}-{}'.format(idx_horizon, idx_feature)
                filepath = self._init_plot_path(filename)

                self._plotseries_compare(tss=[preds[:, h, i], trues[:, h, i]],
                                         labels=['preds', 'trues'],
                                         title=title,
                                         filepath=filepath)

        # if self.save_dir is not None:
        #     print('Save single horizon plot success.')



