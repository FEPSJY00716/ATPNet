from dltools.trainer import Trainer, TimeSeriesStep, setup_seed
from dltools.callbacks import ModelCheckpoint, TensorBoardLogger, TimeSeriesPlotter
from dltools.dataload import TsDataMouleFromNumpy
from model.model import MPTCN

import matplotlib.pyplot as plt
import os
import torch
import itertools
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser


def start(data_name, model_name, horizon, n_seq, save_dir='out', on_script=False,
          epoch=1000, batch_size=128, lr=0.001, one_step=False, on_time=False,
          pin_memory=False, num_workers=0, multi_gpu=False, **hparams):
    # print(data_name, model_name, horizon, nl_seq, ns_seq, save_dir)
    # print(on_val, epoch, batch_size, lr, one_step, on_time, pin_memory, num_workers, multi_gpu)
    # print(hparams)
    # return 0
    setup_seed(2021)

    if data_name == 'AQ2021_WD':
        dm = TsDataMouleFromNumpy('data/AQ2021_WD.npz', time_window=n_seq, horizon=horizon,
                                  batch_size=batch_size, one_step_model=one_step,
                                  pin_memory=pin_memory, num_workers=num_workers,
                                  data_splite=(-1, 204, 0),
                                  time_flag=on_time)
        n_features = 9
    elif data_name == 'AQ2021_TS':
        dm = TsDataMouleFromNumpy('data/AQ2021_TS.npz', time_window=n_seq, horizon=horizon,
                                  batch_size=batch_size, one_step_model=one_step,
                                  pin_memory=pin_memory, num_workers=num_workers,
                                  data_splite=(-1, 204, 0),
                                  time_flag=on_time)
        n_features = 9
    else:
        raise RuntimeError('No dataset {}'.format(data_name))

    model = MPTCN(dm.time_window, dm.horizon, n_features)

    if multi_gpu:
        model = torch.nn.DataParallel(model)

    plotter = TimeSeriesPlotter(plt)
    trainer = Trainer(epoch, 'cuda', [plotter], lr=lr, w_decay=0, clip=5, verbose=True)

    step = TimeSeriesStep(rescale=dm.scaler_y.inverse_transfrom_from_tensor)

    trainer.fit(model, step, dm.train_dataloader(shuffle=True), dm.val_dataloader(shuffle=False))

    train_preds, train_trues = trainer.test(model, step, dm.train_dataloader(shuffle=False))
    val_preds, val_trues = trainer.test(model, step, dm.val_dataloader(shuffle=False))

    pred = val_preds[:, -1, 0].cpu()
    ture = val_trues[:, -1, 0].cpu()

    out = pd.DataFrame({"pred": pred, "true": ture})
    out.to_csv("out/results.csv")

    plt.ioff()
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(pred)), pred, label='pred')
    plt.plot(range(len(ture)), ture, label='true')
    plt.title('Air temperature')
    plt.legend(loc='best')
    plt.savefig('out/temperature.png', dpi=300)
    # plt.show()
    # test_preds, test_trues = trainer.test(model, step, dm.test_dataloader(shuffle=False))


if __name__ == '__main__':
    parse = ArgumentParser()
    parse.add_argument('--data_name', type=str, default='AQ2021_TS')
    parse.add_argument('--model_name', type=str, default='MPTCN')

    parse.add_argument('--n_seq', type=int, default=48)
    parse.add_argument('--horizon', type=int, default=3)

    parse.add_argument('--pin_memory', dest='pin_memory', action='store_true')
    parse.add_argument('--num_workers', type=int, default=0)
    parse.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parse.add_argument('--batch_size', type=int, default=128)
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--epoch', type=int, default=200)
    parse.add_argument('--save_dir', type=str, default='out/dmsnet')
    parse.add_argument('--on_script', dest='on_script', action='store_true')

    args = parse.parse_args()
    # print(vars(args))
    start(**vars(args))
