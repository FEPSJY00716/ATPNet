import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch

from dltools.trainer import Trainer, TimeSeriesStep, setup_seed
from dltools.callbacks import ModelCheckpoint, TensorBoardLogger, TimeSeriesPlotter
from dltools.dataload import TsDataMouleFromNumpy
from model.baseline import AR, MLP, RNN, Seq2Seq, TCN


def start(data_name, model_name, horizon, n_seq, save_dir='out', on_script=True,
          epoch=1000, batch_size=128, lr=0.001, one_step=False,
          pin_memory=False, num_workers=0, multi_gpu=False, **hparams):
    # print(data_name, model_name, horizon, n_seq, save_dir)
    # print(on_val, epoch, batch_size, lr, one_step, pin_memory, num_workers, multi_gpu)
    # print(hparams)
    # return 0
    setup_seed(2021)

    if data_name == 'AQ2021_WD':
        dm = TsDataMouleFromNumpy('data/AQ2021_WD.npz', time_window=n_seq, horizon=horizon,
                                  batch_size=batch_size, one_step_model=one_step,
                                  pin_memory=pin_memory, num_workers=num_workers,
                                  data_splite=(-1, 204, 0))
        n_features = 9
    elif data_name == 'AQ2021_TS':
        dm = TsDataMouleFromNumpy('data/AQ2021_TS.npz', time_window=n_seq, horizon=horizon,
                                  batch_size=batch_size, one_step_model=one_step,
                                  pin_memory=pin_memory, num_workers=num_workers,
                                  data_splite=(-1, 204, 0))
        n_features = 9
    else:
        raise RuntimeError('No dataset {}'.format(data_name))

    # model
    h = 1 if one_step else horizon
    if model_name == 'AR':
        model = AR(h, n_seq)
    elif model_name == 'MLP':
        n_hidden = hparams['n_hidden']
        model = MLP(h, n_seq, n_hidden=n_hidden)
    elif model_name == 'RNN':
        n_hidden = hparams['n_hidden']
        n_layers = hparams['n_layers']
        on_exogenous = hparams['on_exogenous']
        whole_output = hparams['whole_output']
        model = RNN(h, n_seq, n_hidden=n_hidden, n_layers=n_layers,
                    on_exogenous=on_exogenous, nx=n_features, whole_output=whole_output)
    elif model_name == 'SEQ':
        n_hidden = hparams['n_hidden']
        n_layers = hparams['n_layers']
        on_exogenous = hparams['on_exogenous']
        model = Seq2Seq(h, n_seq, n_hidden=n_hidden, n_layers=n_layers,
                        on_exogenous=on_exogenous, nx=n_features)
    elif model_name == 'TCN':
        n_hidden = hparams['n_hidden']
        n_layers = hparams['n_layers']
        kernel_size = hparams['kernel_size']
        on_exogenous = hparams['on_exogenous']
        model = TCN(h, n_seq, n_channels=[n_hidden] * n_layers, kernel_size=kernel_size,
                    on_exogenous=on_exogenous, nx=n_features)
    else:
        raise RuntimeError('No model {}'.format(model_name))

    if multi_gpu:
        model = torch.nn.DataParallel(model)

    subdir = '{}-{}-{}-{}'.format(data_name, model_name, horizon, n_seq)
    filename = ''
    for value in hparams.values():
        filename = filename + '{}-'.format(value)
    filename = filename[:-1]

    # trainer
    if on_script:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        checkpoint = ModelCheckpoint(os.path.join(save_dir, 'cp', subdir), prefix=filename, save_hp=True)
        logger = TensorBoardLogger(os.path.join(save_dir, 'log', subdir), version=filename)
        # trainer = Trainer(epoch, 'cuda', [checkpoint, logger], lr=lr, w_decay=0, clip=5, verbose=False)
        trainer = Trainer(epoch, 'cuda', [checkpoint], lr=lr, w_decay=0, clip=5, verbose=False)
    else:
        print(subdir, filename)
        plotter = TimeSeriesPlotter(plt)
        trainer = Trainer(epoch, 'cuda', [plotter], lr=lr, w_decay=0, clip=5, verbose=True)

    step = TimeSeriesStep(rescale=dm.scaler_y.inverse_transfrom_from_tensor)
    trainer.fit(model, step, dm.train_dataloader(shuffle=True), dm.val_dataloader(shuffle=False))

    train_preds, train_trues = trainer.test(model, step, dm.train_dataloader(shuffle=False))
    val_preds, val_trues = trainer.test(model, step, dm.val_dataloader(shuffle=False))
    # test_preds, test_trues = trainer.test(model, step, dm.test_dataloader(shuffle=False))


if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help='model describe')

    parser.add_argument('--data_name', type=str, default='AQ2021_WD')
    parser.add_argument('--save_dir', type=str, default='out/baseline')

    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=1000)

    parser.add_argument('--n_seq', type=int, default=84)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--on_script', dest='on_script', action='store_true')

    ar = subparsers.add_parser('ar', help='Autoregression')
    ar.add_argument('--model_name', type=str, default='AR')

    mlp = subparsers.add_parser('mlp', help='Multi-Layer Perception')
    mlp.add_argument('--model_name', type=str, default='MLP')
    mlp.add_argument('--n_hidden', type=int, default=128)

    rnn = subparsers.add_parser('rnn', help='Recurrent Neural Network')
    rnn.add_argument('--model_name', type=str, default='RNN')
    rnn.add_argument('--n_hidden', type=int, default=64)
    rnn.add_argument('--n_layers', type=int, default=1)
    rnn.add_argument('--on_exogenous', type=bool, default=False)
    rnn.add_argument('--whole_output', type=bool, default=True)
    rnn.add_argument('--n_seq', type=int, default=24)

    seq = subparsers.add_parser('seq', help='Sequence to Sequence')
    seq.add_argument('--model_name', type=str, default='SEQ')
    seq.add_argument('--n_hidden', type=int, default=32)
    seq.add_argument('--n_layers', type=int, default=1)
    seq.add_argument('--on_exogenous', type=bool, default=False)

    tcn = subparsers.add_parser('tcn', help='Temporal Convolutional Neural Network')
    tcn.add_argument('--model_name', type=str, default='TCN')
    tcn.add_argument('--n_hidden', type=int, default=64)  # 3,6:32; 12:64
    tcn.add_argument('--n_layers', type=int, default=3)
    tcn.add_argument('--kernel_size', type=int, default=3)
    tcn.add_argument('--on_exogenous', type=bool, default=True)

    args = parser.parse_args()
    if 'model_name' not in vars(args):
        print('You need to specify the model. Use the argument \'-h\' for details.')
    else:
        print(args.data_name, args.model_name)
        start(**vars(args))
