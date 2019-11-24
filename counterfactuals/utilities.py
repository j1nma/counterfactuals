import argparse

import mxnet as mx
import numpy as np
from mxnet import gluon, autograd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filename):
    if filename[-3:] != 'npz':
        raise Exception("Data file format is not npz.")

    data_in = np.load(filename)
    data = {'x': data_in['x'],
            't': data_in['t'],
            'yf': data_in['yf'],
            'ycf': data_in['ycf'],
            'mu0': data_in['mu0'],
            'mu1': data_in['mu1']}

    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]

    # Adjust binary feature at index 13: {1, 2} -> {0, 1}
    data['x'][:, 13] -= 1

    # Normalize the continuous features
    for experiment in range(data['x'][:, :6, :].shape[2]):
        data['x'][:, :6, experiment] = StandardScaler().fit_transform(data['x'][:, :6, experiment])

    return data


def split_data_in_train_valid_test(x, t, yf, ycf, mu0, mu1, test_size=0.1, validation_size=0.27, seed=1):
    """ Split train data into train, validation and test indexes.
        63/27/10 train/validation/test split
    """
    train_valid_idx, test_idx = train_test_split(np.arange(x.shape[0]), test_size=test_size, random_state=seed)
    train_idx, valid_idx = train_test_split(train_valid_idx, test_size=validation_size, random_state=seed)

    train = {'x': x[train_idx],
             't': t[train_idx],
             'yf': yf[train_idx],
             'ycf': ycf[train_idx],
             'mu0': mu0[train_idx],
             'mu1': mu1[train_idx]}

    valid = {'x': x[valid_idx],
             't': t[valid_idx],
             'yf': yf[valid_idx],
             'ycf': ycf[valid_idx],
             'mu0': mu0[valid_idx],
             'mu1': mu1[valid_idx]}

    test = {'x': x[test_idx],
            't': t[test_idx],
            'yf': yf[test_idx],
            'ycf': ycf[test_idx],
            'mu0': mu0[test_idx],
            'mu1': mu1[test_idx]}

    return train, valid, test


def test_net(net, test_data, ctx):
    """ Test data on net and get metric (RMSE as default). """
    metric = mx.metric.RMSE()
    metric.reset()
    for i, (data, label) in enumerate(test_data):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, even_split=False)
        label = gluon.utils.split_and_load(label, ctx_list=ctx, even_split=False)
        with autograd.predict_mode():
            outputs = [net(x) for x in data]
        metric.update(label, outputs)
    return metric.get()


def predict_treated_and_controlled(net, test_rmse_ite_loader, ctx):
    """ Predict treated and controlled outcomes. """

    y_t0 = np.array([])
    y_t1 = np.array([])
    for i, (x) in enumerate(test_rmse_ite_loader):
        x = gluon.utils.split_and_load(x, ctx_list=ctx, even_split=False)

        t0_features = mx.nd.concat(x[0], mx.nd.zeros((len(x[0]), 1)))
        t1_features = mx.nd.concat(x[0], mx.nd.ones((len(x[0]), 1)))

        t0_controlled_predicted = net(t0_features)
        t1_treated_predicted = net(t1_features)

        y_t0 = np.append(y_t0, t0_controlled_predicted)
        y_t1 = np.append(y_t1, t1_treated_predicted)

    return y_t0, y_t1


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        help="Number of epochs per experiment."
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.001,
        help="Initial learning rate."
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        default=0.0001,
        help="L2 weight decay lambda."
    )
    parser.add_argument(
        "-is",
        "--input_size",
        default=26,
        help="Neural network input size."
    )
    parser.add_argument(
        "-hs",
        "--hidden_size",
        default=25,
        help="Number of hidden nodes per layer."
    )
    parser.add_argument(
        "-te",
        "--train_experiments",
        default=10,
        help="Number of train experiments/replications from train data."
    )
    parser.add_argument(
        "-lf",
        "--learning_rate_factor",
        default=0.96
    )
    parser.add_argument(
        "-ls",
        "--learning_rate_steps",
        default=2000,
        help="Changes the learning rate for every given number of updates."
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        default=2,
        help="Number of cores."
    )
    parser.add_argument(
        "-bs",
        "--batch_size_per_unit",
        default=32,
        help="Mini-batch size per processing unit."
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=1,
        help="Random seed."
    )
    parser.add_argument(
        "-od",
        "--outdir",
        default='results/ihdp'
    )
    parser.add_argument(
        "-dd",
        "--data_dir",
        default='data/'
    )
    parser.add_argument(
        "-td",
        "--data_train",
        default='ihdp_npci_1-100.train.npz',
        help="Train data npz file."
    )
    parser.add_argument(
        "-sd",
        "--data_test",
        default='ihdp_npci_1-100.test.npz',
        help="Test data npz file."
    )
    parser.add_argument(
        "-a",
        "--architecture",
        default='nn4',
        choices=['nn4', 'bnn'],
        help="Neural network architecture to use."
    )
    parser.add_argument(
        "-ms",
        "--means_stds",
        default='ihdp_means_stds.nd',
        help="Saved means and stds from training. Inside an outdir folder."
    )
    parser.add_argument(
        "-sy",
        "--symbol",
        default='ihdp-predictions-symbol.json',
        help="Saved symbol json file from training. Inside an outdir folder."
    )
    parser.add_argument(
        "-ps",
        "--params",
        default='ihdp-predictions-0100.params',
        help="Parameter dictionary for arguments and auxiliary states of outputs that are not inputs."
             "File from training. Inside an outdir folder."
    )

    return parser
