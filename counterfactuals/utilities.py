import argparse

import mxnet as mx
import numpy as np
from mxnet import gluon, autograd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from counterfactuals.cfr.net import WassersteinLoss

SQRT_CONST = 1e-10


def mx_pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * mx.nd.dot(X, mx.nd.transpose(Y))
    nx = mx.nd.sum(mx.nd.square(X), 1, keepdims=True)
    ny = mx.nd.sum(mx.nd.square(Y), 1, keepdims=True)
    D = (C + mx.nd.transpose(ny)) + nx

    return D


def mx_safe_sqrt(x, lbound=SQRT_CONST):
    """ Numerically safe version based on TensorFlow sqrt. """

    return mx.nd.sqrt(mx.nd.clip(x, lbound, np.inf))


def log(logfile, s):
    """ Log a string into a file and print it. """
    with open(logfile, 'a', encoding='utf8') as f:
        f.write(s + '\n')
    print(s)


def load_data(filename, normalize=False):
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

    if normalize:
        # Adjust binary feature at index 13: {1, 2} -> {0, 1}
        data['x'][:, 13] -= 1

        # Normalize the continuous features
        for experiment in range(data['x'][:, :6, :].shape[2]):
            data['x'][:, :6, experiment] = StandardScaler().fit_transform(data['x'][:, :6, experiment])

    return data


def save_config(filename, FLAGS):
    """ Save configuration file """
    flag_dictionary = FLAGS.__dict__
    s = '\n'.join(['%s: %s' % (k, str(flag_dictionary[k])) for k in sorted(flag_dictionary.keys())])
    f = open(filename, 'w')
    f.write(s)
    f.close()


def split_data_in_train_valid(x, t, yf, ycf, mu0, mu1, validation_size=0.27, seed=1):
    """ Split train data into train and validation indices.
        Default 73/27 train/validation split.
    """
    train_idx, valid_idx = train_test_split(np.arange(x.shape[0]), test_size=validation_size, random_state=seed)

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

    return train, valid


def split_data_in_train_valid_test(x, t, yf, ycf, mu0, mu1, test_size=0.1, validation_size=0.27, seed=1):
    """ Split train data into train, validation and test indices.
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

    return train, valid, test, valid_idx


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


def hybrid_test_net_with_cfr(net, test_data_loader, ctx, FLAGS, p_treated):
    """ Test data on t1_net and t0_net for CFR and get metric (RMSE as default). """
    metric = mx.metric.RMSE()
    metric.reset()

    l2_loss = gluon.loss.L2Loss()
    wass_loss = WassersteinLoss(lam=FLAGS.wass_lambda,
                                its=FLAGS.wass_iterations,
                                square=True, backpropT=FLAGS.wass_bpg)
    obj_loss = 0
    imb_err = 0

    for i, (x, t, batch_yf) in enumerate(test_data_loader):
        x = x.as_in_context(ctx)
        t = t.as_in_context(ctx)
        batch_yf = batch_yf.as_in_context(ctx)

        # Get treatment and control indices
        t1_idx = np.where(x[:, -1] == 1)[0]
        t0_idx = np.where(x[:, -1] == 0)[0]

        # Compute sample reweighing
        if FLAGS.reweight_sample:
            w_t = t / (2 * p_treated)
            w_c = (1 - t) / (2 * 1 - p_treated)
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        ''' Initialize outputs '''
        outputs = np.zeros(batch_yf.shape)
        loss = np.zeros(batch_yf.shape)

        with autograd.predict_mode():
            t1_o, t0_o, rep_o = net(x, mx.nd.array(t1_idx), mx.nd.array(t0_idx))

            risk = 0

            t1_o_loss = l2_loss(t1_o, batch_yf[t1_idx], sample_weight[t1_idx])
            np.put(loss, t1_idx, t1_o_loss.asnumpy())
            np.put(outputs, t1_idx, t1_o.asnumpy())
            risk = risk + t1_o_loss.sum()

            t0_o_loss = l2_loss(t0_o, batch_yf[t0_idx], sample_weight[t0_idx])
            np.put(loss, t0_idx, t0_o_loss.asnumpy())
            np.put(outputs, t0_idx, t0_o.asnumpy())
            risk = risk + t0_o_loss.sum()

            if FLAGS.normalization == 'divide':
                h_rep_norm = rep_o / mx_safe_sqrt(mx.nd.sum(mx.nd.square(rep_o), axis=1, keepdims=True))
            else:
                h_rep_norm = 1.0 * rep_o

            imb_dist = wass_loss(h_rep_norm[t1_idx], h_rep_norm[t0_idx])

            imb_error = FLAGS.p_alpha * imb_dist

            tot_error = risk

            if FLAGS.p_alpha > 0:
                tot_error = tot_error + imb_error

        metric.update(batch_yf, mx.nd.array(outputs))

        obj_loss += tot_error
        imb_err += imb_error

    return metric.get(), obj_loss, imb_err


def predict_treated_and_controlled(net, test_rmse_ite_loader, ctx):
    """ Predict treated and controlled outcomes. """

    y_t0 = np.array([])
    y_t1 = np.array([])
    for i, (x) in enumerate(test_rmse_ite_loader):
        x = gluon.utils.split_and_load(x, ctx_list=ctx, even_split=False)

        t0_features = mx.nd.concat(x[0], mx.nd.zeros((len(x[0]), 1)))
        t1_features = mx.nd.concat(x[0], mx.nd.ones((len(x[0]), 1)))

        with autograd.predict_mode():
            t0_controlled_predicted = net(t0_features)
            t1_treated_predicted = net(t1_features)

        y_t0 = np.append(y_t0, t0_controlled_predicted)
        y_t1 = np.append(y_t1, t1_treated_predicted)

    return y_t0, y_t1


def predict_treated_and_controlled_vb(net, test_rmse_ite_loader, layer_params, ctx):
    """ Predict treated and controlled outcomes. """

    y_t0 = np.array([])
    y_t1 = np.array([])
    for i, (x) in enumerate(test_rmse_ite_loader):
        x = gluon.utils.split_and_load(x, ctx_list=ctx, even_split=False)

        t0_features = mx.nd.concat(x[0], mx.nd.zeros((len(x[0]), 1)))
        t1_features = mx.nd.concat(x[0], mx.nd.ones((len(x[0]), 1)))

        for l_param, param in zip(layer_params, net.collect_params().values()):
            param._data[0] = l_param

        with autograd.predict_mode():
            t0_controlled_predicted = net(t0_features)
            t1_treated_predicted = net(t1_features)

        y_t0 = np.append(y_t0, t0_controlled_predicted)
        y_t1 = np.append(y_t1, t1_treated_predicted)

    return y_t0, y_t1


def predict_treated_and_controlled_with_cfr(net, data_loader, ctx):
    """ Predict treated and controlled outcomes. """

    y_t1 = np.array([])
    y_t0 = np.array([])

    for i, (x, _, _) in enumerate(data_loader):
        x = x.as_in_context(ctx)

        with autograd.predict_mode():
            t1_treated_predicted, t0_controlled_predicted, _ = net(x, mx.nd.arange(len(x)), mx.nd.arange(len(x)))

        y_t1 = np.append(y_t1, t1_treated_predicted)
        y_t0 = np.append(y_t0, t0_controlled_predicted)

    return y_t0, y_t1


def hybrid_predict_treated_and_controlled_with_cfr(net, data_loader, ctx):
    """ Predict treated and controlled outcomes. """

    y_t1 = np.array([])
    y_t0 = np.array([])

    for i, (x, _, _) in enumerate(data_loader):
        x = x.as_in_context(ctx)

        with autograd.predict_mode():
            t1_treated_predicted, t0_controlled_predicted, _ = net(x, mx.nd.arange(len(x)), mx.nd.arange(len(x)))

        y_t1 = np.append(y_t1, t1_treated_predicted)
        y_t0 = np.append(y_t0, t0_controlled_predicted)

    return y_t0, y_t1


def predict_treated_and_controlled_with_cnn(net, test_rmse_ite_loader, ctx):
    """ Predict treated and controlled outcomes with CNN architecture modifications. """

    y_t0 = np.array([])
    y_t1 = np.array([])
    for i, (x) in enumerate(test_rmse_ite_loader):
        x = gluon.utils.split_and_load(x, ctx_list=ctx, even_split=False)

        t0_features = mx.nd.concat(x[0].reshape(len(x[0]), x[0].shape[2], 1), mx.nd.zeros((len(x[0]), 1, 1)))
        t1_features = mx.nd.concat(x[0].reshape(len(x[0]), x[0].shape[2], 1), mx.nd.ones((len(x[0]), 1, 1)))

        t0_features = t0_features.reshape(len(x[0]), 1, x[0].shape[2])
        t1_features = t1_features.reshape(len(x[0]), 1, x[0].shape[2])

        t0_controlled_predicted = net(t0_features)
        t1_treated_predicted = net(t1_features)

        y_t0 = np.append(y_t0, t0_controlled_predicted)
        y_t1 = np.append(y_t1, t1_treated_predicted)

    return y_t0, y_t1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected, received:\t' + str(type(bool)))


def get_parent_args_parser():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "-e",
        "--experiments",
        default=2,
        type=int,
        help="Number of experiments."
    )
    parent_parser.add_argument(
        "-i",
        "--iterations",
        default=3000,
        type=int,
        help="Number of iterations or epochs."
    )
    parent_parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.001,
        type=float,
        help="Initial learning rate."
    )
    parent_parser.add_argument(
        "-lf",
        "--learning_rate_factor",
        default=0.97,
        type=float,
        help="Learning rate factor."
    )
    parent_parser.add_argument(
        "-ls",
        "--learning_rate_steps",
        default=2000,
        help="Changes the learning rate for every given number of updates."
    )
    parent_parser.add_argument(
        "-od",
        "--outdir",
        default='results/ihdp'
    )
    parent_parser.add_argument(
        "-rsd",
        "--results_dir",
        default='results/ihdp',
        help="Only when testing, to later find mean, stds, params and symbol files."
    )
    parent_parser.add_argument(
        "-dd",
        "--data_dir",
        default='../data'
    )
    parent_parser.add_argument(
        "-td",
        "--data_train",
        default='ihdp_npci_1-100.train.npz',
        help="Train data npz file."
    )
    parent_parser.add_argument(
        "-sd",
        "--data_test",
        default='ihdp_npci_1-100.test.npz',
        help="Test data npz file."
    )
    parent_parser.add_argument(
        "-s",
        "--seed",
        default=1,
        type=int,
        help="Random seed."
    )
    parent_parser.add_argument(
        "-w",
        "--num_workers",
        default=2,
        type=int,
        help="Number of cores."
    )
    parent_parser.add_argument(
        "-bs",
        "--batch_size_per_unit",
        default=32,
        type=int,
        help="Mini-batch size per processing unit."
    )
    parent_parser.add_argument(
        "-wd",
        "--weight_decay",
        default=0.0001,
        type=float,
        help="Weight decay L2 regularization parameter."
    )
    parent_parser.add_argument(
        "-ei",
        "--epoch_output_iter",
        default=10,
        type=int,
        help="Print results after given number of epochs."
    )

    return parent_parser


def get_nn_args_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', parents=[get_parent_args_parser()])
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
        "-a",
        "--architecture",
        default='nn4',
        choices=['nn4', 'cnn', 'nn4_vb'],
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


def get_cfr_args_parser():
    cfr_parser = argparse.ArgumentParser(fromfile_prefix_chars='@', parents=[get_parent_args_parser()])
    cfr_parser.add_argument(
        "-il",
        "--rep_lay",
        default=2,
        type=int,
        help="Number of representation layers."
    )
    cfr_parser.add_argument(
        "-ol",
        "--reg_lay",
        default=2,
        type=int,
        help="Number of regression layers."
    )
    cfr_parser.add_argument(
        "-a",
        "--p_alpha",
        default=0,
        type=float,
        help="Imbalance penalty."
    )
    cfr_parser.add_argument(
        '-rd',
        '--rep_weight_decay',
        default=0,
        type=int,
        help='Whether to penalize representation layers with weight decay.'
    )
    cfr_parser.add_argument(
        "-di",
        "--dropout_in",
        default=1.0,
        type=float,
        help="Dropout keep rate of input layers."
    )
    cfr_parser.add_argument(
        "-do",
        "--dropout_out",
        default=1.0,
        type=float,
        help="Dropout keep rate of output layers."
    )
    cfr_parser.add_argument(
        "-pd",
        "--rms_prop_decay",
        default=0.3,
        type=float,
        help="RMSProp decay."
    )
    cfr_parser.add_argument(
        "-id",
        "--dim_rep",
        default=200,
        type=int,
        help="Dimension of representation layers."
    )
    cfr_parser.add_argument(
        "-hd",
        "--dim_hyp",
        default=100,
        type=int,
        help="Dimension of hypothesis layers."
    )
    cfr_parser.add_argument(
        '-bn',
        '--batch_norm',
        type=str2bool,
        nargs='?',
        const=False,
        help='Whether to use batch-normalization.'
    )
    cfr_parser.add_argument(
        "-nr",
        "--normalization",
        default='divide',
        choices=['none', 'bn_fixed', 'divide', 'project'],
        help="How to normalize representation after batch-normalization."
    )
    cfr_parser.add_argument(
        "-ws",
        "--weight_init_scale",
        default=0.1,
        type=float,
        help="Weight initialization scale."
    )
    cfr_parser.add_argument(
        "-wi",
        "--wass_iterations",
        default=10,
        type=int,
        help="Number of iterations in Wasserstein computation."
    )
    cfr_parser.add_argument(
        "-wl",
        "--wass_lambda",
        default=10.0,
        type=float,
        help="Wasserstein lambda."
    )
    cfr_parser.add_argument(
        '-wb',
        '--wass_bpg',
        default=1,
        type=int,
        help='Whether to backpropagate through T matrix.'
    )
    cfr_parser.add_argument(
        '-oc',
        '--output_csv',
        default=0,
        type=int,
        help='Whether to save a CSV file with the results.'
    )
    cfr_parser.add_argument(
        "-oy",
        "--output_delay",
        default=100,
        type=int,
        help="Number of iterations between log/loss outputs."
    )
    cfr_parser.add_argument(
        "-pi",
        "--pred_output_iter",
        default=200,
        type=int,
        help="Number of delay iterations between prediction outputs. (-1 gives no intermediate output)."
    )
    cfr_parser.add_argument(
        "-v",
        "--val_size",
        default=0.27,
        type=float,
        help="Validation part. Note that this should be small enough to have t=1 samples for validation/testing."
    )
    cfr_parser.add_argument(
        '-so',
        '--split_output',
        type=str2bool,
        nargs='?',
        const=True,
        help='Whether to split output layers between treated and control.'
    )
    cfr_parser.add_argument(
        '-rw',
        '--reweight_sample',
        type=str2bool,
        nargs='?',
        const=True,
        help='Whether to reweight sample for prediction loss with average treatment probability.'
    )
    cfr_parser.add_argument(
        "-ni",
        "--normalize_input",
        type=str2bool,
        nargs='?',
        const=True,
        help='Whether to normalize input.'
    )

    return cfr_parser
