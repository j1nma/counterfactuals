import datetime
import os
import pathlib
import random
import sys
import time
import traceback

import mxnet as mx
import numpy as np
from mxnet import gluon, autograd
from scipy.stats import sem

from counterfactuals.cfr.net import CFRNet, WassersteinLoss
from counterfactuals.evaluation import Evaluator
from counterfactuals.utilities import log, load_data, get_cfr_args_parser, \
    split_data_in_train_valid, hybrid_test_net_with_cfr, \
    hybrid_predict_treated_and_controlled_with_cfr, mx_safe_sqrt, save_config
from examples.mxnet.tsne_plot import tsne_plot_pca

FLAGS = 0


def mx_run(outdir):
    """ Runs a set of training and validation experiments and stores result in a directory. """

    ''' Set up paths and start log '''
    logfile = outdir + 'log.txt'
    f = open(logfile, 'w')
    f.close()

    ''' Hyperparameters '''
    epochs = int(FLAGS.iterations)
    learning_rate = float(FLAGS.learning_rate)
    wd = float(FLAGS.weight_decay)
    train_experiments = int(FLAGS.experiments)
    learning_rate_factor = float(FLAGS.learning_rate_factor)
    learning_rate_steps = int(FLAGS.learning_rate_steps)  # changes the learning rate for every n updates.

    ''' Logging '''
    logfile = outdir + 'log.txt'
    f = open(logfile, 'w')
    f.close()
    data_train = FLAGS.data_dir + FLAGS.data_train
    data_train_valid = FLAGS.data_dir + FLAGS.data_test

    ''' Set GPUs/CPUs '''
    num_gpus = mx.context.num_gpus()
    num_workers = int(FLAGS.num_workers)  # replace num_workers with the number of cores
    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    units = num_gpus if num_gpus > 0 else 1
    batch_size_per_unit = int(FLAGS.batch_size_per_unit)  # mini-batch size
    batch_size = batch_size_per_unit * max(units, 1)

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    mx.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir + 'config.txt', FLAGS)

    log(logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha, FLAGS.weight_decay))

    ''' Load datasets '''
    train_dataset = load_data(data_train, normalize=FLAGS.normalize_input)

    log(logfile, 'Training data: ' + data_train)
    log(logfile, 'Valid data:     ' + data_train_valid)
    log(logfile, 'Loaded data with shape [%d,%d]' % (train_dataset['n'], train_dataset['dim']))

    ''' CFR Neural Network Architecture for ITE estimation '''
    net = CFRNet(FLAGS.dim_rep, FLAGS.dim_hyp, FLAGS.weight_init_scale, train_dataset['dim'], FLAGS.batch_norm)

    ''' Instantiate net '''
    net.initialize(ctx=ctx)
    net.hybridize()  # hybridize for better performance

    ''' Metric, Loss and Optimizer '''
    rmse_metric = mx.metric.RMSE()
    l2_loss = gluon.loss.L2Loss()
    wass_loss = WassersteinLoss(lam=FLAGS.wass_lambda,
                                its=FLAGS.wass_iterations,
                                square=True, backpropT=FLAGS.wass_bpg)  # Change too at hybrid_test_net_with_cfr
    scheduler = mx.lr_scheduler.FactorScheduler(step=learning_rate_steps, factor=learning_rate_factor,
                                                base_lr=learning_rate)
    optimizer = mx.optimizer.Adam(learning_rate=learning_rate, lr_scheduler=scheduler)
    # optimizer = mx.optimizer.Adam(learning_rate=learning_rate, lr_scheduler=scheduler, wd=wd)
    trainer = gluon.Trainer(net.collect_params(), optimizer=optimizer)

    ''' Initialize train score results '''
    train_scores = np.zeros((train_experiments, 3))

    ''' Initialize train experiment durations '''
    train_durations = np.zeros((train_experiments, 1))

    ''' Initialize valid score results '''
    valid_scores = np.zeros((train_experiments, 3))

    ''' Train experiments means and stds '''
    means = np.array([])
    stds = np.array([])

    ''' Train '''
    for train_experiment in range(train_experiments):

        ''' Create training dataset '''
        x = train_dataset['x'][:, :, train_experiment]
        t = np.reshape(train_dataset['t'][:, train_experiment], (-1, 1))
        yf = train_dataset['yf'][:, train_experiment]
        ycf = train_dataset['ycf'][:, train_experiment]
        mu0 = train_dataset['mu0'][:, train_experiment]
        mu1 = train_dataset['mu1'][:, train_experiment]

        train, valid = split_data_in_train_valid(x, t, yf, ycf, mu0, mu1, validation_size=FLAGS.val_size)

        ''' Plot first experiment original TSNE visualization '''
        if train_experiment == 0:
            ''' Learned representations of first experiment for TSNE visualization '''
            first_exp_reps = []

        ''' Train, Valid Evaluators, with labels not normalized '''
        train_evaluator = Evaluator(train['t'], train['yf'], train['ycf'], train['mu0'], train['mu1'])
        valid_evaluator = Evaluator(valid['t'], valid['yf'], valid['ycf'], valid['mu0'], valid['mu1'])

        ''' Normalize yf '''
        if FLAGS.normalize_input:
            yf_m, yf_std = np.mean(train['yf'], axis=0), np.std(train['yf'], axis=0)
            train['yf'] = (train['yf'] - yf_m) / yf_std
            valid['yf'] = (valid['yf'] - yf_m) / yf_std

            ''' Save mean and std '''
            means = np.append(means, yf_m)
            stds = np.append(stds, yf_std)

        ''' Train dataset '''
        train_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(train['x']), mx.nd.array(train['t']),
                                                        mx.nd.array(train['yf']))

        ''' Valid dataset '''
        valid_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(valid['x']), mx.nd.array(valid['t']),
                                                        mx.nd.array(valid['yf']))

        ''' Train DataLoader '''
        train_factual_loader = gluon.data.DataLoader(train_factual_dataset, batch_size=batch_size, shuffle=True,
                                                     num_workers=num_workers)

        ''' Valid DataLoader '''
        valid_factual_loader = gluon.data.DataLoader(valid_factual_dataset, batch_size=batch_size, shuffle=False,
                                                     num_workers=num_workers)

        number_of_batches = len(train_factual_loader)

        ''' Compute treatment probability '''
        treatment_probability = np.mean(train['t'])

        train_start = time.time()

        ''' Train model '''
        for epoch in range(1, epochs + 1):  # start with epoch 1 for easier learning rate calculation

            train_loss = 0
            rmse_metric.reset()
            obj_loss = 0
            imb_err = 0

            for i, (x, t, batch_yf) in enumerate(train_factual_loader):
                ''' Get data and labels into slices and copy each slice into a context. '''
                x = x.as_in_context(ctx)
                t = t.as_in_context(ctx)
                batch_yf = batch_yf.as_in_context(ctx)

                ''' Get treatment and control indices. Batch_size must be enough to have at least one t=1 sample '''
                t1_idx = np.where(t == 1)[0]
                t0_idx = np.where(t == 0)[0]

                if t1_idx.shape[0] == 0:
                    log(logfile, 'Encountered no treatment samples at batch ' + str(i) + '.')

                ''' Compute sample reweighing '''
                if FLAGS.reweight_sample:
                    w_t = t / (2 * treatment_probability)
                    w_c = (1 - t) / (2 * 1 - treatment_probability)
                    sample_weight = w_t + w_c
                else:
                    sample_weight = 1.0

                ''' Initialize outputs '''
                outputs = np.zeros(batch_yf.shape)
                loss = np.zeros(batch_yf.shape)

                ''' Forward (Factual) '''
                with autograd.record():
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

                    ''' Save last epoch of first experiment reps for TSNE vis. '''
                    if train_experiment == 0 and epoch == range(epochs + 1)[-1]:
                        first_exp_reps.extend(rep_o)

                ''' Backward '''
                tot_error.backward()

                ''' Optimize '''
                trainer.step(batch_size)

                train_loss += loss.mean()
                rmse_metric.update(batch_yf, mx.nd.array(outputs))

                obj_loss += tot_error.asscalar()
                imb_err += imb_error.asscalar()

            if epoch % FLAGS.epoch_output_iter == 0:
                _, train_rmse_factual = rmse_metric.get()
                train_loss /= number_of_batches
                (_, valid_rmse_factual), _, _ = hybrid_test_net_with_cfr(net, valid_factual_loader, ctx,
                                                                         FLAGS,
                                                                         np.mean(valid['t']))

                log(logfile, '[Epoch %d/%d] Train-rmse-factual: %.3f | L2Loss: %.3f | learning-rate: '
                             '%.3E | ObjLoss: %.3f | ImbErr: %.3f | Valid-rmse-factual: %.3f' % (
                        epoch, epochs, train_rmse_factual, train_loss, trainer.learning_rate,
                        obj_loss, imb_err, valid_rmse_factual))

        ''' Plot first experiment learned TSNE visualization '''
        if train_experiment == 0:
            tsne_plot_pca(data=train['x'],
                          label=train['t'],
                          learned_representation=np.asarray([ind.asnumpy() for ind in first_exp_reps]),
                          outdir=outdir + FLAGS.architecture.lower())

        train_durations[train_experiment, :] = time.time() - train_start

        ''' Test model with valid data '''
        y_t0, y_t1, = hybrid_predict_treated_and_controlled_with_cfr(net,
                                                                     train_factual_loader,
                                                                     ctx)
        if FLAGS.normalize_input:
            y_t0, y_t1 = y_t0 * yf_std + yf_m, y_t1 * yf_std + yf_m
        train_score = train_evaluator.get_metrics(y_t1, y_t0)
        train_scores[train_experiment, :] = train_score

        y_t0, y_t1, = hybrid_predict_treated_and_controlled_with_cfr(net,
                                                                     valid_factual_loader,
                                                                     ctx)
        if FLAGS.normalize_input:
            y_t0, y_t1 = y_t0 * yf_std + yf_m, y_t1 * yf_std + yf_m
        valid_score = valid_evaluator.get_metrics(y_t1, y_t0)
        valid_scores[train_experiment, :] = valid_score

        log(logfile, '[Train Replication {}/{}]: train RMSE ITE: {:0.3f}, train ATE: {:0.3f}, train PEHE: {:0.3f},' \
                     ' valid RMSE ITE: {:0.3f}, valid ATE: {:0.3f}, valid PEHE: {:0.3f}'.format(train_experiment + 1,
                                                                                                train_experiments,
                                                                                                train_score[0],
                                                                                                train_score[1],
                                                                                                train_score[2],
                                                                                                valid_score[0],
                                                                                                valid_score[1],
                                                                                                valid_score[2]))

    ''' Save means and stds NDArray values for inference '''
    if FLAGS.normalize_input:
        mx.nd.save(outdir + FLAGS.architecture.lower() + '_means_stds_ihdp_' + str(train_experiments) + '_.nd',
                   {"means": mx.nd.array(means), "stds": mx.nd.array(stds)})

    ''' Export trained models '''
    # See mxnet.apache.org/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html
    net.export(outdir + FLAGS.architecture.lower() + "-ihdp-predictions-" + str(train_experiments))  # hybrid

    log(logfile, '\n{} architecture total scores:'.format(FLAGS.architecture.upper()))

    ''' Train and test scores '''
    means, stds = np.mean(train_scores, axis=0), sem(train_scores, axis=0, ddof=0)
    r_pehe_mean, r_pehe_std = np.mean(np.sqrt(train_scores[:, 2]), axis=0), sem(np.sqrt(train_scores[:, 2]), axis=0,
                                                                                ddof=0)
    train_total_scores_str = 'train RMSE ITE: {:.2f} ± {:.2f}, train ATE: {:.2f} ± {:.2f}, train PEHE: {:.2f} ± {:.2f}, ' \
                             'test root PEHE: {:.2f} ± {:.2f}' \
                             ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2], r_pehe_mean, r_pehe_std)

    means, stds = np.mean(valid_scores, axis=0), sem(valid_scores, axis=0, ddof=0)
    r_pehe_mean, r_pehe_std = np.mean(np.sqrt(valid_scores[:, 2]), axis=0), sem(np.sqrt(valid_scores[:, 2]), axis=0,
                                                                                ddof=0)
    valid_total_scores_str = 'valid RMSE ITE: {:.2f} ± {:.2f}, valid ATE: {:.2f} ± {:.2f}, valid PEHE: {:.2f} ± {:.2f}, ' \
                             'valid root PEHE: {:.2f} ± {:.2f}' \
                             ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2], r_pehe_mean, r_pehe_std)

    log(logfile, train_total_scores_str)
    log(logfile, valid_total_scores_str)

    mean_duration = float("{0:.2f}".format(np.mean(train_durations, axis=0)[0]))

    return {"ite": "{:.2f} ± {:.2f}".format(means[0], stds[0]),
            "ate": "{:.2f} ± {:.2f}".format(means[1], stds[1]),
            "pehe": "{:.2f} ± {:.2f}".format(means[2], stds[2]),
            "mean_duration": mean_duration}


def main():
    """ Main entry point for training a CFR net. """

    ''' Parse arguments '''
    global FLAGS
    FLAGS = get_cfr_args_parser().parse_args()

    FLAGS.architecture = "cfr"

    ''' Create outdir if inexistent '''
    outdir_path = pathlib.Path(FLAGS.outdir)
    if not outdir_path.is_dir():
        os.mkdir(FLAGS.outdir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = FLAGS.outdir + '/results_' + timestamp + '/'
    os.mkdir(outdir)

    try:
        mx_run(outdir)
    except Exception:
        with open(outdir + 'error.txt', 'w') as error_file:
            error_file.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == '__main__':
    main()
