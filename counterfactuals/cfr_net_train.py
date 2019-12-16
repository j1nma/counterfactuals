import datetime
import os
import pathlib
import random
import sys
import time
import traceback
import warnings

from mxnet import gluon, autograd
from scipy.stats import sem

from counterfactuals.cfr.cfr_net import cfr_net
from counterfactuals.cfr.net import CFRNet, WassersteinLoss
from counterfactuals.cfr.util import *
from counterfactuals.evaluation import Evaluator
from counterfactuals.utilities import log, load_data, validation_split, get_cfr_args_parser, \
    split_data_in_train_valid, hybrid_test_net_with_cfr, \
    hybrid_predict_treated_and_controlled_with_cfr

FLAGS = 0

NUM_ITERATIONS_PER_DECAY = 100


def train(CFR, sess, train_step, D, I_valid, D_test, logfile, i_exp):
    """ Trains a CFR model on supplied data """

    ''' Train/validation split '''
    n = D['x'].shape[0]
    I = range(n)
    I_train = list(set(I) - set(I_valid))
    n_train = len(I_train)

    ''' Compute treatment probability'''
    p_treated = np.mean(D['t'][I_train, :])

    ''' Set up loss feed_dicts'''
    dict_factual = {CFR.x: D['x'][I_train, :], CFR.t: D['t'][I_train, :], CFR.y_: D['yf'][I_train, :], \
                    CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, \
                    CFR.r_lambda: FLAGS.weight_decay, CFR.p_t: p_treated}

    if FLAGS.val_part > 0:
        dict_valid = {CFR.x: D['x'][I_valid, :], CFR.t: D['t'][I_valid, :], CFR.y_: D['yf'][I_valid, :], \
                      CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, \
                      CFR.r_lambda: FLAGS.weight_decay, CFR.p_t: p_treated}

    dict_cfactual = {CFR.x: D['x'][I_train, :], CFR.t: 1 - D['t'][I_train, :], CFR.y_: D['ycf'][I_train, :], \
                     CFR.do_in: 1.0, CFR.do_out: 1.0}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []

    ''' Compute losses '''
    losses = []
    obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], \
                                          feed_dict=dict_factual)

    cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

    valid_obj = np.nan
    valid_imb = np.nan
    valid_f_error = np.nan
    if FLAGS.val_part > 0:
        valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], \
                                                       feed_dict=dict_valid)

    losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])

    objnan = False

    reps = []
    reps_test = []

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):

        ''' Fetch sample '''
        I = random.sample(range(0, n_train), FLAGS.batch_size_per_unit)
        x_batch = D['x'][I_train, :][I, :]
        t_batch = D['t'][I_train, :][I]
        y_batch = D['yf'][I_train, :][I]

        ''' Do one step of gradient descent '''
        if not objnan:
            sess.run(train_step, feed_dict={CFR.x: x_batch, CFR.t: t_batch, \
                                            CFR.y_: y_batch, CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out, \
                                            CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.weight_decay,
                                            CFR.p_t: p_treated})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0 or i == FLAGS.iterations - 1:
            obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                                                  feed_dict=dict_factual)

            rep = sess.run(CFR.h_rep_norm, feed_dict={CFR.x: D['x'], CFR.do_in: 1.0})

            cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan
            valid_imb = np.nan
            valid_f_error = np.nan
            if FLAGS.val_part > 0:
                valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                                                               feed_dict=dict_valid)

            losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])
            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tImb: %.2g,\tVal: %.3f,\tValImb: %.2g,\tValObj: %.2f' \
                       % (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj)

            log(logfile, loss_str)

            if np.isnan(obj_loss):
                log(logfile, 'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True

        ''' Compute predictions every M iterations '''
        if (FLAGS.pred_output_iter > 0 and i % FLAGS.pred_output_iter == 0) or i == FLAGS.iterations - 1:

            y_pred_f = sess.run(CFR.output, feed_dict={CFR.x: D['x'], \
                                                       CFR.t: D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            y_pred_cf = sess.run(CFR.output, feed_dict={CFR.x: D['x'], \
                                                        CFR.t: 1 - D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf), axis=1))

            if D_test is not None:
                y_pred_f_test = sess.run(CFR.output, feed_dict={CFR.x: D_test['x'], \
                                                                CFR.t: D_test['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
                y_pred_cf_test = sess.run(CFR.output, feed_dict={CFR.x: D_test['x'], \
                                                                 CFR.t: 1 - D_test['t'], CFR.do_in: 1.0,
                                                                 CFR.do_out: 1.0})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test), axis=1))

            if FLAGS.save_rep and i_exp == 1:
                reps_i = sess.run([CFR.h_rep], feed_dict={CFR.x: D['x'], \
                                                          CFR.do_in: 1.0, CFR.do_out: 0.0})
                reps.append(reps_i)

                if D_test is not None:
                    reps_test_i = sess.run([CFR.h_rep], feed_dict={CFR.x: D_test['x'], \
                                                                   CFR.do_in: 1.0, CFR.do_out: 0.0})
                    reps_test.append(reps_test_i)

    return losses, preds_train, preds_test, reps, reps_test


def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = outdir + 'result'
    npzfile_test = outdir + 'result.test'
    repfile = outdir + 'reps'
    repfile_test = outdir + 'reps.test'
    outform = outdir + 'y_pred'
    outform_test = outdir + 'y_pred.test'
    lossform = outdir + 'loss'
    logfile = outdir + 'log.txt'
    f = open(logfile, 'w')
    f.close()
    data_train = FLAGS.data_dir + FLAGS.data_train
    data_train_test = FLAGS.data_dir + FLAGS.data_test

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    mx.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir + 'config.txt', FLAGS)

    log(logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha, FLAGS.weight_decay))

    ''' Load Data '''
    datapath = data_train
    datapath_test = data_train_test

    log(logfile, 'Training data: ' + datapath)
    log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath)
    D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    ''' Start Session '''
    sess = tf.Session()

    ''' Initialize input placeholders '''
    x = tf.placeholder("float", shape=[None, D['dim']], name='x')  # Features
    t = tf.placeholder("float", shape=[None, 1], name='t')  # Treatment
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    # mx
    mx_x = mx.sym.Variable(name='x', dtype="float", shape=(None, D['dim']))
    mx_t = mx.sym.Variable(name='t', dtype="float", shape=(None, 1))
    mx_y_ = mx.sym.Variable(name='y_', dtype="float", shape=(None, 1))

    ''' Parameter placeholders '''
    r_alpha = tf.placeholder("float", name='r_alpha')
    r_lambda = tf.placeholder("float", name='r_lambda')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')

    # mx
    mx_r_alpha = mx.sym.Variable(name='r_alpha', dtype="float")
    mx_r_lambda = mx.sym.Variable(name='r_lambda', dtype="float")
    mx_do_in = FLAGS.dropout_in
    mx_do_out = FLAGS.dropout_out
    mx_p = mx.sym.Variable(name='p_treated', dtype="float")

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_rep, FLAGS.dim_hyp]
    CFR = cfr_net(x, t, y_, p, FLAGS, r_alpha, r_lambda, do_in, do_out, dims,
                  mx_do_in, mx_do_out, mx_x, mx_t, mx_y_, mx_p)

    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, \
                                    NUM_ITERATIONS_PER_DECAY, FLAGS.learning_rate_factor, staircase=True)

    # TODO
    # opt = tf.train.AdamOptimizer(lr)
    opt = tf.train.RMSPropOptimizer(lr, FLAGS.rms_prop_decay)

    train_step = opt.minimize(CFR.tot_loss, global_step=global_step)

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []

    all_preds_test = []

    n_experiments = FLAGS.experiments

    ''' Run for all repeated experiments '''
    for i_exp in range(1, n_experiments + 1):

        log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))

        ''' Load Data)'''

        if i_exp == 1 or FLAGS.experiments > 1:
            D_exp = {}
            D_exp['x'] = D['x'][:, :, i_exp - 1]
            D_exp['t'] = D['t'][:, i_exp - 1:i_exp]
            D_exp['yf'] = D['yf'][:, i_exp - 1:i_exp]
            D_exp['ycf'] = D['ycf'][:, i_exp - 1:i_exp]

            D_exp_test = {}
            D_exp_test['x'] = D_test['x'][:, :, i_exp - 1]
            D_exp_test['t'] = D_test['t'][:, i_exp - 1:i_exp]
            D_exp_test['yf'] = D_test['yf'][:, i_exp - 1:i_exp]
            D_exp_test['ycf'] = D_test['ycf'][:, i_exp - 1:i_exp]

        ''' Split into training and validation sets '''
        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)
        # TODO: replace?
        # I_train, I_valid = train_test_split(np.arange(D_exp['x'].shape[0]), test_size=FLAGS.val_part,
        #                                     random_state=FLAGS.seed)

        ''' Run training loop '''
        losses, preds_train, preds_test, reps, reps_test = \
            train(CFR, sess, train_step, D_exp, I_valid, \
                  D_exp_test, logfile, i_exp)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_losses.append(losses)

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train, 1, 3), 0, 2)
        out_preds_test = np.swapaxes(np.swapaxes(all_preds_test, 1, 3), 0, 2)
        out_losses = np.swapaxes(np.swapaxes(all_losses, 0, 2), 0, 1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)
        if FLAGS.output_csv:
            np.savetxt('%s_%d.csv' % (outform, i_exp), preds_train[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (outform_test, i_exp), preds_test[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (lossform, i_exp), losses, delimiter=',')

        ''' Save results and predictions '''
        all_valid.append(I_valid)
        np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        np.savez(npzfile_test, pred=out_preds_test)

        ''' Save representations '''
        if FLAGS.save_rep and i_exp == 1:
            np.savez(repfile, rep=reps)

            np.savez(repfile_test, rep=reps_test)


def mx_run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    logfile = outdir + 'log.txt'
    f = open(logfile, 'w')
    f.close()

    # Hyperparameters
    epochs = int(FLAGS.iterations)
    learning_rate = float(FLAGS.learning_rate)
    wd = float(FLAGS.weight_decay)
    train_experiments = int(FLAGS.experiments)
    learning_rate_factor = float(FLAGS.learning_rate_factor)
    learning_rate_steps = int(FLAGS.learning_rate_steps)  # changes the learning rate for every n updates.

    # Logging # todo consistent commenting '''''
    logfile = outdir + 'log.txt'
    f = open(logfile, 'w')
    f.close()
    data_train = FLAGS.data_dir + FLAGS.data_train
    data_train_valid = FLAGS.data_dir + FLAGS.data_test

    # Set GPUs/CPUs
    num_gpus = mx.context.num_gpus()
    num_workers = int(FLAGS.num_workers)  # replace num_workers with the number of cores
    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    units = num_gpus if num_gpus > 0 else 1
    batch_size_per_unit = int(FLAGS.batch_size_per_unit)  # mini-batch size
    batch_size = batch_size_per_unit * max(units, 1)

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    mx.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir + 'config.txt', FLAGS)

    log(logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha, FLAGS.weight_decay))

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')

    # Load datasets
    train_dataset = load_data(data_train, normalize=FLAGS.normalize_input)

    log(logfile, 'Training data: ' + data_train)
    log(logfile, 'Valid data:     ' + data_train_valid)
    log(logfile, 'Loaded data with shape [%d,%d]' % (train_dataset['n'], train_dataset['dim']))

    # CFR Neural Network Architecture for ITE estimation
    net = CFRNet(FLAGS.dim_rep, FLAGS.dim_hyp, FLAGS.weight_init_scale, train_dataset['dim'], FLAGS.batch_norm)

    # Instantiate net
    net.initialize(ctx=ctx)
    net.hybridize()  # hybridize for better performance

    # Metric, Loss and Optimizer
    rmse_metric = mx.metric.RMSE()
    l2_loss = gluon.loss.L2Loss()
    wass_loss = WassersteinLoss(lam=FLAGS.wass_lambda,
                                its=FLAGS.wass_iterations,
                                square=True, backpropT=FLAGS.wass_bpg)
    scheduler = mx.lr_scheduler.FactorScheduler(step=learning_rate_steps, factor=learning_rate_factor,
                                                base_lr=learning_rate)
    optimizer = mx.optimizer.Adam(learning_rate=learning_rate, lr_scheduler=scheduler, wd=wd)
    trainer = gluon.Trainer(net.collect_params(), optimizer=optimizer)

    # Initialize train score results
    train_scores = np.zeros((train_experiments, 3))

    # Initialize train experiment durations
    train_durations = np.zeros((train_experiments, 1))

    # Initialize valid score results
    valid_scores = np.zeros((train_experiments, 3))

    # Train experiments means and stds
    means = np.array([])
    stds = np.array([])

    # Train
    for train_experiment in range(train_experiments):

        # Create training dataset
        x = train_dataset['x'][:, :, train_experiment]
        t = np.reshape(train_dataset['t'][:, train_experiment], (-1, 1))
        yf = train_dataset['yf'][:, train_experiment]
        ycf = train_dataset['ycf'][:, train_experiment]
        mu0 = train_dataset['mu0'][:, train_experiment]
        mu1 = train_dataset['mu1'][:, train_experiment]

        train, valid, valid_idx = split_data_in_train_valid(x, t, yf, ycf, mu0, mu1)

        # Train, Valid Evaluators, with labels not normalized
        train_evaluator = Evaluator(train['t'], train['yf'], train['ycf'], train['mu0'], train['mu1'])
        valid_evaluator = Evaluator(valid['t'], valid['yf'], valid['ycf'], valid['mu0'], valid['mu1'])

        # Normalize yf
        if FLAGS.normalize_input:
            yf_m, yf_std = np.mean(train['yf'], axis=0), np.std(train['yf'], axis=0)
            train['yf'] = (train['yf'] - yf_m) / yf_std
            valid['yf'] = (valid['yf'] - yf_m) / yf_std

            # Save mean and std
            means = np.append(means, yf_m)
            stds = np.append(stds, yf_std)

        # todo: what about paper:
        # "The results of the experiments on IHDP are presented in Table 1 (left).
        # We average over 1000 realizations of the outcomes with 63/27/10 train/validation/test splits."
        # Train dataset
        train_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(train['x']), mx.nd.array(train['t']),
                                                        mx.nd.array(train['yf']))

        # Valid dataset
        valid_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(valid['x']), mx.nd.array(valid['t']),
                                                        mx.nd.array(valid['yf']))

        # Train DataLoader
        train_factual_loader = gluon.data.DataLoader(train_factual_dataset, batch_size=batch_size, shuffle=True,
                                                     num_workers=num_workers)

        # Valid DataLoader
        valid_factual_loader = gluon.data.DataLoader(valid_factual_dataset, batch_size=batch_size, shuffle=False,
                                                     num_workers=num_workers)

        number_of_batches = len(train_factual_loader)

        # Compute treatment probability
        treatment_probability = np.mean(train['t'])

        train_start = time.time()

        # Train model
        for epoch in range(1, epochs + 1):  # start with epoch 1 for easier learning rate calculation

            train_loss = 0
            rmse_metric.reset()
            obj_loss = 0
            imb_err = 0

            for i, (x, t, batch_yf) in enumerate(train_factual_loader):
                # Get data and labels into slices and copy each slice into a context.
                x = x.as_in_context(ctx)
                t = t.as_in_context(ctx)
                batch_yf = batch_yf.as_in_context(ctx)

                # Get treatment and control indices. Make sure batch_size is enough to have at least one treated sample
                t1_idx = np.where(t == 1)[0]
                t0_idx = np.where(t == 0)[0]

                if t1_idx.shape[0] == 0:
                    log(logfile, 'Encountered no treatment samples at batch ' + str(i) + '.')

                # Compute sample reweighing
                if FLAGS.reweight_sample:
                    w_t = t / (2 * treatment_probability)
                    w_c = (1 - t) / (2 * 1 - treatment_probability)
                    sample_weight = w_t + w_c
                else:
                    sample_weight = 1.0

                # Initialize outputs
                outputs = np.zeros(batch_yf.shape)
                loss = np.zeros(batch_yf.shape)

                # Forward (Factual)
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
                        h_rep_norm = rep_o / np_safe_sqrt(mx.nd.sum(mx.nd.square(rep_o), axis=1, keepdims=True))
                    else:
                        h_rep_norm = 1.0 * rep_o

                    imb_dist = wass_loss(h_rep_norm[t1_idx], h_rep_norm[t0_idx])

                    imb_error = FLAGS.p_alpha * imb_dist

                    tot_error = risk

                    if FLAGS.p_alpha > 0:
                        tot_error = tot_error + imb_error

                # Backward
                tot_error.backward()

                # Optimize
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
                        obj_loss, imb_err, valid_rmse_factual,))

        train_durations[train_experiment, :] = time.time() - train_start

        # Test model with valid data
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

    if FLAGS.normalize_input:
        # Save means and stds NDArray values for inference
        mx.nd.save(outdir + FLAGS.architecture.lower() + '_means_stds_ihdp_' + str(train_experiments) + '_.nd',
                   {"means": mx.nd.array(means), "stds": mx.nd.array(stds)})

    # Export trained models. See mxnet.apache.org/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html
    net.export(outdir + FLAGS.architecture.lower() + "-ihdp-predictions-" + str(train_experiments))  # hybrid

    log(logfile, '\n{} architecture total scores:'.format(FLAGS.architecture.upper()))

    means, stds = np.mean(train_scores, axis=0), sem(train_scores, axis=0, ddof=0)
    train_total_scores_str = 'train RMSE ITE: {:.2f} ± {:.2f}, train ATE: {:.2f} ± {:.2f}, train PEHE: {:.2f} ± {:.2f}' \
                             ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2])

    means, stds = np.mean(valid_scores, axis=0), sem(valid_scores, axis=0, ddof=0)
    valid_total_scores_str = 'valid RMSE ITE: {:.2f} ± {:.2f}, valid ATE: {:.2f} ± {:.2f}, valid PEHE: {:.2f} ± {:.2f}' \
        .format(means[0], stds[0], means[1], stds[1], means[2], stds[2])

    log(logfile, train_total_scores_str)
    log(logfile, valid_total_scores_str)

    mean_duration = float("{0:.2f}".format(np.mean(train_durations, axis=0)[0]))

    return {"ite": "{:.2f} ± {:.2f}".format(means[0], stds[0]),
            "ate": "{:.2f} ± {:.2f}".format(means[1], stds[1]),
            "pehe": "{:.2f} ± {:.2f}".format(means[2], stds[2]),
            "mean_duration": mean_duration}


# todo file not found friendly error
def mx_run_out_of_sample_test(outdir):
    # Logging
    logfile = outdir + 'log.txt'
    f = open(logfile, 'w')
    f.close()

    # TODO: dont mix things: imported means and stds have nothing to do with the 75 test "out of sample" data"
    # Set GPUs/CPUs
    num_gpus = mx.context.num_gpus()
    num_workers = int(FLAGS.num_workers)  # replace num_workers with the number of cores
    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    units = num_gpus if num_gpus > 0 else 1
    batch_size_per_unit = int(FLAGS.batch_size_per_unit)  # mini-batch size
    batch_size = batch_size_per_unit * max(units, 1)

    # Load test dataset
    test_dataset = load_data(FLAGS.data_dir + FLAGS.data_test, normalize=FLAGS.normalize_input)

    # Import CFRNet
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net_prefix = FLAGS.results_dir + "/" + FLAGS.architecture.lower() + "-ihdp-predictions-" + str(
            FLAGS.experiments) + "-"
        net = gluon.nn.SymbolBlock.imports(net_prefix + "symbol.json",
                                           ['data0', 'data1', 'data2'],
                                           net_prefix + "0000.params",
                                           ctx=ctx)

    # Calculate number of test experiments
    test_experiments = test_dataset['x'].shape[2]

    # Initialize test score results
    test_scores = np.zeros((test_experiments, 3))

    # Test model
    for test_experiment in range(test_experiments):

        # Create testing dataset
        x = test_dataset['x'][:, :, test_experiment]
        t = np.reshape(test_dataset['t'][:, test_experiment], (-1, 1))
        yf = test_dataset['yf'][:, test_experiment]
        ycf = test_dataset['ycf'][:, test_experiment]
        mu0 = test_dataset['mu0'][:, test_experiment]
        mu1 = test_dataset['mu1'][:, test_experiment]

        # Test Evaluator, with labels not normalized
        test_evaluator = Evaluator(t, yf, ycf, mu0, mu1)

        # Retrieve training mean and std
        # train_yf_m, train_yf_std = train_means[test_experiment].asnumpy(), train_stds[test_experiment].asnumpy()

        # Normalize yf
        if FLAGS.normalize_input:
            test_yf_m, test_yf_std = np.mean(yf, axis=0), np.std(yf, axis=0)
            yf = (yf - test_yf_m) / test_yf_std

        # Test dataset
        test_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(x), mx.nd.array(t), mx.nd.array(yf))

        # Test DataLoader
        test_rmse_ite_loader = gluon.data.DataLoader(test_factual_dataset, batch_size=batch_size, shuffle=False,
                                                     num_workers=num_workers)

        # Test model with test data
        y_t0, y_t1 = hybrid_predict_treated_and_controlled_with_cfr(net, test_rmse_ite_loader, ctx)
        if FLAGS.normalize_input:
            y_t0, y_t1 = y_t0 * test_yf_std + test_yf_m, y_t1 * test_yf_std + test_yf_m
        test_score = test_evaluator.get_metrics(y_t1, y_t0)
        test_scores[test_experiment, :] = test_score

        log(logfile,
            '[Test Replication {}/{}]: RMSE ITE: {:0.3f}, ATE: {:0.3f}, PEHE: {:0.3f}'.format(test_experiment + 1,
                                                                                              test_experiments,
                                                                                              test_score[0],
                                                                                              test_score[1],
                                                                                              test_score[2]))

    means, stds = np.mean(test_scores, axis=0), sem(test_scores, axis=0, ddof=0)
    log(logfile, 'test RMSE ITE: {:.3f} ± {:.3f}, test ATE: {:.3f} ± {:.3f}, test PEHE: {:.3f} ± {:.3f}' \
                 ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))


def main(argv=None):
    # Parse arguments
    global FLAGS
    FLAGS = get_cfr_args_parser().parse_args()

    FLAGS.architecture = "cfr"

    # Create outdir if inexistent
    outdir_path = pathlib.Path(FLAGS.outdir)
    if not outdir_path.is_dir():
        os.mkdir(FLAGS.outdir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = FLAGS.outdir + '/results_' + timestamp + '/'
    os.mkdir(outdir)

    try:  # todo fix this please, find good practice, if test, dont create above dir
        # run(outdir)
        # mx_run(outdir)
        mx_run_out_of_sample_test(outdir)
    except Exception as e:
        with open(outdir + 'error.txt', 'w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == '__main__':
    main()
