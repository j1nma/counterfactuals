import datetime
import os
import pathlib
import random
import sys
import traceback

from counterfactuals.cfr.cfr_net import cfr_net, mx_cfr_net
from counterfactuals.cfr.util import *
from counterfactuals.utilities import log, load_data, validation_split, get_cfr_args_parser

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
                    CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}

    if FLAGS.val_part > 0:
        dict_valid = {CFR.x: D['x'][I_valid, :], CFR.t: D['t'][I_valid, :], CFR.y_: D['yf'][I_valid, :], \
                      CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, \
                      CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}

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
        I = random.sample(range(0, n_train), FLAGS.batch_size)
        x_batch = D['x'][I_train, :][I, :]
        t_batch = D['t'][I_train, :][I]
        y_batch = D['yf'][I_train, :][I]

        ''' Do one step of gradient descent '''
        if not objnan:
            sess.run(train_step, feed_dict={CFR.x: x_batch, CFR.t: t_batch, \
                                            CFR.y_: y_batch, CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out, \
                                            CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.p_lambda,
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

    log(logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha, FLAGS.p_lambda))

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
    npzfile = outdir + 'result'
    npzfile_test = outdir + 'result.test'
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

    log(logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha, FLAGS.p_lambda))

    ''' Load Data '''
    datapath = data_train
    datapath_test = data_train_test

    log(logfile, 'Training data: ' + datapath)
    log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath)
    D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    ''' Initialize input placeholders '''
    mx_x = mx.sym.Variable(name='x', dtype="float", shape=(None, D['dim']))
    mx_t = mx.sym.Variable(name='t', dtype="float", shape=(None, 1))
    mx_y_ = mx.sym.Variable(name='y_', dtype="float", shape=(None, 1))

    ''' Parameter placeholders '''
    mx_r_alpha = mx.sym.Variable(name='r_alpha', dtype="float")
    mx_r_lambda = mx.sym.Variable(name='r_lambda', dtype="float")
    mx_do_in = FLAGS.dropout_in
    mx_do_out = FLAGS.dropout_out
    mx_p = mx.sym.Variable(name='p_treated', dtype="float")

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_rep, FLAGS.dim_hyp]
    CFR = mx_cfr_net(FLAGS, mx_r_alpha, mx_r_lambda, mx_do_in, mx_do_out, dims, mx_do_in, mx_do_out, mx_x, mx_t, mx_y_,
                     mx_p)

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


        ''' Collect all reps '''


        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train, 1, 3), 0, 2)
        out_preds_test = np.swapaxes(np.swapaxes(all_preds_test, 1, 3), 0, 2)
        out_losses = np.swapaxes(np.swapaxes(all_losses, 0, 2), 0, 1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)

        ''' Save results and predictions '''
        all_valid.append(I_valid)
        np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        np.savez(npzfile_test, pred=out_preds_test)


def main(argv=None):
    # Parse arguments
    global FLAGS
    FLAGS = get_cfr_args_parser().parse_args()

    # Create outdir if inexistent
    outdir_path = pathlib.Path(FLAGS.outdir)
    if not outdir_path.is_dir():
        os.mkdir(FLAGS.outdir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = FLAGS.outdir + '/results_' + timestamp + '/'
    os.mkdir(outdir)

    try:
        # run(outdir)
        mx_run(outdir)
    except Exception as e:
        with open(outdir + 'error.txt', 'w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == '__main__':
    tf.app.run()
