import datetime
import os
import pathlib
import sys
import traceback
import warnings

from mxnet import gluon
from scipy.stats import sem

from counterfactuals.cfr.util import *
from counterfactuals.evaluation import Evaluator
from counterfactuals.utilities import log, load_data, get_cfr_args_parser, \
    hybrid_predict_treated_and_controlled_with_cfr

FLAGS = 0


def mx_run_out_of_sample_test(outdir):
    """ Runs a set of test experiments and stores result in a directory. """

    ''' Logging. '''
    logfile = outdir + 'log.txt'
    f = open(logfile, 'w')
    f.close()

    ''' Set GPUs/CPUs '''
    num_gpus = mx.context.num_gpus()
    num_workers = int(FLAGS.num_workers)  # replace num_workers with the number of cores
    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    units = num_gpus if num_gpus > 0 else 1
    batch_size_per_unit = int(FLAGS.batch_size_per_unit)  # mini-batch size
    batch_size = batch_size_per_unit * max(units, 1)

    ''' Load test dataset '''
    test_dataset = load_data(FLAGS.data_dir + FLAGS.data_test, normalize=FLAGS.normalize_input)

    ''' Import CFRNet '''
    try:
        warnings.simplefilter("ignore")
        net_prefix = FLAGS.results_dir + "/" + FLAGS.architecture.lower() + "-ihdp-predictions-" + str(
            FLAGS.experiments) + "-"
        net = gluon.nn.SymbolBlock.imports(net_prefix + "symbol.json",
                                           ['data0', 'data1', 'data2'],
                                           net_prefix + "0000.params",
                                           ctx=ctx)
    except Exception as e:
        with open(outdir + 'error.txt', 'w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        print(e.args[0].split('Stack trace')[0])
        print("More details at:\t" + str(outdir + 'error.txt'))
        sys.exit(-1)

    ''' Calculate number of test experiments '''
    test_experiments = test_dataset['x'].shape[2]

    ''' Initialize test score results '''
    test_scores = np.zeros((test_experiments, 3))

    ''' Test model '''
    for test_experiment in range(test_experiments):

        ''' Create testing dataset '''
        x = test_dataset['x'][:, :, test_experiment]
        t = np.reshape(test_dataset['t'][:, test_experiment], (-1, 1))
        yf = test_dataset['yf'][:, test_experiment]
        ycf = test_dataset['ycf'][:, test_experiment]
        mu0 = test_dataset['mu0'][:, test_experiment]
        mu1 = test_dataset['mu1'][:, test_experiment]

        ''' Test Evaluator, with labels not normalized '''
        test_evaluator = Evaluator(t, yf, ycf, mu0, mu1)

        ''' Normalize yf '''
        if FLAGS.normalize_input:
            test_yf_m, test_yf_std = np.mean(yf, axis=0), np.std(yf, axis=0)
            yf = (yf - test_yf_m) / test_yf_std

        ''' Test dataset '''
        test_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(x), mx.nd.array(t), mx.nd.array(yf))

        ''' Test DataLoader '''
        test_rmse_ite_loader = gluon.data.DataLoader(test_factual_dataset, batch_size=batch_size, shuffle=False,
                                                     num_workers=num_workers)

        ''' Test model with test data '''
        y_t0, y_t1 = hybrid_predict_treated_and_controlled_with_cfr(net, test_rmse_ite_loader, ctx)
        if FLAGS.normalize_input:
            y_t0, y_t1 = y_t0 * test_yf_std + test_yf_m, y_t1 * test_yf_std + test_yf_m
        test_score = test_evaluator.get_metrics(y_t1, y_t0)
        test_scores[test_experiment, :] = test_score

        log(logfile,
            '[Test Replication {}/{}]:\tRMSE ITE: {:0.3f},\t\t ATE: {:0.3f},\t\t PEHE: {:0.3f}'.format(test_experiment + 1,
                                                                                              test_experiments,
                                                                                              test_score[0],
                                                                                              test_score[1],
                                                                                              test_score[2]))

    means, stds = np.mean(test_scores, axis=0), sem(test_scores, axis=0, ddof=0)
    log(logfile, 'test RMSE ITE: {:.3f} ± {:.3f}, test ATE: {:.3f} ± {:.3f}, test PEHE: {:.3f} ± {:.3f}' \
                 ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))


def main(argv=None):
    """ Main entry point for testing a CFR net. """

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
        mx_run_out_of_sample_test(outdir)
    except Exception as e:
        with open(outdir + 'error.txt', 'w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == '__main__':
    main()
