# Standard feed-forward neural network
# Trained with 4 hidden layers, to predict the factual outcome based on X and t, without a penalty for imbalance.
#
# Referred to as NN-4 from Johansson et al. paper:
# "Learning Representations for Counterfactual Inference"
# arXiv:1605.03661v3 [stat.ML] 6 Jun 2018

# TODO are start end time duration spots OK?
import time
import warnings

import mxnet as mx
import numpy as np
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
from scipy.stats import sem

from counterfactuals.evaluation import Evaluator
from counterfactuals.utilities import load_data, split_data_in_train_valid_test, predict_treated_and_controlled, \
    predict_treated_and_controlled_vb, test_net_vb, log


# todo cutting condition on difference from last x epochs?

# todo citation code


def ff4_relu_architecture(hidden_size):
    net = nn.HybridSequential()
    net.add(nn.Dense(hidden_size, activation='relu'),
            nn.Dense(hidden_size, activation='relu'),
            nn.Dense(hidden_size, activation='relu'),
            nn.Dense(hidden_size, activation='relu'),
            nn.Dense(1)),
    return net


def sample_epsilons(param_shapes, ctx):
    epsilons = [nd.random_normal(shape=shape, loc=0., scale=1.0, ctx=ctx) for shape in param_shapes]
    return epsilons


def sample_exponential(param_shapes, ctx):
    exponentials = [nd.random_exponential(shape=shape, lam=1.0, ctx=ctx) for shape in param_shapes]
    return exponentials


def softplus(x):
    return nd.log(1. + nd.exp(x))


def transform_rhos(rhos):
    return [softplus(rho) for rho in rhos]


def transform_gaussian_samples(mus, sigmas, epsilons):
    samples = []
    for j in range(len(mus)):
        samples.append(mus[j] + sigmas[j] * epsilons[j])
    return samples


def transform_exponential_samples(lambdas, exponentials):
    samples = []
    for j in range(len(lambdas)):
        samples.append(exponentials[j] * (1 / (lambdas[j])))
    return samples


def generate_weight_sample(layer_param_shapes, mus, rhos, ctx):
    ''' sample epsilons from standard normal '''
    epsilons = sample_epsilons(layer_param_shapes, ctx)

    ''' compute softplus for variance '''
    sigmas = transform_rhos(rhos)

    ''' obtain a sample from q(w|theta) by transforming the epsilons '''
    layer_params = transform_gaussian_samples(mus, sigmas, epsilons)

    return layer_params, sigmas


def generate_weight_sample_exp(layer_param_shapes, raw_lambdas, ctx):
    ''' sample exponential rv. with rate 1 '''
    exponentials = sample_exponential(layer_param_shapes, ctx)

    ''' obtain a sample from q(w|theta) from exponential distribution '''
    layer_params = transform_exponential_samples(raw_lambdas, exponentials)

    return layer_params


# Bayes by Backpropagation Loss
# from https://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop-gluon.html
class BBBLoss(gluon.loss.Loss):
    def __init__(self, ctx, log_prior, sigma_p1=1.0, sigma_p2=0.1, pi=0.5, lambda_p=25.0, weight=None, batch_axis=0,
                 **kwargs):
        super(BBBLoss, self).__init__(weight, batch_axis, **kwargs)
        self.log_prior = log_prior
        self.sigma_p1 = sigma_p1
        self.sigma_p2 = sigma_p2
        self.pi = pi
        self.lambda_p = lambda_p
        self.ctx = ctx

    def log_gaussian(self, x, mu, sigma):
        """ https://bit.ly/2QqJbzL """
        return mx.nd.log(self.nd_gaussian(x, mu, sigma))

    def log_exponential(self, x, lambda_p):
        # return mx.nd.log(lambda_p * nd.exp(lambda_p * nd.negative(x)))
        # return mx.nd.log(lambda_p) - lambda_p * x
        if np.isinf(lambda_p.asnumpy()).any():
            return mx.nd.log(lambda_p + 1) - lambda_p * x
        else:
            return mx.nd.log(lambda_p) - lambda_p * x

    def gaussian_prior(self, x):
        sigma_p = nd.array([1.0], ctx=self.ctx)
        # sigma_p = nd.array([self.sigma_p1], ctx=self.ctx)
        return nd.sum(self.log_gaussian(x, 0., sigma_p))

    def exponential_prior(self, x):
        sigma_p = nd.array([self.lambda_p], ctx=self.ctx)
        # sigma_p = nd.array([self.sigma_p1], ctx=self.ctx)
        return nd.sum(self.log_exponential(x, sigma_p))

    def nd_gaussian(self, x, mu, sigma):  # TODO probably should be normalized???
        '''  nd.sqrt instead of np.sqrt '''
        scaling = 1.0 / nd.sqrt(2.0 * np.pi * (sigma ** 2))
        bell = nd.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))
        return scaling * bell + 1e-6

    def gaussian(self, x, mu, sigma):
        scaling = 1.0 / np.sqrt(2.0 * np.pi * (sigma ** 2))
        bell = nd.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))

        return scaling * bell + 1e-6

    def scale_mixture_prior(self, x):
        """ log_prior_prob """
        sigma_p1 = nd.array([self.sigma_p1], ctx=self.ctx)
        sigma_p2 = nd.array([self.sigma_p2], ctx=self.ctx)
        pi = self.pi

        first_gaussian = pi * self.nd_gaussian(x, 0., sigma_p1)
        second_gaussian = (1 - pi) * self.nd_gaussian(x, 0., sigma_p2)

        return mx.nd.log(first_gaussian + second_gaussian)

    def neg_log_likelihood(self, y_obs, y_pred, sigma=0.5):
        """ Loss for regression tasks """
        ''' from http://krasserm.github.io/2019/03/14/bayesian-neural-networks/ '''

        ''' The network can now be trained with a Gaussian negative log likelihood
        function (neg_log_likelihood) as loss function assuming a fixed standard deviation (noise).
        This corresponds to the likelihood cost, the last term in equation 3. '''

        # return mx.nd.sum(-1 * mx.nd.log(y_pred * nd.exp(y_pred * nd.negative(y_obs))))
        return mx.nd.sum(-1 * mx.nd.log(self.gaussian(y_obs, y_pred, sigma)))

    # def hybrid_forward(self, F, output, label, params, mus, sigmas, num_batches, sample_weight=None):
    def hybrid_forward(self, F, output, label, params, lambdas, sigmas, num_batches, sample_weight=None):
        prior = None
        if self.log_prior == "gaussian":
            prior = self.gaussian_prior
        elif self.log_prior == "scale_mixture":
            prior = self.scale_mixture_prior
        elif self.log_prior == "exponential":
            prior = self.exponential_prior

        # Calculate prior
        log_prior_sum = sum([nd.sum(prior(mx.nd.array(param))) for param in params])
        # log_prior_sum = sum([nd.nansum(prior(mx.nd.array(param))) for param in params])

        # Calculate variational posterior
        # log_var_posterior_sum = sum(
        #     [nd.sum(self.log_gaussian(mx.nd.array(params[i]), mx.nd.array(mus[i]), mx.nd.array(sigmas[i]))) for i in
        #      range(len(params))])
        log_var_posterior_sum = sum(
            [nd.nansum(self.log_exponential(mx.nd.array(params[i]), mx.nd.array(lambdas[i]))) for i in
             range(len(params))])

        # return (1.0 / num_batches) * (log_var_posterior_sum - log_prior_sum) + self.neg_log_likelihood(label, output)
        kl_loss = (1.0 / num_batches) * (log_var_posterior_sum - log_prior_sum)
        # return kl_loss + self.neg_log_likelihood(label, output)  # for gaussian
        return kl_loss  # for expo


def evaluate_RMSE(data_iterator, net, layer_params, ctx):
    metric = mx.metric.RMSE()
    metric.reset()
    for i, (data, label) in enumerate(data_iterator):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, even_split=False)
        label = gluon.utils.split_and_load(label, ctx_list=ctx, even_split=False)

        for l_param, param in zip(layer_params, net.collect_params().values()):
            param._data[0] = l_param

        outputs = [net(x.reshape((-1, 26))) for x in data]

        metric.update(label, outputs)

    return metric.get()


def run(args, outdir):
    """ Run training for NN4 architecture with Variational Bayes. """

    ''' Hyperparameters '''
    epochs = int(args.iterations)
    learning_rate = float(args.learning_rate)
    wd = float(args.weight_decay)
    hidden_size = int(args.hidden_size)
    train_experiments = int(args.experiments)
    learning_rate_factor = float(args.learning_rate_factor)
    learning_rate_steps = int(args.learning_rate_steps)  # changes the learning rate for every n updates.
    epoch_output_iter = int(args.epoch_output_iter)

    ''' Logging '''
    logfile = outdir + 'log.txt'
    f = open(logfile, 'w')
    f.close()

    config = {  # TODO may need adjustments
        # "sigma_p1": 1.5,
        "sigma_p1": 1.75,  # og
        # "sigma_p2": 0.25,
        # "sigma_p2": 0.5, # og
        "sigma_p2": 0.5,
        "pi": 0.5,
        "lambda_p": 24.5
    }

    ''' Set GPUs/CPUs '''
    num_gpus = mx.context.num_gpus()
    num_workers = int(args.num_workers)  # replace num_workers with the number of cores
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]  # todo change as cfr_net_train
    batch_size_per_unit = int(args.batch_size_per_unit)  # mini-batch size
    batch_size = batch_size_per_unit * max(num_gpus, 1)

    ''' Set seeds '''
    for c in ctx:
        mx.random.seed(int(args.seed), c)
    np.random.seed(int(args.seed))

    ''' Feed Forward Neural Network Model (4 hidden layers) '''
    net = ff4_relu_architecture(hidden_size)

    ''' Load datasets '''
    # train_dataset = load_data('../' + args.data_dir + args.data_train) # PyCharm run
    train_dataset = load_data(args.data_dir + args.data_train) # Terminal run

    log(logfile, 'Training data: ' + args.data_dir + args.data_train)
    log(logfile, 'Valid data:     ' + args.data_dir + args.data_test)
    log(logfile, 'Loaded data with shape [%d,%d]' % (train_dataset['n'], train_dataset['dim']))

    # ''' Feature correlation '''
    # import pandas as pd
    # df = pd.DataFrame.from_records(train_dataset['x'][:, :, 20])
    # df.insert(25, "t", train_dataset['t'][:, 20])
    # corr = df.corr()
    # import seaborn as sns
    # sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, fmt='.1f')

    ''' Instantiate net '''
    ''' Param. init. '''
    net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    net.hybridize()

    ''' Forward-propagate a single data set entry once to set up all network 
    parameters (weights and biases) with the desired initializer specified above. '''
    x = train_dataset['x'][:, :, 0]
    t = np.reshape(train_dataset['t'][:, 0], (-1, 1))
    yf = train_dataset['yf'][:, 0]
    yf_m, yf_std = np.mean(yf, axis=0), np.std(yf, axis=0)
    yf = (yf - yf_m) / yf_std
    factual_features = np.hstack((x, t))
    zero_train_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(factual_features), mx.nd.array(yf))
    zero_train_factual_loader = gluon.data.DataLoader(zero_train_factual_dataset, batch_size=batch_size, shuffle=True,
                                                      num_workers=num_workers)
    for i, (batch_f_features, batch_yf) in enumerate(zero_train_factual_loader):
        batch_f_features = gluon.utils.split_and_load(batch_f_features, ctx_list=ctx, even_split=False)
        [net(x) for x in batch_f_features]
        break

    weight_scale = .1
    rho_offset = -3
    lambda_init = 25

    ''' Initialize variational parameters; mean and variance for each weight '''
    mus = []
    rhos = []
    lambdas = []

    shapes = list(map(lambda x: x.shape, net.collect_params().values()))

    for shape in shapes:
        # mu = gluon.Parameter('mu', shape=shape, init=mx.init.Normal(weight_scale))
        # rho = gluon.Parameter('rho', shape=shape, init=mx.init.Constant(rho_offset))
        lmb = gluon.Parameter('lmb', shape=shape, init=mx.init.Constant(lambda_init))
        # mu.initialize(ctx=ctx)
        # rho.initialize(ctx=ctx)
        lmb.initialize(ctx=ctx)
        # mus.append(mu)
        # rhos.append(rho)
        lambdas.append(lmb)
    # variational_params = mus + rhos
    variational_params = lambdas

    # raw_mus = list(map(lambda x: x.data(ctx[0]), mus))
    # raw_rhos = list(map(lambda x: x.data(ctx[0]), rhos))
    raw_lambdas = list(map(lambda x: x.data(ctx[0]), lambdas))

    ''' Metric, Loss and Optimizer '''
    rmse_metric = mx.metric.RMSE()
    l2_loss = gluon.loss.L2Loss()
    bbb_loss = BBBLoss(ctx[0], log_prior="exponential", sigma_p1=config['sigma_p1'], sigma_p2=config['sigma_p2'],
                       pi=config['pi'], lambda_p=config['lambda_p'])
    # bbb_loss = BBBLoss(ctx[0], log_prior="scale_mixture", sigma_p1=config['sigma_p1'], sigma_p2=config['sigma_p2'],
    #                    pi=config['pi'])
    scheduler = mx.lr_scheduler.FactorScheduler(step=learning_rate_steps, factor=learning_rate_factor,
                                                base_lr=learning_rate)
    # optimizer = mx.optimizer.Adam(learning_rate=learning_rate, lr_scheduler=scheduler)
    optimizer = mx.optimizer.RMSProp(learning_rate=learning_rate, lr_scheduler=scheduler, wd=wd)
    # optimizer = mx.optimizer.Adam(learning_rate=learning_rate)
    trainer = gluon.Trainer(variational_params, optimizer=optimizer)

    ''' Initialize train score results '''
    train_scores = np.zeros((train_experiments, 3))

    ''' Initialize train experiment durations '''
    train_durations = np.zeros((train_experiments, 1))

    ''' Initialize test score results '''
    test_scores = np.zeros((train_experiments, 3))

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

        train, valid, test, _ = split_data_in_train_valid_test(x, t, yf, ycf, mu0, mu1)

        ''' With-in sample '''
        train_evaluator = Evaluator(np.concatenate([train['t'], valid['t']]),
                                    np.concatenate([train['yf'], valid['yf']]),
                                    y_cf=np.concatenate([train['ycf'], valid['ycf']], axis=0),
                                    mu0=np.concatenate([train['mu0'], valid['mu0']], axis=0),
                                    mu1=np.concatenate([train['mu1'], valid['mu1']], axis=0))
        test_evaluator = Evaluator(test['t'], test['yf'], test['ycf'], test['mu0'], test['mu1'])

        ''' Normalize yf '''  # TODO check for normalize input?
        yf_m, yf_std = np.mean(train['yf'], axis=0), np.std(train['yf'], axis=0)
        train['yf'] = (train['yf'] - yf_m) / yf_std
        valid['yf'] = (valid['yf'] - yf_m) / yf_std
        test['yf'] = (test['yf'] - yf_m) / yf_std

        ''' Save mean and std '''
        means = np.append(means, yf_m)
        stds = np.append(stds, yf_std)

        ''' Train dataset '''
        factual_features = np.hstack((train['x'], train['t']))
        train_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(factual_features), mx.nd.array(train['yf']))

        ''' With-in sample '''
        train_rmse_ite_dataset = gluon.data.ArrayDataset(mx.nd.array(np.concatenate([train['x'], valid['x']])))

        ''' Valid dataset '''
        valid_factual_features = np.hstack((valid['x'], valid['t']))
        valid_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(valid_factual_features), mx.nd.array(valid['yf']))

        ''' Test dataset '''
        test_rmse_ite_dataset = gluon.data.ArrayDataset(mx.nd.array(test['x']))

        ''' Train DataLoader '''
        train_factual_loader = gluon.data.DataLoader(train_factual_dataset, batch_size=batch_size, shuffle=True,
                                                     num_workers=num_workers)
        train_rmse_ite_loader = gluon.data.DataLoader(train_rmse_ite_dataset, batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers)

        ''' Valid DataLoader '''
        valid_factual_loader = gluon.data.DataLoader(valid_factual_dataset, batch_size=batch_size, shuffle=False,
                                                     num_workers=num_workers)

        ''' Test DataLoader '''
        test_rmse_ite_loader = gluon.data.DataLoader(test_rmse_ite_dataset, batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=num_workers)

        num_batch = len(train_factual_loader)

        train_start = time.time()

        train_acc = []
        test_acc = []

        ''' Train model '''
        for epoch in range(1, epochs + 1):  # start with epoch 1 for easier learning rate calculation

            train_loss = 0
            rmse_metric.reset()

            for i, (batch_f_features, batch_yf) in enumerate(train_factual_loader):
                ''' Get data and labels into slices and copy each slice into a context.'''
                batch_f_features = batch_f_features.as_in_context(ctx[0]).reshape((-1, 26))
                batch_yf = batch_yf.as_in_context(ctx[0]).reshape((len(batch_yf), -1))

                ''' Forward '''
                with autograd.record():
                    ''' Generate sample '''
                    # layer_params, sigmas = generate_weight_sample(shapes, raw_mus, raw_rhos, ctx[0])
                    layer_params = generate_weight_sample_exp(shapes, raw_lambdas, ctx[0])

                    ''' Overwrite network parameters with sampled parameters '''
                    for sample, param in zip(layer_params, net.collect_params().values()):
                        param._data[0] = sample

                    ''' Forward-propagate the batch '''
                    outputs = net(batch_f_features)

                    # if epoch == epochs:
                    #     ''' Factual outcomes and batch_yf histograms '''
                    #     import pandas as pd
                    #     df = pd.DataFrame({'layer_params': layer_params[6][0].asnumpy().flatten()}, columns=['layer_params'])
                    #     df = pd.DataFrame(
                    #         {'outputs': outputs.asnumpy().flatten(), 'batch_yf': batch_yf.asnumpy().flatten()},
                    #         columns=['outputs', 'batch_yf'])
                    #     df.plot(kind='hist', alpha=0.5)
                    #     df.plot.kde()

                    ''' Calculate the loss '''
                    l2_loss_value = l2_loss(outputs, batch_yf)
                    # bbb_loss_value = bbb_loss(outputs, batch_yf, layer_params, raw_mus, sigmas, num_batch)
                    bbb_loss_value = bbb_loss(outputs, batch_yf, layer_params, raw_lambdas, [], num_batch)
                    loss = bbb_loss_value + l2_loss_value
                    # loss = bbb_loss_value
                    # loss = l2_loss_value

                    ''' Backpropagate for gradient calculation '''
                    loss.backward()

                ''' Optimize '''
                trainer.step(batch_size)

                train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

                rmse_metric.update(batch_yf, outputs)

            if epoch % epoch_output_iter == 0 or epoch == 1:
                _, train_rmse_factual = rmse_metric.get()
                train_loss /= num_batch
                _, valid_rmse_factual = test_net_vb(net, valid_factual_loader, layer_params, ctx)

                # _, train_RMSE = evaluate_RMSE(train_factual_loader, net, raw_mus, ctx)
                # _, test_RMSE = evaluate_RMSE(valid_factual_loader, net, raw_mus, ctx)
                # train_acc.append(np.asscalar(train_RMSE))
                # test_acc.append(np.asscalar(test_RMSE))
                # print("Epoch %s. Train-RMSE %s, Test-RMSE %s" %
                #       (epoch, train_RMSE, test_RMSE))

                log(logfile,
                    'l2-loss: %.3f, bbb-loss: %.3f' % (l2_loss_value[0].asscalar(), bbb_loss_value[0].asscalar()))

                log(logfile,
                    '[Epoch %d/%d] Train-rmse-factual: %.3f, loss: %.3f | Valid-rmse-factual: %.3f | learning-rate: '
                    '%.3E' % (
                        epoch, epochs, train_rmse_factual, train_loss, valid_rmse_factual, trainer.learning_rate))

        train_durations[train_experiment, :] = time.time() - train_start

        ''' Test model '''
        # y_t0, y_t1 = predict_treated_and_controlled_vb(net, train_rmse_ite_loader, raw_mus, ctx)
        y_t0, y_t1 = predict_treated_and_controlled_vb(net, train_rmse_ite_loader, layer_params, ctx)
        y_t0, y_t1 = y_t0 * yf_std + yf_m, y_t1 * yf_std + yf_m
        train_score = train_evaluator.get_metrics(y_t1, y_t0)
        train_scores[train_experiment, :] = train_score

        # y_t0, y_t1 = predict_treated_and_controlled_vb(net, test_rmse_ite_loader, raw_mus, ctx)
        y_t0, y_t1 = predict_treated_and_controlled_vb(net, test_rmse_ite_loader, layer_params, ctx)
        y_t0, y_t1 = y_t0 * yf_std + yf_m, y_t1 * yf_std + yf_m
        test_score = test_evaluator.get_metrics(y_t1, y_t0)
        test_scores[train_experiment, :] = test_score

        log(logfile, '[Train Replication {}/{}]: train RMSE ITE: {:0.3f}, train ATE: {:0.3f}, train PEHE: {:0.3f},' \
                     ' test RMSE ITE: {:0.3f}, test ATE: {:0.3f}, test PEHE: {:0.3f}'.format(train_experiment + 1,
                                                                                             train_experiments,
                                                                                             train_score[0],
                                                                                             train_score[1],
                                                                                             train_score[2],
                                                                                             test_score[0],
                                                                                             test_score[1],
                                                                                             test_score[2]))
        # plt.plot(train_acc)
        # plt.plot(test_acc)

    ''' Save means and stds NDArray values for inference '''
    mx.nd.save(outdir + args.architecture.lower() + '_means_stds_ihdp_' + str(train_experiments) + '_.nd',
               {"means": mx.nd.array(means), "stds": mx.nd.array(stds)})

    ''' Export trained model '''
    net.export(outdir + args.architecture.lower() + "-ihdp-predictions-" + str(train_experiments), epoch=epochs)

    log(logfile, '\n{} architecture total scores:'.format(args.architecture.upper()))

    ''' Train and test scores '''
    means, stds = np.mean(train_scores, axis=0), sem(train_scores, axis=0, ddof=0)
    r_pehe_mean, r_pehe_std = np.mean(np.sqrt(train_scores[:, 2]), axis=0), sem(np.sqrt(train_scores[:, 2]), axis=0,
                                                                                ddof=0)
    train_total_scores_str = 'train RMSE ITE: {:.2f} ± {:.2f}, train ATE: {:.2f} ± {:.2f}, train PEHE: {:.2f} ± {:.2f}, ' \
                             'train root PEHE: {:.2f} ± {:.2f}' \
                             ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2], r_pehe_mean, r_pehe_std)

    means, stds = np.mean(test_scores, axis=0), sem(test_scores, axis=0, ddof=0)
    r_pehe_mean, r_pehe_std = np.mean(np.sqrt(test_scores[:, 2]), axis=0), sem(np.sqrt(test_scores[:, 2]), axis=0,
                                                                               ddof=0)
    test_total_scores_str = 'test RMSE ITE: {:.2f} ± {:.2f}, test ATE: {:.2f} ± {:.2f}, test PEHE: {:.2f} ± {:.2f}, ' \
                            'test root PEHE: {:.2f} ± {:.2f}' \
                            ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2], r_pehe_mean, r_pehe_std)

    log(logfile, train_total_scores_str)
    log(logfile, test_total_scores_str)

    mean_duration = float("{0:.2f}".format(np.mean(train_durations, axis=0)[0]))

    with open(outdir + args.architecture.lower() + "-total-scores-" + str(train_experiments), "w",
              encoding="utf8") as text_file:
        print(train_total_scores_str, "\n", test_total_scores_str, file=text_file)

    return {"ite": "{:.2f} ± {:.2f}".format(means[0], stds[0]),
            "ate": "{:.2f} ± {:.2f}".format(means[1], stds[1]),
            "pehe": "{:.2f} ± {:.2f}".format(means[2], stds[2]),
            "mean_duration": mean_duration}


def run_test(args):
    ''' Set GPUs/CPUs '''
    num_gpus = mx.context.num_gpus()
    num_workers = int(args.num_workers)  # replace num_workers with the number of cores
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    batch_size_per_unit = int(args.batch_size_per_unit)  # mini-batch size
    batch_size = batch_size_per_unit * max(num_gpus, 1)

    ''' Load test dataset '''
    test_dataset = load_data('../' + args.data_dir + args.data_test)

    ''' Load training means and stds '''
    train_means_stds = mx.nd.load(args.means_stds)
    train_means = train_means_stds['means']
    train_stds = train_means_stds['stds']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = gluon.nn.SymbolBlock.imports(args.symbol, ['data'], args.params, ctx=ctx)

    ''' Calculate number of test experiments '''
    test_experiments = np.min([test_dataset['x'].shape[2], len(train_means)])

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

        ''' With-in sample '''
        test_evaluator = Evaluator(t, yf, ycf, mu0, mu1)

        ''' Retrieve training mean and std '''
        train_yf_m, train_yf_std = train_means[test_experiment].asnumpy(), train_stds[test_experiment].asnumpy()

        ''' Test dataset '''
        test_rmse_ite_dataset = gluon.data.ArrayDataset(mx.nd.array(x))

        ''' Test DataLoader '''
        test_rmse_ite_loader = gluon.data.DataLoader(test_rmse_ite_dataset, batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=num_workers)

        ''' Test model '''
        y_t0, y_t1 = predict_treated_and_controlled(net, test_rmse_ite_loader, ctx)
        y_t0, y_t1 = y_t0 * train_yf_std + train_yf_m, y_t1 * train_yf_std + train_yf_m
        test_score = test_evaluator.get_metrics(y_t1, y_t0)
        test_scores[test_experiment, :] = test_score

        print(
            '[Test Replication {}/{}]:\tRMSE ITE: {:0.3f},\t\t ATE: {:0.3f},\t\t PEHE: {:0.3f}'.format(
                test_experiment + 1,
                test_experiments,
                test_score[0],
                test_score[1],
                test_score[2]))

    means, stds = np.mean(test_scores, axis=0), sem(test_scores, axis=0, ddof=0)
    print('test RMSE ITE: {:.3f} ± {:.3f}, test ATE: {:.3f} ± {:.3f}, test PEHE: {:.3f} ± {:.3f}' \
          ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))
