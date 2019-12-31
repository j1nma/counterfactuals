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
from matplotlib import pyplot as plt
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
from scipy.stats import sem

from counterfactuals.evaluation import Evaluator
from counterfactuals.utilities import load_data, split_data_in_train_valid_test, test_net, \
    predict_treated_and_controlled


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


def softplus(x):
    return nd.log(1. + nd.exp(x))


def transform_rhos(rhos):
    return [softplus(rho) for rho in rhos]


def transform_gaussian_samples(mus, sigmas, epsilons):
    samples = []
    for j in range(len(mus)):
        samples.append(mus[j] + sigmas[j] * epsilons[j])
    return samples


def generate_weight_sample(layer_param_shapes, mus, rhos, ctx):
    # sample epsilons from standard normal
    epsilons = sample_epsilons(layer_param_shapes, ctx)

    # compute softplus for variance
    sigmas = transform_rhos(rhos)

    # obtain a sample from q(w|theta) by transforming the epsilons
    layer_params = transform_gaussian_samples(mus, sigmas, epsilons)

    return layer_params, sigmas


class BBBLoss(gluon.loss.Loss):
    def __init__(self, ctx, log_prior="gaussian", log_likelihood="softmax_cross_entropy",
                 sigma_p1=1.0, sigma_p2=0.1, pi=0.5, weight=None, batch_axis=0, **kwargs):
        super(BBBLoss, self).__init__(weight, batch_axis, **kwargs)
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.sigma_p1 = sigma_p1
        self.sigma_p2 = sigma_p2
        self.pi = pi
        self.ctx = ctx

    def log_softmax_likelihood(self, yhat_linear, y):
        return nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

    def log_gaussian(self, x, mu, sigma):
        return -0.5 * np.log(2.0 * np.pi) - nd.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)

    def gaussian_prior(self, x):
        sigma_p = nd.array([self.sigma_p1], ctx=self.ctx)
        return nd.sum(self.log_gaussian(x, 0., sigma_p))

    def gaussian(self, x, mu, sigma):
        scaling = 1.0 / nd.sqrt(2.0 * np.pi * (sigma ** 2))
        bell = nd.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))

        return scaling * bell

    def scale_mixture_prior(self, x):
        sigma_p1 = nd.array([self.sigma_p1], ctx=self.ctx)
        sigma_p2 = nd.array([self.sigma_p2], ctx=self.ctx)
        pi = self.pi

        first_gaussian = pi * self.gaussian(x, 0., sigma_p1)
        second_gaussian = (1 - pi) * self.gaussian(x, 0., sigma_p2)

        return nd.log(first_gaussian + second_gaussian)

    def hybrid_forward(self, F, output, label, params, mus, sigmas, num_batches, sample_weight=None):
        log_likelihood_sum = nd.sum(self.log_softmax_likelihood(output, label))
        prior = None
        if self.log_prior == "gaussian":
            prior = self.gaussian_prior
        elif self.log_prior == "scale_mixture":
            prior = self.scale_mixture_prior
        log_prior_sum = sum([nd.sum(prior(param)) for param in params])
        log_var_posterior_sum = sum(
            [nd.sum(self.log_gaussian(params[i], mus[i], sigmas[i])) for i in range(len(params))])
        return 1.0 / num_batches * (log_var_posterior_sum - log_prior_sum) - log_likelihood_sum


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
    # Hyperparameters
    epochs = int(args.iterations)
    learning_rate = float(args.learning_rate)
    hidden_size = int(args.hidden_size)
    train_experiments = int(args.experiments)
    learning_rate_factor = float(args.learning_rate_factor)
    learning_rate_steps = int(args.learning_rate_steps)  # changes the learning rate for every n updates.

    config = {
        "num_hidden_layers": 2,
        "num_hidden_units": 400,
        "batch_size": 128,
        "epochs": 10,
        "learning_rate": 0.001,
        "num_samples": 1,
        "pi": 0.25,
        "sigma_p": 1.0,
        "sigma_p1": 0.75,
        "sigma_p2": 0.01,
    }

    # Set GPUs/CPUs
    num_gpus = mx.context.num_gpus()
    num_workers = int(args.num_workers)  # replace num_workers with the number of cores
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    batch_size_per_unit = int(args.batch_size_per_unit)  # mini-batch size
    batch_size = batch_size_per_unit * max(num_gpus, 1)

    # Set seeds
    for c in ctx:
        mx.random.seed(int(args.seed), c)
    np.random.seed(int(args.seed))

    # Feed Forward Neural Network Model (4 hidden layers)
    net = ff4_relu_architecture(hidden_size)

    # Load datasets
    train_dataset = load_data('../' + args.data_dir + args.data_train)

    # Instantiate net
    ''' Param. init. '''
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    net.hybridize()

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

    # Initialize variational parameters; mean and variance for each weight
    mus = []
    rhos = []

    shapes = list(map(lambda x: x.shape, net.collect_params().values()))

    for shape in shapes:
        mu = gluon.Parameter('mu', shape=shape, init=mx.init.Normal(weight_scale))
        rho = gluon.Parameter('rho', shape=shape, init=mx.init.Constant(rho_offset))
        mu.initialize(ctx=ctx)
        rho.initialize(ctx=ctx)
        mus.append(mu)
        rhos.append(rho)

    variational_params = mus + rhos

    raw_mus = list(map(lambda x: x.data(ctx[0]), mus))
    raw_rhos = list(map(lambda x: x.data(ctx[0]), rhos))

    # Plot net graph TODO: change for all
    # x_sym = mx.sym.var('data')
    # sym = net(x_sym)
    # mx.viz.plot_network(sym, title=args.architecture.lower() + "_plot").view(
    #     filename=outdir + args.architecture.lower() + "_plot")

    # Metric, Loss and Optimizer
    rmse_metric = mx.metric.RMSE()
    bbb_loss = BBBLoss(ctx[0], log_prior="scale_mixture", sigma_p1=config['sigma_p1'], sigma_p2=config['sigma_p2'])
    scheduler = mx.lr_scheduler.FactorScheduler(step=learning_rate_steps, factor=learning_rate_factor,
                                                base_lr=learning_rate)
    optimizer = mx.optimizer.Adam(learning_rate=learning_rate, lr_scheduler=scheduler)
    # trainer = gluon.Trainer(net.collect_params(), optimizer=optimizer)
    trainer = gluon.Trainer(variational_params, optimizer=optimizer)

    # Initialize train score results
    train_scores = np.zeros((train_experiments, 3))

    # Initialize train experiment durations
    train_durations = np.zeros((train_experiments, 1))

    # Initialize test score results
    test_scores = np.zeros((train_experiments, 3))

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

        train, valid, test, _ = split_data_in_train_valid_test(x, t, yf, ycf, mu0, mu1)

        # With-in sample
        train_evaluator = Evaluator(np.concatenate([train['t'], valid['t']]),
                                    np.concatenate([train['yf'], valid['yf']]),
                                    y_cf=np.concatenate([train['ycf'], valid['ycf']], axis=0),
                                    mu0=np.concatenate([train['mu0'], valid['mu0']], axis=0),
                                    mu1=np.concatenate([train['mu1'], valid['mu1']], axis=0))
        test_evaluator = Evaluator(test['t'], test['yf'], test['ycf'], test['mu0'], test['mu1'])

        # Normalize yf
        yf_m, yf_std = np.mean(train['yf'], axis=0), np.std(train['yf'], axis=0)
        train['yf'] = (train['yf'] - yf_m) / yf_std
        valid['yf'] = (valid['yf'] - yf_m) / yf_std
        test['yf'] = (test['yf'] - yf_m) / yf_std

        # Save mean and std
        means = np.append(means, yf_m)
        stds = np.append(stds, yf_std)

        # Train dataset
        factual_features = np.hstack((train['x'], train['t']))
        train_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(factual_features), mx.nd.array(train['yf']))

        # With-in sample
        train_rmse_ite_dataset = gluon.data.ArrayDataset(mx.nd.array(np.concatenate([train['x'], valid['x']])))

        # Valid dataset
        valid_factual_features = np.hstack((valid['x'], valid['t']))
        valid_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(valid_factual_features), mx.nd.array(valid['yf']))

        # Test dataset
        test_rmse_ite_dataset = gluon.data.ArrayDataset(mx.nd.array(test['x']))

        # Train DataLoader
        train_factual_loader = gluon.data.DataLoader(train_factual_dataset, batch_size=batch_size, shuffle=True,
                                                     num_workers=num_workers)
        train_rmse_ite_loader = gluon.data.DataLoader(train_rmse_ite_dataset, batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers)

        # Valid DataLoader
        valid_factual_loader = gluon.data.DataLoader(valid_factual_dataset, batch_size=batch_size, shuffle=False,
                                                     num_workers=num_workers)

        # Test DataLoader
        test_rmse_ite_loader = gluon.data.DataLoader(test_rmse_ite_dataset, batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=num_workers)

        num_batch = len(train_factual_loader)

        train_start = time.time()

        smoothing_constant = .01
        train_acc = []
        test_acc = []

        # Train model
        for epoch in range(1, epochs + 1):  # start with epoch 1 for easier learning rate calculation

            start = time.time()
            train_loss = 0
            rmse_metric.reset()
            moving_loss = 0

            for i, (batch_f_features, batch_yf) in enumerate(train_factual_loader):
                # Get data and labels into slices and copy each slice into a context.
                batch_f_features = gluon.utils.split_and_load(batch_f_features, ctx_list=ctx, even_split=False)
                batch_yf = gluon.utils.split_and_load(batch_yf, ctx_list=ctx, even_split=False)

                # Forward
                with autograd.record():
                    # generate sample
                    layer_params, sigmas = generate_weight_sample(shapes, raw_mus, raw_rhos, ctx[0])

                    # overwrite network parameters with sampled parameters
                    for sample, param in zip(layer_params, net.collect_params().values()):
                        param._data[0] = sample

                    # forward-propagate the batch
                    outputs = [net(x) for x in batch_f_features]

                    # calculate the loss
                    loss = [bbb_loss(yhat, y, layer_params, raw_mus, sigmas, num_batch) for yhat, y in
                            zip(outputs, batch_yf)]

                # Backward
                for l in loss:
                    l.backward()

                # Optimize
                trainer.step(batch_size)

                train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
                moving_loss = (train_loss if ((i == 0) and (epoch == 0))
                               else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * train_loss)
                rmse_metric.update(batch_yf, outputs)

            if epoch % 50 == 0:
                _, train_rmse_factual = rmse_metric.get()
                train_loss /= num_batch
                _, valid_rmse_factual = test_net(net, valid_factual_loader, ctx)

                _, test_accuracy = evaluate_RMSE(valid_factual_loader, net, raw_mus, ctx)
                _, train_accuracy = evaluate_RMSE(train_factual_dataset, net, raw_mus, ctx)
                train_acc.append(np.asscalar(train_accuracy))
                test_acc.append(np.asscalar(test_accuracy))
                print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
                      (epoch, moving_loss, train_accuracy, test_accuracy))

                print('[Epoch %d/%d] Train-rmse-factual: %.3f, loss: %.3f | Valid-rmse-factual: %.3f | learning-rate: '
                      '%.3E' % (
                          epoch, epochs, train_rmse_factual, train_loss, valid_rmse_factual, trainer.learning_rate))

        train_durations[train_experiment, :] = time.time() - train_start

        # Test model
        y_t0, y_t1 = predict_treated_and_controlled(net, train_rmse_ite_loader, ctx)
        y_t0, y_t1 = y_t0 * yf_std + yf_m, y_t1 * yf_std + yf_m
        train_score = train_evaluator.get_metrics(y_t1, y_t0)
        train_scores[train_experiment, :] = train_score

        y_t0, y_t1 = predict_treated_and_controlled(net, test_rmse_ite_loader, ctx)
        y_t0, y_t1 = y_t0 * yf_std + yf_m, y_t1 * yf_std + yf_m
        test_score = test_evaluator.get_metrics(y_t1, y_t0)
        test_scores[train_experiment, :] = test_score

        print('[Train Replication {}/{}]: train RMSE ITE: {:0.3f}, train ATE: {:0.3f}, train PEHE: {:0.3f},' \
              ' test RMSE ITE: {:0.3f}, test ATE: {:0.3f}, test PEHE: {:0.3f}'.format(train_experiment + 1,
                                                                                      train_experiments,
                                                                                      train_score[0], train_score[1],
                                                                                      train_score[2],
                                                                                      test_score[0], test_score[1],
                                                                                      test_score[2]))
        plt.plot(train_acc)
        plt.plot(test_acc)

    # Save means and stds NDArray values for inference
    mx.nd.save(outdir + args.architecture.lower() + '_means_stds_ihdp_' + str(train_experiments) + '_.nd',
               {"means": mx.nd.array(means), "stds": mx.nd.array(stds)})

    # Export trained model
    net.export(outdir + args.architecture.lower() + "-ihdp-predictions-" + str(train_experiments), epoch=epochs)

    print('\n{} architecture total scores:'.format(args.architecture.upper()))

    means, stds = np.mean(train_scores, axis=0), sem(train_scores, axis=0, ddof=0)
    train_total_scores_str = 'train RMSE ITE: {:.2f} ± {:.2f}, train ATE: {:.2f} ± {:.2f}, train PEHE: {:.2f} ± {:.2f}' \
                             ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2])

    means, stds = np.mean(test_scores, axis=0), sem(test_scores, axis=0, ddof=0)
    test_total_scores_str = 'test RMSE ITE: {:.2f} ± {:.2f}, test ATE: {:.2f} ± {:.2f}, test PEHE: {:.2f} ± {:.2f}' \
                            ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2])

    print(train_total_scores_str)
    print(test_total_scores_str)

    mean_duration = float("{0:.2f}".format(np.mean(train_durations, axis=0)[0]))

    with open(outdir + args.architecture.lower() + "-total-scores-" + str(train_experiments), "w",
              encoding="utf8") as text_file:
        print(train_total_scores_str, "\n", test_total_scores_str, file=text_file)

    return {"ite": "{:.2f} ± {:.2f}".format(means[0], stds[0]),
            "ate": "{:.2f} ± {:.2f}".format(means[1], stds[1]),
            "pehe": "{:.2f} ± {:.2f}".format(means[2], stds[2]),
            "mean_duration": mean_duration}


def run_test(args):
    # Set GPUs/CPUs
    num_gpus = mx.context.num_gpus()
    num_workers = int(args.num_workers)  # replace num_workers with the number of cores
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    batch_size_per_unit = int(args.batch_size_per_unit)  # mini-batch size
    batch_size = batch_size_per_unit * max(num_gpus, 1)

    # Load test dataset
    test_dataset = load_data('../' + args.data_dir + args.data_test)

    # Load training means and stds
    train_means_stds = mx.nd.load(args.means_stds)
    train_means = train_means_stds['means']
    train_stds = train_means_stds['stds']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = gluon.nn.SymbolBlock.imports(args.symbol, ['data'], args.params, ctx=ctx)

    # Calculate number of test experiments
    test_experiments = np.min([test_dataset['x'].shape[2], len(train_means)])

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

        # With-in sample
        test_evaluator = Evaluator(t, yf, ycf, mu0, mu1)

        # Retrieve training mean and std
        train_yf_m, train_yf_std = train_means[test_experiment].asnumpy(), train_stds[test_experiment].asnumpy()

        # Test dataset
        test_rmse_ite_dataset = gluon.data.ArrayDataset(mx.nd.array(x))

        # Test DataLoader
        test_rmse_ite_loader = gluon.data.DataLoader(test_rmse_ite_dataset, batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=num_workers)

        # Test model
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
