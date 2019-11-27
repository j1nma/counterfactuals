# Convolutional neural network
import time
import warnings

import mxnet as mx
import numpy as np
from mxnet import gluon, autograd, init
from mxnet.gluon import nn
from scipy.stats import sem

from counterfactuals.evaluation import Evaluator
from counterfactuals.utilities import load_data, split_data_in_train_valid_test, test_net, \
    predict_treated_and_controlled, predict_treated_and_controlled_with_cnn


def cnn_architecture():
    NUM_OUTPUTS = 1  # number of classes
    NUM_FILTERS = 1  # number of convolutional filters per convolutional layer
    KERNEL_SIZE = 3
    PADDING = 1
    STRIDES = 2
    POOL_SIZE = 3
    FULLY_CONNECTED = 25  # number of unit in the fully connected dense layer

    net = nn.HybridSequential()

    net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, activation='relu'))
    net.add(gluon.nn.MaxPool1D(pool_size=POOL_SIZE, strides=STRIDES))
    net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=KERNEL_SIZE, activation='relu'))
    net.add(gluon.nn.MaxPool1D(pool_size=POOL_SIZE, strides=STRIDES))
    net.add(gluon.nn.Flatten())
    # net.add(gluon.nn.Dense(FULLY_CONNECTED, activation='relu'))
    net.add(gluon.nn.Dense(NUM_OUTPUTS))
    return net


def run(args, outdir):
    # Hyperparameters
    epochs = int(args.epochs)
    learning_rate = float(args.learning_rate)
    wd = float(args.weight_decay)
    hidden_size = int(args.hidden_size)
    train_experiments = int(args.train_experiments)
    learning_rate_factor = float(args.learning_rate_factor)
    learning_rate_steps = int(args.learning_rate_steps)  # changes the learning rate for every n updates.

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

    # Convolution Neural Network Model
    net = cnn_architecture()

    # Load datasets
    train_dataset = load_data('../' + args.data_dir + args.data_train)

    # Instantiate net
    net.initialize(init=init.Xavier(), ctx=ctx)
    net.hybridize()  # hybridize for better performance

    # Metric, Loss and Optimizer
    rmse_metric = mx.metric.RMSE()
    l2_loss = gluon.loss.L2Loss()
    scheduler = mx.lr_scheduler.FactorScheduler(step=learning_rate_steps, factor=learning_rate_factor,
                                                base_lr=learning_rate)
    optimizer = mx.optimizer.RMSProp(learning_rate=learning_rate, lr_scheduler=scheduler, wd=wd)
    trainer = gluon.Trainer(net.collect_params(), optimizer=optimizer)

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

        # Reshaping for CNN
        individuals_size = train_dataset['x'].shape[0]
        x = x.reshape((individuals_size, 1, 25))
        t = t.reshape((individuals_size, 1, 1))
        yf = yf.reshape((individuals_size, 1, 1))
        ycf = ycf.reshape((individuals_size, 1, 1))

        train, valid, test = split_data_in_train_valid_test(x, t, yf, ycf, mu0, mu1)

        # With-in sample
        train_evaluator = Evaluator(np.concatenate([train['t'], valid['t']]),
                                    np.concatenate([train['yf'], valid['yf']]),
                                    y_cf=np.concatenate([train['ycf'], valid['ycf']], axis=0),
                                    mu0=np.concatenate([train['mu0'], valid['mu0']], axis=0),
                                    mu1=np.concatenate([train['mu1'], valid['mu1']], axis=0))
        test_evaluator = Evaluator(test['t'], test['yf'], test['ycf'], test['mu0'], test['mu1'])

        # Normalize yf
        yf_m, yf_std = np.mean(train['yf'], axis=0), np.std(train['yf'], axis=0)  # todo fijate bien
        train['yf'] = (train['yf'] - yf_m) / yf_std
        valid['yf'] = (valid['yf'] - yf_m) / yf_std
        test['yf'] = (test['yf'] - yf_m) / yf_std

        # Save mean and std
        means = np.append(means, yf_m)
        stds = np.append(stds, yf_std)

        # Train dataset
        # factual_features = np.stack((train['x'], train['t']), axis=-1)
        factual_features = np.block([train['x'], train['t']])
        train_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(factual_features), mx.nd.array(train['yf']))

        # With-in sample
        train_rmse_ite_dataset = gluon.data.ArrayDataset(mx.nd.array(np.concatenate([train['x'], valid['x']])))

        # Valid dataset
        # valid_factual_features = np.stack((valid['x'], valid['t']), axis=-1)
        valid_factual_features = np.block([valid['x'], valid['t']])
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

        # Train model
        for epoch in range(1, epochs + 1):  # start with epoch 1 for easier learning rate calculation

            start = time.time()
            train_loss = 0
            rmse_metric.reset()

            for i, (batch_f_features, batch_yf) in enumerate(train_factual_loader):
                # Get data and labels into slices and copy each slice into a context.
                batch_f_features = gluon.utils.split_and_load(batch_f_features, ctx_list=ctx, even_split=False)
                batch_yf = gluon.utils.split_and_load(batch_yf, ctx_list=ctx, even_split=False)

                # Forward
                with autograd.record():
                    outputs = [net(x) for x in batch_f_features]
                    loss = [l2_loss(yhat, y) for yhat, y in zip(outputs, batch_yf)]

                # Backward
                for l in loss:
                    l.backward()

                # Optimize
                trainer.step(batch_size)

                train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
                rmse_metric.update(batch_yf, outputs)

            _, train_rmse_factual = rmse_metric.get()
            train_loss /= num_batch
            _, valid_rmse_factual = test_net(net, valid_factual_loader, ctx)

            if epoch % 50 == 0:
                print(
                    '[Epoch %d/%d] Train-rmse-factual: %.3f, loss: %.3f | Valid-rmse-factual: %.3f | learning-rate: '
                    '%.3E' %
                    (epoch, epochs, train_rmse_factual, train_loss, valid_rmse_factual, trainer.learning_rate))

        train_durations[train_experiment, :] = time.time() - train_start

        # Reshape for CNN testing
        train_evaluator.y_f = train_evaluator.y_f.reshape(-1)
        train_evaluator.y_cf = train_evaluator.y_cf.reshape(-1)
        test_evaluator.y_f = test_evaluator.y_f.reshape(-1)
        test_evaluator.y_cf = test_evaluator.y_cf.reshape(-1)

        # Test model
        y_t0, y_t1 = predict_treated_and_controlled_with_cnn(net, train_rmse_ite_loader, ctx)
        y_t0, y_t1 = y_t0 * yf_std + yf_m, y_t1 * yf_std + yf_m
        y_t0, y_t1 = y_t0.reshape(-1), y_t1.reshape(-1)
        train_score = train_evaluator.get_metrics(y_t1, y_t0)
        train_scores[train_experiment, :] = train_score

        y_t0, y_t1 = predict_treated_and_controlled_with_cnn(net, test_rmse_ite_loader, ctx)
        y_t0, y_t1 = y_t0 * yf_std + yf_m, y_t1 * yf_std + yf_m
        y_t0, y_t1 = y_t0.reshape(-1), y_t1.reshape(-1)
        test_score = test_evaluator.get_metrics(y_t1, y_t0)
        test_scores[train_experiment, :] = test_score

        print('[Train Replication {}/{}]: train RMSE ITE: {:0.3f}, train ATE: {:0.3f}, train PEHE: {:0.3f},' \
              ' test RMSE ITE: {:0.3f}, test ATE: {:0.3f}, test PEHE: {:0.3f}'.format(train_experiment + 1,
                                                                                      train_experiments,
                                                                                      train_score[0], train_score[1],
                                                                                      train_score[2],
                                                                                      test_score[0], test_score[1],
                                                                                      test_score[2]))

    # Save means and stds NDArray values for inference
    mx.nd.save(outdir + 'nn4_means_stds_ihdp_' + str(train_experiments) + '_.nd',
               {"means": mx.nd.array(means), "stds": mx.nd.array(stds)})

    # Export trained model
    net.export(outdir + "nn4-ihdp-predictions-" + str(train_experiments), epoch=epochs)

    print('\n{} architecture total scores:'.format(args.architecture.upper()))
    means, stds = np.mean(train_scores, axis=0), sem(train_scores, axis=0, ddof=0)
    print('train RMSE ITE: {:.2f}±{:.2f}, train ATE: {:.2f}±{:.2f}, train PEHE: {:.2f}±{:.2f}' \
          ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))

    means, stds = np.mean(test_scores, axis=0), sem(test_scores, axis=0, ddof=0)
    print('test RMSE ITE: {:.2f}±{:.2f}, test ATE: {:.2f}±{:.2f}, test PEHE: {:.2f}±{:.2f}' \
          ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))

    mean_duration = float("{0:.2f}".format(np.mean(train_durations, axis=0)[0]))
    return {"ite": "{:.2f}±{:.2f}".format(means[0], stds[0]),
            "ate": "{:.2f}±{:.2f}".format(means[1], stds[1]),
            "pehe": "{:.2f}±{:.2f}".format(means[2], stds[2]),
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
        net = gluon.nn.SymbolBlock.imports(args.symbol, ['data'], args.params,
                                           ctx=ctx)

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
            '[Test Replication {}/{}]: RMSE ITE: {:0.3f}, ATE: {:0.3f}, PEHE: {:0.3f},'.format(test_experiment + 1,
                                                                                               test_experiments,
                                                                                               test_score[0],
                                                                                               test_score[1],
                                                                                               test_score[2]))

    means, stds = np.mean(test_scores, axis=0), sem(test_scores, axis=0, ddof=0)
    print('test RMSE ITE: {:.3f}±{:.3f}, test ATE: {:.3f}±{:.3f}, test PEHE: {:.3f}±{:.3f}' \
          ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))
