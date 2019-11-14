# Standard feed-forward neural network
# Trained with 4 hidden layers, to predict the factual outcome based on X and t, without a penalty for imbalance.
#
# Referred to as NN-4 from Johansson et al. paper:
# "Learning Representations for Counterfactual Inference"
# arXiv:1605.03661v3 [stat.ML] 6 Jun 2018

import time

import mxnet as mx
import numpy as np
from mxnet import gluon, autograd, init
from mxnet.gluon import data, nn
from scipy import stats

from counterfactuals.effects import get_errors
from counterfactuals.utilities import load_data

gpu = False
ctx = mx.gpu() if gpu else mx.cpu()

start = time.time()

# mx.random.seed(int(round(start * 1000)), ctx)
mx.random.seed(1, ctx)
np.random.seed(1)

# Train Hyperparameters
input_size = 26
hidden_size = 25
batch_size = 100  # mini-batch size
learning_rate = 0.001
l2_weight_decay_lambda = 0.001
train_experiments = 100
epochs = 200


# Feed Forward Neural Network Model (4 hidden layers)
class NN4Net(nn.Block):
    def __init__(self, hidden_nodes_size):
        super(NN4Net, self).__init__()
        with self.name_scope():
            self.fc1 = nn.Dense(hidden_nodes_size, activation='relu')
            self.fc2 = nn.Dense(hidden_nodes_size, activation='relu')
            self.fc3 = nn.Dense(hidden_nodes_size, activation='relu')
            self.fc4 = nn.Dense(hidden_nodes_size, activation='relu')
            self.fc5 = nn.Dense(1)

    def forward(self, input):
        return self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(input)))))


# Load datasets
train_dataset = load_data('../data/ihdp_npci_1-100.train.npz')
test_dataset = load_data('../data/ihdp_npci_1-100.test.npz')

# Instantiate net
net = NN4Net(hidden_size)
net.initialize(init=init.Xavier(), ctx=ctx)  # TODO review Xavier

# Loss and Optimizer
criterion = gluon.loss.L2Loss()
optimizer = gluon.Trainer(net.collect_params(),
                          'RMSProp',
                          {'learning_rate': learning_rate, 'wd': l2_weight_decay_lambda})

# Train
for experiment in range(train_experiments):

    # Create training dataset
    x = train_dataset['x'][:, :, experiment]
    t = np.reshape(train_dataset['t'][:, experiment], (-1, 1))
    yf = train_dataset['yf'][:, experiment]
    ycf = train_dataset['ycf'][:, experiment]
    factual_features = np.hstack((x, t))
    train_factual_dataset = gluon.data.ArrayDataset(mx.nd.array(factual_features), mx.nd.array(yf))

    # Train Data Loaders
    train_factual_loader = data.DataLoader(train_factual_dataset, batch_size, shuffle=True)

    # Train the Model
    for epoch in range(epochs):
        for i, (batch_f_features, batch_yf) in enumerate(train_factual_loader):
            batch_f_features = batch_f_features.reshape((-1, input_size))
            batch_f_features, batch_yf = batch_f_features.as_in_context(ctx), batch_yf.as_in_context(ctx)

            # Forward, Backward and Optimize
            with autograd.record():
                outputs = net(batch_f_features)
                loss = criterion(outputs, batch_yf)
            loss.backward()
            optimizer.step(batch_size)

            if (i + 1) % 2 == 0:
                print('Train Experiment %d, Epoch [%d/%d], Step [%d/%d], RMSE Loss: %.4f'
                      % (experiment, epoch + 1, epochs, i + 1,
                         len(train_factual_dataset) // batch_size,  # // is the floor division operator
                         loss.sum().asscalar() ** 0.5))

# Test hyperparameters
test_experiments = test_dataset['x'].shape[2]
test_batch_size = test_dataset['x'].shape[0]  # batch is complete size of testing data

# Result arrays
rmse_ite_arr = np.array([])
abs_ate_arr = np.array([])
pehe_arr = np.array([])

# Test
for test_experiment in range(test_experiments):

    # Create testing dataset
    x = test_dataset['x'][:, :, test_experiment]
    t = np.reshape(test_dataset['t'][:, test_experiment], (-1, 1))
    yf = test_dataset['yf'][:, test_experiment]
    ycf = test_dataset['ycf'][:, test_experiment]
    mu0 = test_dataset['mu0'][:, test_experiment]
    mu1 = test_dataset['mu1'][:, test_experiment]

    factual_features = np.hstack((x, t))
    counterfactual_features = np.hstack((x, np.ones(len(t)).reshape(len(t), -1) - t))

    test_factual_dataset = gluon.data.ArrayDataset(
        mx.nd.array(factual_features),
        mx.nd.array(counterfactual_features),
        mx.nd.array(yf),
        mx.nd.array(ycf),
        mx.nd.array(mu0),
        mx.nd.array(mu1))

    # Test Data Loader
    test_factual_loader = data.DataLoader(test_factual_dataset, test_batch_size, shuffle=False)

    # Test the model in the factual and counterfactual regimes
    y_treated_predicted = np.array([])
    y_controlled_predicted = np.array([])
    y_treated_true = np.array([])
    y_controlled_true = np.array([])
    for batch_f_features, batch_cf_features, batch_yf, batch_ycf, batch_mu0, batch_mu1 in test_factual_loader:
        batch_f_features = batch_f_features.reshape((-1, input_size))
        batch_cf_features = batch_cf_features.reshape((-1, input_size))

        batch_f_features, batch_cf_features, batch_yf, batch_ycf, batch_mu0, batch_mu1 = \
            batch_f_features.as_in_context(ctx), \
            batch_cf_features.as_in_context(ctx), \
            batch_yf.as_in_context(ctx), \
            batch_ycf.as_in_context(ctx), \
            batch_mu0.as_in_context(ctx), \
            batch_mu1.as_in_context(ctx)

        factual_predict = net(batch_f_features)
        counterfactual_predict = net(batch_cf_features)

        for idx, feature_arr in enumerate(batch_f_features):
            if feature_arr[-1].asscalar() == 1:
                # Factual regime
                y_treated_predicted = np.append(y_treated_predicted, factual_predict.asnumpy()[idx])
                y_treated_true = np.append(y_treated_true, batch_mu1.asnumpy()[idx])

                # Counterfactual regime
                y_controlled_predicted = np.append(y_controlled_predicted, counterfactual_predict.asnumpy()[idx])
                y_controlled_true = np.append(y_controlled_true, batch_mu0.asnumpy()[idx])
            else:
                # Factual regime
                y_controlled_predicted = np.append(y_controlled_predicted, factual_predict.asnumpy()[idx])
                y_controlled_true = np.append(y_controlled_true, batch_mu0.asnumpy()[idx])

                # Counterfactual regime
                y_treated_predicted = np.append(y_treated_predicted, counterfactual_predict.asnumpy()[idx])
                y_treated_true = np.append(y_treated_true, batch_mu1.asnumpy()[idx])

    rmse_ite, abs_ate, pehe = get_errors(t, yf, y_treated_predicted, y_controlled_predicted, y_treated_true,
                                         y_controlled_true)
    rmse_ite_arr = np.append(rmse_ite_arr, rmse_ite)
    abs_ate_arr = np.append(abs_ate_arr, abs_ate)
    pehe_arr = np.append(pehe_arr, pehe)

    print('Test experiment {:d}, RMSE ITE: {:.1f}, abs ATE: {:.1f}, PEHE: {:.1f}' \
          ''.format(test_experiment, rmse_ite, abs_ate, pehe))

print('{:d} test experiments: RMSE ITE: {:.1f} ± {:.1f}, abs ATE: {:.1f} ± {:.1f}, PEHE: {:.1f} ± {:.1f}' \
      ''.format(test_experiments,
                np.mean(rmse_ite_arr), stats.sem(rmse_ite_arr, ddof=0),
                np.mean(abs_ate_arr), stats.sem(abs_ate_arr, ddof=0),
                np.mean(pehe_arr), stats.sem(pehe_arr, ddof=0)))
