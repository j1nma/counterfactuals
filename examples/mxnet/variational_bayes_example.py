from __future__ import print_function

import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet import nd, autograd

config = {
    "num_hidden_layers": 2,
    "num_hidden_units": 200,
    "batch_size": 16,
    "epochs": 1,
    "learning_rate": 0.001,
    "num_samples": 1,
    "pi": 0.25,
    "sigma_p": 1.0,
    "sigma_p1": 0.75,
    "sigma_p2": 0.01,
}

ctx = mx.cpu()

''' Load MNIST data'''


def transform(data, label):
    return data.astype(np.float32) / 126.0, label.astype(np.float32)


mnist = mx.test_utils.get_mnist()
num_inputs = 784
num_outputs = 10
batch_size = config['batch_size']

train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

num_train = sum([batch_size for i in train_data])
num_batches = num_train / batch_size

''' Neural net modeling '''
num_layers = config['num_hidden_layers']
num_hidden = config['num_hidden_units']

net = gluon.nn.Sequential()
with net.name_scope():
    for i in range(num_layers):
        net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))

''' Build objective/loss '''


class BBBLoss(gluon.loss.Loss):
    def __init__(self, log_prior="gaussian", log_likelihood="softmax_cross_entropy",
                 sigma_p1=1.0, sigma_p2=0.1, pi=0.5, weight=None, batch_axis=0, **kwargs):
        super(BBBLoss, self).__init__(weight, batch_axis, **kwargs)
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.sigma_p1 = sigma_p1
        self.sigma_p2 = sigma_p2
        self.pi = pi

    def log_softmax_likelihood(self, yhat_linear, y):
        return nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

    def log_gaussian(self, x, mu, sigma):
        return -0.5 * np.log(2.0 * np.pi) - nd.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)

    def log_exponential(self, x, lmbda):
        # return np.log(scipy.stats.expon.pdf(x, loc=0.0, scale=1.0 / lmbda) + 1e-6)
        return np.log(lmbda) - (lmbda * x)
        # return nd.log(lmbda * nd.exp(nd.negative(lmbda * x)) + 1e-6)

    def gaussian_prior(self, x):
        sigma_p = nd.array([self.sigma_p1], ctx=ctx)
        return nd.sum(self.log_gaussian(x, 0., sigma_p))

    def exponential_prior(self, x):
        # lambda_p = nd.array([1.0], ctx=ctx)
        return nd.sum(self.log_exponential(x, 150.0))

    def gaussian(self, x, mu, sigma):
        scaling = 1.0 / nd.sqrt(2.0 * np.pi * (sigma ** 2))
        bell = nd.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))

        return scaling * bell

    def scale_mixture_prior(self, x):
        sigma_p1 = nd.array([self.sigma_p1], ctx=ctx)
        sigma_p2 = nd.array([self.sigma_p2], ctx=ctx)
        pi = self.pi

        first_gaussian = pi * self.gaussian(x, 0., sigma_p1)
        second_gaussian = (1 - pi) * self.gaussian(x, 0., sigma_p2)

        return nd.log(first_gaussian + second_gaussian)

    def hybrid_forward(self, F, output, label, params, lambdas, sample_weight=None):
        log_likelihood_sum = nd.sum(self.log_softmax_likelihood(output, label))
        # print("output:\t" + str(output))
        # print("label:\t" + str(label))
        # print("log_likelihood_sum:\t" + str(log_likelihood_sum))
        prior = None
        if self.log_prior == "gaussian":
            prior = self.gaussian_prior
        elif self.log_prior == "scale_mixture":
            prior = self.scale_mixture_prior
        elif self.log_prior == "exponential":
            prior = self.exponential_prior
        log_prior_sum = sum([nd.sum(prior(param)) for param in params])
        log_var_posterior_sum = sum(
            [nd.sum(self.log_exponential(params[i], lambdas[i])) for i in range(len(params))])
        # print("log_prior_sum:\t" + str(log_prior_sum))
        # print("log_var_posterior_sum:\t" + str(log_var_posterior_sum))
        return 1.0 / num_batches * (log_var_posterior_sum - log_prior_sum) - log_likelihood_sum

    # def hybrid_forward(self, F, output, label, params, mus, sigmas, sample_weight=None):
    #     log_likelihood_sum = nd.sum(self.log_softmax_likelihood(output, label))
    #     prior = None
    #     if self.log_prior == "gaussian":
    #         prior = self.gaussian_prior
    #     elif self.log_prior == "scale_mixture":
    #         prior = self.scale_mixture_prior
    #     log_prior_sum = sum([nd.sum(prior(param)) for param in params])
    #     log_var_posterior_sum = sum(
    #         [nd.sum(self.log_gaussian(params[i], mus[i], sigmas[i])) for i in range(len(params))])
    #     return 1.0 / num_batches * (log_var_posterior_sum - log_prior_sum) - log_likelihood_sum


bbb_loss = BBBLoss(log_prior="exponential", sigma_p1=config['sigma_p1'], sigma_p2=config['sigma_p2'])

''' Param. init. '''
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

for i, (data, label) in enumerate(train_data):
    data = data.as_in_context(ctx).reshape((-1, 784))
    net(data)
    break

weight_scale = .1
rho_offset = -3
lambda_scale = 150.0

# initialize variational parameters; mean and variance for each weight
mus = []
rhos = []
lambdas = []

shapes = list(map(lambda x: x.shape, net.collect_params().values()))

for shape in shapes:
    mu = gluon.Parameter('mu', shape=shape, init=mx.init.Normal(weight_scale))
    rho = gluon.Parameter('rho', shape=shape, init=mx.init.Constant(rho_offset))
    lbd = gluon.Parameter('lbd', shape=shape, init=mx.init.Constant(lambda_scale))
    mu.initialize(ctx=ctx)
    rho.initialize(ctx=ctx)
    lbd.initialize(ctx=ctx)
    mus.append(mu)
    rhos.append(rho)
    lambdas.append(lbd)

# variational_params = mus + rhos
variational_params = lambdas

raw_mus = list(map(lambda x: x.data(ctx), mus))
raw_rhos = list(map(lambda x: x.data(ctx), rhos))
raw_lambdas = list(map(lambda x: x.data(ctx), lambdas))

''' Optimizer '''
trainer = gluon.Trainer(variational_params, 'adam', {'learning_rate': config['learning_rate']})

''' Main training loop '''
''' Sampling'''


def sample_epsilons(param_shapes):
    epsilons = [nd.random_normal(shape=shape, loc=0., scale=1.0, ctx=ctx) for shape in param_shapes]
    return epsilons


def sample_expos(param_shapes, lambdas):
    epsilons = [nd.random_exponential(lam=lambdas[idx][0][0].asscalar(), shape=shape, ctx=ctx) for idx, shape in
                enumerate(param_shapes)]
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


def transform_exponential_samples(expos):
    samples = []
    for j in range(len(expos)):
        samples.append(expos[j])
    return samples


def generate_weight_sample(layer_param_shapes, lambdas):
    # sample epsilons from standard normal
    # epsilons = sample_epsilons(layer_param_shapes)
    expos = sample_expos(layer_param_shapes, lambdas)

    # compute softplus for variance
    # sigmas = transform_rhos(rhos)

    # obtain a sample from q(w|theta) by transforming the epsilons
    # layer_params = transform_gaussian_samples(mus, sigmas, epsilons)
    layer_params = transform_exponential_samples(expos)

    return layer_params


''' Metric '''


def evaluate_accuracy(data_iterator, net, layer_params):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)

        for l_param, param in zip(layer_params, net.collect_params().values()):
            param._data[0] = l_param

        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()


''' Complete loop '''
epochs = config['epochs']
learning_rate = config['learning_rate']
smoothing_constant = .01
train_acc = []
test_acc = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)

        with autograd.record():
            # generate sample
            layer_params = generate_weight_sample(shapes, raw_lambdas)

            # overwrite network parameters with sampled parameters
            for sample, param in zip(layer_params, net.collect_params().values()):
                param._data[0] = sample

            # print("net.collect_params() GRAD:\t" + str(net.collect_params()._params['sequential0_dense0_weight']._grad[0][0][:10]))
            # forward-propagate the batch
            if i == 32:
                b = 1

            output = net(data)

            output = output + 1e-8

            # calculate the loss
            loss = bbb_loss(output, label_one_hot, layer_params, raw_lambdas)

            # backpropagate for gradient calculation
            loss.backward()

        trainer.step(data.shape[0])

        # calculate moving loss for monitoring convergence
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

        if curr_loss == 0.0:
            a = 1
        print("curr_loss:\t" + str(curr_loss))
        # print("output:\t" + str(output[0][0].asscalar()))

    test_accuracy = evaluate_accuracy(test_data, net, layer_params)
    train_accuracy = evaluate_accuracy(train_data, net, layer_params)
    train_acc.append(np.asscalar(train_accuracy))
    test_acc.append(np.asscalar(test_accuracy))
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, moving_loss, train_accuracy, test_accuracy))

# plt.plot(train_acc)
# plt.plot(test_acc)
# plt.show()
