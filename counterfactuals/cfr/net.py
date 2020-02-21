import mxnet as mx
import numpy as np
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.loss import Loss

BATCH_NORM_EPSILON = 1e-3


# todo what about rep_lay and reg_lay? clearly hardcoded here to 3
class CFRNet(nn.HybridBlock):
    def __init__(self, rep_hidden_size, hyp_hidden_size, weight_init_scale, dim_input, batch_norm, **kwargs):
        nn.HybridBlock.__init__(self, **kwargs)

        self.input_shape = None
        self.batch_norm = batch_norm

        with self.name_scope():
            self.relu = nn.Activation(activation='relu')

            # Representation Layers
            self.rep_fc1 = nn.Dense(rep_hidden_size,
                                    in_units=dim_input,
                                    weight_initializer=mx.init.Normal(
                                        sigma=weight_init_scale / np.sqrt(dim_input)))

            if self.batch_norm:
                self.rep_fc1_bn = nn.BatchNorm(epsilon=BATCH_NORM_EPSILON, in_channels=rep_hidden_size)

            self.rep_fc2 = nn.Dense(rep_hidden_size,
                                    in_units=rep_hidden_size,
                                    weight_initializer=mx.init.Normal(
                                        sigma=weight_init_scale / np.sqrt(rep_hidden_size)))

            if self.batch_norm:
                self.rep_fc2_bn = nn.BatchNorm(epsilon=BATCH_NORM_EPSILON, in_channels=rep_hidden_size)

            self.rep_fc3 = nn.Dense(rep_hidden_size,
                                    in_units=rep_hidden_size,
                                    weight_initializer=mx.init.Normal(
                                        sigma=weight_init_scale / np.sqrt(rep_hidden_size)))

            if self.batch_norm:
                self.rep_fc3_bn = nn.BatchNorm(epsilon=BATCH_NORM_EPSILON, in_channels=rep_hidden_size)

            # Hypothesis Layers for t = 1
            self.t1_hyp_fc1 = nn.Dense(hyp_hidden_size,
                                       activation='relu',
                                       in_units=rep_hidden_size,
                                       weight_initializer=mx.init.Normal(
                                           sigma=weight_init_scale / np.sqrt(rep_hidden_size)))
            self.t1_hyp_fc2 = nn.Dense(hyp_hidden_size,
                                       activation='relu',
                                       in_units=hyp_hidden_size,
                                       weight_initializer=mx.init.Normal(
                                           sigma=weight_init_scale / np.sqrt(hyp_hidden_size)))
            self.t1_hyp_fc3 = nn.Dense(hyp_hidden_size,
                                       activation='relu',
                                       in_units=hyp_hidden_size,
                                       weight_initializer=mx.init.Normal(
                                           sigma=weight_init_scale / np.sqrt(hyp_hidden_size)))
            self.t1_hyp_fc4 = nn.Dense(1)

            # Hypothesis Layers for t = 0
            self.t0_hyp_fc1 = nn.Dense(hyp_hidden_size,
                                       activation='relu',
                                       in_units=rep_hidden_size,
                                       weight_initializer=mx.init.Normal(
                                           sigma=weight_init_scale / np.sqrt(rep_hidden_size)))
            self.t0_hyp_fc2 = nn.Dense(hyp_hidden_size,
                                       activation='relu',
                                       in_units=hyp_hidden_size,
                                       weight_initializer=mx.init.Normal(
                                           sigma=weight_init_scale / np.sqrt(hyp_hidden_size)))
            self.t0_hyp_fc3 = nn.Dense(hyp_hidden_size,
                                       activation='relu',
                                       in_units=hyp_hidden_size,
                                       weight_initializer=mx.init.Normal(
                                           sigma=weight_init_scale / np.sqrt(hyp_hidden_size)))
            self.t0_hyp_fc4 = nn.Dense(1)

    def forward(self, x, t1_indices, t0_indices):
        self.input_shape = x.shape[0]

        return HybridBlock.forward(self, x, t1_indices, t0_indices)

    def hybrid_forward(self, F, x, t1_indices, t0_indices):
        if self.batch_norm:
            rep_relu1 = self.relu(self.rep_fc1_bn(self.rep_fc1(x)))
            rep_relu2 = self.relu(self.rep_fc2_bn(self.rep_fc2(rep_relu1)))
            rep_relu3 = self.relu(self.rep_fc3_bn(self.rep_fc3(rep_relu2)))
        else:
            rep_relu1 = self.relu(self.rep_fc1(x))
            rep_relu2 = self.relu(self.rep_fc2(rep_relu1))
            rep_relu3 = self.relu(self.rep_fc3(rep_relu2))

        if F.size_array(t1_indices).__getitem__(0).__gt__(0).__bool__:
            t1_hyp_relu1 = self.t1_hyp_fc1(F.take(rep_relu3, t1_indices))
            t1_hyp_relu2 = self.t1_hyp_fc2(t1_hyp_relu1)
            t1_hyp_relu3 = self.t1_hyp_fc3(t1_hyp_relu2)
            t1_hyp_relu4 = self.t1_hyp_fc4(t1_hyp_relu3)

        if F.size_array(t0_indices).__getitem__(0).__gt__(0).__bool__:
            t0_hyp_relu1 = self.t0_hyp_fc1(F.take(rep_relu3, t0_indices))
            t0_hyp_relu2 = self.t0_hyp_fc2(t0_hyp_relu1)
            t0_hyp_relu3 = self.t0_hyp_fc3(t0_hyp_relu2)
            t0_hyp_relu4 = self.t0_hyp_fc4(t0_hyp_relu3)

        return t1_hyp_relu4, t0_hyp_relu4, rep_relu3


class WassersteinLoss(Loss):
    # For the purpose of calculating the Wasserstein distance between distribution,
    # the algorithm from below was adapted to MXNet from
    # https://github.com/clinicalml/cfrnet/blob/master/cfr/util.py

    def __init__(self, p=0.5, lam=10, its=10, square=False, backpropT=False, weight=None, batch_axis=0,
                 **kwargs):
        super(WassersteinLoss, self).__init__(weight, batch_axis, **kwargs)
        self.p = p
        self.lam = lam
        self.its = its
        self.square = square
        self.backpropT = backpropT

    def hybrid_forward(self, F, Xt, Xc):
        from counterfactuals.utilities import mx_safe_sqrt, mx_pdist2sq

        nt = np.float(Xt.shape[0])
        nc = np.float(Xc.shape[0])

        ''' Compute distance matrix (opposite to clinicalml) '''
        if self.square:
            M = mx_safe_sqrt(mx_pdist2sq(Xt, Xc))
        else:
            M = mx_pdist2sq(Xt, Xc)

        ''' Estimate lambda and delta '''
        M_mean = mx.nd.mean(M)
        delta = mx.nd.stop_gradient(mx.nd.max(M))
        eff_lam = mx.nd.stop_gradient(self.lam / M_mean)

        ''' Compute new distance matrix '''
        row = delta * mx.nd.ones_like(mx.nd.slice_axis(M, axis=0, begin=0, end=1))
        col = mx.nd.Concat(delta * mx.nd.ones_like(mx.nd.slice_axis(M, axis=1, begin=0, end=1)),
                           mx.nd.zeros((1, 1)), dim=0)
        Mt = mx.nd.Concat(M, row, dim=0)
        Mt = mx.nd.Concat(Mt, col, dim=1)

        ''' Compute marginal vectors '''
        # j1nma edit, mx -> np -> mx, not encouraged
        a = mx.nd.Concat(
            mx.nd.array(self.p * mx.nd.ones_like(mx.nd.slice_axis(Xt, axis=1, begin=0, end=1)).asnumpy() / nt),
            mx.nd.array((1 - self.p) * mx.nd.ones((1, 1)).asnumpy()),
            dim=0)

        # j1nma edit, mx -> np -> mx, not encouraged
        b = mx.nd.Concat(
            mx.nd.array((1 - self.p) * mx.nd.ones_like(mx.nd.slice_axis(Xc, axis=1, begin=0, end=1)).asnumpy() / nc),
            mx.nd.array(self.p * mx.nd.ones((1, 1)).asnumpy()),
            dim=0)

        ''' Compute kernel matrix'''
        Mlam = eff_lam * Mt
        K = mx.nd.exp(-Mlam) + 1e-6  # added constant to avoid nan
        ainvK = K / a

        u = a
        for i in range(0, self.its):
            u = 1.0 / (
                mx.nd.dot(ainvK, (b / mx.nd.transpose(mx.nd.dot(mx.nd.transpose(u), K)))))
        v = b / (mx.nd.transpose(mx.nd.dot(mx.nd.transpose(u), K)))

        T = u * (mx.nd.transpose(v) * K)

        if not self.backpropT:
            T = mx.nd.stop_gradient(T)

        E = T * Mt
        D = 2 * mx.nd.sum(E)

        return D
