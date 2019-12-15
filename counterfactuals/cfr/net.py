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

        with self.name_scope():
            self.nonlin = nn.Activation(activation='relu')

            # Representation Layers
            self.rep_fc1 = nn.Dense(rep_hidden_size,
                                    in_units=dim_input,
                                    weight_initializer=mx.init.Normal(
                                        sigma=weight_init_scale / np.sqrt(dim_input)))

            if batch_norm:
                self.rep_fc1_bn = nn.BatchNorm(epsilon=BATCH_NORM_EPSILON, in_channels=rep_hidden_size)

            self.rep_fc2 = nn.Dense(rep_hidden_size,
                                    in_units=rep_hidden_size,
                                    weight_initializer=mx.init.Normal(
                                        sigma=weight_init_scale / np.sqrt(rep_hidden_size)))

            if batch_norm:
                self.rep_fc2_bn = nn.BatchNorm(epsilon=BATCH_NORM_EPSILON, in_channels=rep_hidden_size)

            self.rep_fc3 = nn.Dense(rep_hidden_size,
                                    in_units=rep_hidden_size,
                                    weight_initializer=mx.init.Normal(
                                        sigma=weight_init_scale / np.sqrt(rep_hidden_size)))

            if batch_norm:
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

    def forward(self, x, t1_indices, t0_indices, batch_norm):
        self.input_shape = x.shape[0]

        return HybridBlock.forward(self, x, t1_indices, t0_indices, batch_norm)

    def hybrid_forward(self, F, x, t1_indices, t0_indices, batch_norm):
        if batch_norm:
            rep_relu1 = self.nonlin(self.rep_fc1_bn(self.rep_fc1(x)))
            rep_relu2 = self.nonlin(self.rep_fc2_bn(self.rep_fc2(rep_relu1)))
            rep_relu3 = self.nonlin(self.rep_fc3_bn(self.rep_fc3(rep_relu2)))
        else:
            rep_relu1 = self.nonlin(self.rep_fc1(x))
            rep_relu2 = self.nonlin(self.rep_fc2(rep_relu1))
            rep_relu3 = self.nonlin(self.rep_fc3(rep_relu2))

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


class Wasserstein(Loss):
    def __init__(self, margin=6., weight=None, batch_axis=0, **kwargs):
        super(Wasserstein, self).__init__(weight, batch_axis, **kwargs)
        self.margin = margin

    def hybrid_forward(self, F, image1, image2, label):
        distances = image1 - image2
        distances_squared = F.sum(F.square(distances), 1, keepdims=True)
        euclidean_distances = F.sqrt(distances_squared + 0.0001)
        d = F.clip(self.margin - euclidean_distances, 0, self.margin)
        loss = (1 - label) * distances_squared + label * F.square(d)
        loss = 0.5 * loss
        return loss
