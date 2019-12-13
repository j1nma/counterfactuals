import numpy as np
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.loss import Loss


class CFRNet(nn.HybridBlock):
    def __init__(self, rep_hidden_size, hyp_hidden_size, **kwards):
        nn.HybridBlock.__init__(self, **kwards)

        self.input_shape = None

        with self.name_scope():
            # Representation Layers
            self.rep_fc1 = nn.Dense(rep_hidden_size, activation='relu')
            self.rep_fc2 = nn.Dense(rep_hidden_size, activation='relu')
            self.rep_fc3 = nn.Dense(rep_hidden_size, activation='relu')

            # Hypothesis Layers for t = 1
            self.t1_hyp_fc1 = nn.Dense(hyp_hidden_size, activation='relu')
            self.t1_hyp_fc2 = nn.Dense(hyp_hidden_size, activation='relu')
            self.t1_hyp_fc3 = nn.Dense(hyp_hidden_size, activation='relu')
            self.t1_hyp_fc4 = nn.Dense(1)

            # Hypothesis Layers for t = 0
            self.t0_hyp_fc1 = nn.Dense(hyp_hidden_size, activation='relu')
            self.t0_hyp_fc2 = nn.Dense(hyp_hidden_size, activation='relu')
            self.t0_hyp_fc3 = nn.Dense(hyp_hidden_size, activation='relu')
            self.t0_hyp_fc4 = nn.Dense(1)

    def forward(self, x, t):
        self.input_shape = x.shape

        return HybridBlock.forward(self, x, t)

    def hybrid_forward(self, F, x, t):
        rep_relu1 = self.rep_fc1(x)
        rep_relu2 = self.rep_fc2(rep_relu1)
        rep_relu3 = self.rep_fc3(rep_relu2)

        t1_hyp_relu1 = self.t1_hyp_fc1(rep_relu3[np.where(t == 1)[0]])
        t1_hyp_relu2 = self.t1_hyp_fc2(t1_hyp_relu1)
        t1_hyp_relu3 = self.t1_hyp_fc3(t1_hyp_relu2)
        t1_hyp_relu4 = self.t1_hyp_fc4(t1_hyp_relu3)

        t0_hyp_relu1 = self.t0_hyp_fc1(rep_relu3[np.where(t == 0)[0]])
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
