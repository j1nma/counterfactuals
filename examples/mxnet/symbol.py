import mxnet as mx
import numpy as np
from mxnet import gluon, autograd, nd

np.random.seed(42)
mx.random.seed(42)
# ctx = mx.gpu()
ctx = mx.cpu()

def data_xform(data):
    """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
    return nd.moveaxis(data, 2, 0).astype('float32') / 255


# prepare data
train_data = mx.gluon.data.vision.MNIST(train=True).transform_first(data_xform)
val_data = mx.gluon.data.vision.MNIST(train=False).transform_first(data_xform)
batch_size = 100
train_loader = mx.gluon.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = mx.gluon.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)

# create network
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data=act2, name='fc3', num_hidden=10)

net = gluon.SymbolBlock(outputs=[fc3], inputs=[data])
net.initialize(ctx=ctx)

# create trainer, metric
trainer = gluon.Trainer(
    params=net.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.1, 'momentum': 0.9, 'wd': 0.00001},
)
metric = mx.metric.Accuracy()

# learn
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs = inputs.as_in_context(ctx)
        labels = labels.as_in_context(ctx)

        with autograd.record():
            outputs = net(inputs)
            # softmax
            # exps = nd.exp(outputs)
            # exps = exps / exps.sum(axis=1).reshape((-1, 1))
            # cross entropy
            # loss = nd.MakeLoss(-nd.log(exps.pick(labels)))
            #
            loss = gluon.loss.SoftmaxCrossEntropyLoss()(outputs, labels)
            # print(loss)

        loss.backward()
        metric.update(labels, outputs)

        trainer.step(batch_size=inputs.shape[0])

    name, acc = metric.get()
    print('After epoch {}: {} = {}'.format(epoch + 1, name, acc))
    metric.reset()
