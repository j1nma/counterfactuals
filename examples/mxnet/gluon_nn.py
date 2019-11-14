# Create a neural network

from mxnet import nd
from mxnet.gluon import nn

# Create your neural network’s first layer
# Let’s start with a dense layer with 2 output units.

layer = nn.Dense(2)
# print(layer)

# Then initialize its weights with the default initialization method, which draws random values uniformly
# from [−0.7, 0.7].
layer.initialize()

# print(layer)

# Then we do a forward pass with random data. We create a  (3,4)
#   shape random input x and feed into the layer to compute the output.
x = nd.random.uniform(-1, 1, (3, 4))
layer(x)
# print(layer(x))
# print(layer)

#  As can be seen, the layer’s input limit of 2 produced a  (3,2)
#   shape output from our  (3,4)
#   input. Note that we didn’t specify the input size of layer before
#   (though we can specify it with the argument in_units=4 here),
#   the system will automatically infer it during the first time we feed in data, create
#   and initialize the weights. So we can access the weight after the first forward pass:
layer.weight.data()
# print(layer.weight.data())

# Chain layers into a neural network¶
#
# Let’s first consider a simple case that a neural network is a chain of layers.
# During the forward pass, we run layers sequentially one-by-one.
# The following code implements a famous network called LeNet through nn.Sequential.

net = nn.Sequential()
# Add a sequence of layers.
net.add(  # Similar to Dense, it is not necessary to specify the input channels
    # by the argument `in_channels`, which will be  automatically inferred
    # in the first forward pass. Also, we apply a relu activation on the
    # output. In addition, we can use a tuple to specify a  non-square
    # kernel size, such as `kernel_size=(2,4)`
    nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
    # One can also use a tuple to specify non-symmetric pool and stride sizes
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
    nn.MaxPool2D(pool_size=2, strides=2),
    # The dense layer will automatically reshape the 4-D output of last
    # max pooling layer into the 2-D shape: (x.shape[0], x.size/x.shape[0])
    nn.Dense(120, activation="relu"),
    nn.Dense(84, activation="relu"),
    nn.Dense(10))
# print(net)

# The usage of nn.Sequential is similar to nn.Dense. In fact, both of them are subclasses of nn.Block.
# The following codes show how to initialize the weights and run the forward pass.
net.initialize()
# Input shape is (batch_size, color_channels, height, width)
x = nd.random.uniform(shape=(4, 1, 28, 28))
y = net(x)


# print(y.shape)

# We can use [] to index a particular layer.
# For example, the following accesses the 1st layer’s weight and 6th layer’s bias.
# print(net[0].weight.data().shape, net[5].bias.data().shape)


# Create a neural network flexibly
class MixMLP(nn.Block):
    def __init__(self, **kwargs):
        # Run `nn.Block`'s init method
        super(MixMLP, self).__init__(**kwargs)
        self.blk = nn.Sequential()
        self.blk.add(nn.Dense(3, activation='relu'),
                     nn.Dense(4, activation='relu'))
        self.dense = nn.Dense(5)

    def forward(self, x):
        y = nd.relu(self.blk(x))
        print(y)
        return self.dense(y)


net = MixMLP()
net.initialize()
x = nd.random.uniform(shape=(2, 2))
h = net(x)

print(h)

# print(net)
