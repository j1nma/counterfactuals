from matplotlib import pyplot as plt
from mxnet import image
from mxnet.gluon import data as gdata, utils

utils.download('https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/cat.jpg')
img = image.imread('cat.jpg')
plt.imshow(img.asnumpy())
plt.show()


def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


def apply(img, aug, num_rows=2, num_cols=4, scale=3):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)

apply(img, gdata.vision.transforms.RandomFlipLeftRight())
apply(img, gdata.vision.transforms.RandomFlipTopBottom())

plt.show()

