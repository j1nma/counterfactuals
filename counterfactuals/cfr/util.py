import numpy as np
import tensorflow as tf

SQRT_CONST = 1e-10


def save_config(fname, FLAGS):
    """ Save configuration """
    flag_dictionary = FLAGS.__dict__
    s = '\n'.join(['%s: %s' % (k, str(flag_dictionary[k])) for k in sorted(flag_dictionary.keys())])
    f = open(fname, 'w')
    f.write(s)
    f.close()


def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * tf.matmul(X, tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D


def wasserstein(X, t, p, lam=10, its=10, sq=False, backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]
    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt, Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt, Xc))

    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M, 10 / (nc * nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam / M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta * tf.ones(tf.shape(M[0:1, :]))
    col = tf.concat(0, [delta * tf.ones(tf.shape(M[:, 0:1])), tf.zeros((1, 1))])
    Mt = tf.concat(0, [M, row])
    Mt = tf.concat(1, [Mt, col])

    ''' Compute marginal vectors '''
    a = tf.concat(0, [p * tf.ones(tf.shape(tf.where(t > 0)[:, 0:1])) / nt, (1 - p) * tf.ones((1, 1))])
    b = tf.concat(0, [(1 - p) * tf.ones(tf.shape(tf.where(t < 1)[:, 0:1])) / nc, p * tf.ones((1, 1))])

    ''' Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = tf.exp(-Mlam) + 1e-6  # added constant to avoid nan
    U = K * Mt
    ainvK = K / a

    u = a
    for i in range(0, its):
        u = 1.0 / (tf.matmul(ainvK, (b / tf.transpose(tf.matmul(tf.transpose(u), K)))))
    v = b / (tf.transpose(tf.matmul(tf.transpose(u), K)))

    T = u * (tf.transpose(v) * K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T * Mt
    D = 2 * tf.reduce_sum(E)

    return D, Mlam
