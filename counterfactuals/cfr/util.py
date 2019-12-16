import mxnet as mx
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

    # mx
    # return mx.symbol.sqrt(mx.symbol.clip(x, lbound, np.inf))

    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))


def mx_safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''

    # mx
    return mx.symbol.sqrt(mx.symbol.clip(x, lbound, np.inf))


def np_safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''

    return mx.nd.sqrt(mx.nd.clip(x, lbound, np.inf))


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * tf.matmul(X, tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keep_dims=True)
    D = (C + tf.transpose(ny)) + nx

    # mx
    # mx_C = -2 * mx.sym.dot(X, mx.symbol.transpose(Y))
    # mx_nx = mx.symbol.sum(mx.symbol.square(X), 1, keep_dims=True)
    # mx_ny = mx.symbol.sum(mx.symbol.square(Y), 1, keep_dims=True)
    # mx_D = (mx_C + mx.symbol.transpose(mx_ny)) + mx_nx

    return D


def mx_pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    # mx
    mx_C = -2 * mx.sym.dot(X, mx.symbol.transpose(Y))
    mx_nx = mx.symbol.sum(mx.symbol.square(X), 1, keepdims=True)
    mx_ny = mx.symbol.sum(mx.symbol.square(Y), 1, keepdims=True)
    mx_D = (mx_C + mx.symbol.transpose(mx_ny)) + mx_nx

    return mx_D


def np_pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * mx.nd.dot(X, mx.nd.transpose(Y))
    nx = mx.nd.sum(mx.nd.square(X), 1, keepdims=True)
    ny = mx.nd.sum(mx.nd.square(Y), 1, keepdims=True)
    D = (C + mx.nd.transpose(ny)) + nx

    return D


def wasserstein(X, t, p, lam=10, its=10, sq=False, backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]
    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    # mx
    # mx_it = mx.nd.where(t > 0)[:, 0]
    # mx_ic = mx.nd.where(t < 1)[:, 0]
    # mx_Xc = mx.nd.gather_nd(X, mx_ic)
    # mx_Xt = mx.nd.gather_nd(X, mx_it)
    # mx_nc = mx.nd.cast(mx_Xc.shape[0], dtype='float32')
    # mx_nt = mx.nd.cast(mx_Xt.shape[0], dtype='float32')

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

    # mx
    # mx_M_mean = mx.sym.mean(M)
    # mx_M_drop = mx.symbol.Dropout(M, 1 - (10 / (nc * nt))) # TODO watchout, added 1 - x because it is not keep as tf
    # mx_delta = mx.nd.stop_gradient(mx.symbol.max(M))
    # mx_eff_lam = mx.nd.stop_gradient(lam / mx_M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta * tf.ones(tf.shape(M[0:1, :]))
    col = tf.concat(0, [delta * tf.ones(tf.shape(M[:, 0:1])), tf.zeros((1, 1))])
    Mt = tf.concat(0, [M, row])
    Mt = tf.concat(1, [Mt, col])

    # mx
    # mx_Mt = M
    # mx_row = mx_delta * mx.nd.ones(M[0:1, :].shape())
    # mx_col = mx.symbol.Concat(mx_delta * tf.ones(tf.shape(M[:, 0:1])), tf.zeros((1, 1)), dim=0)
    # mx_Mt = mx.symbol.Concat(M, row, dim=0)
    # mx_Mt = mx.symbol.Concat(mx_Mt, col, dim=1)

    ''' Compute marginal vectors '''
    a = tf.concat(0, [p * tf.ones(tf.shape(tf.where(t > 0)[:, 0:1])) / nt, (1 - p) * tf.ones((1, 1))])
    b = tf.concat(0, [(1 - p) * tf.ones(tf.shape(tf.where(t < 1)[:, 0:1])) / nc, p * tf.ones((1, 1))])

    # mx
    # mx_a = mx.symbol.Concat(p * mx.nd.ones(mx.nd.where(t > 0)[:, 0:1].shape()) / nt, (1 - p) * mx.nd.ones((1, 1)), dim=0)
    # mx_b = mx.symbol.Concat((1 - p) * mx.nd.ones(mx.nd.where(t < 1)[:, 0:1].shape()) / nc, p * mx.nd.ones((1, 1)), dim=0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = tf.exp(-Mlam) + 1e-6  # added constant to avoid nan
    U = K * Mt
    ainvK = K / a

    # mx
    # mx_Mlam = mx_eff_lam * mx_Mt
    # mx_K = mx.symbol.exp(-mx_Mlam) + 1e-6  # added constant to avoid nan
    # mx_U = mx_K * mx_Mt
    # mx_ainvK = mx_K / mx_a

    u = a
    for i in range(0, its):
        u = 1.0 / (tf.matmul(ainvK, (b / tf.transpose(tf.matmul(tf.transpose(u), K)))))
    v = b / (tf.transpose(tf.matmul(tf.transpose(u), K)))

    # mx
    # mx_u = mx_a
    # for i in range(0, its):
    #     mx_u = 1.0 / (mx.symbol.dot(mx_ainvK, (mx_b / mx.symbol.transpose(mx.symbol.dot(mx.symbol.transpose(mx_u), mx_K)))))
    # mx_v = mx_b / (mx.symbol.transpose(mx.symbol.dot(mx.symbol.transpose(mx_u), mx_K)))

    T = u * (tf.transpose(v) * K)
    # mx
    # mx_T = mx_u * (mx.symbol.transpose(mx_v) * mx_K)

    if not backpropT:
        T = tf.stop_gradient(T)
        # mx
        # mx_T = mx.nd.stop_gradient(mx_T)

    E = T * Mt
    D = 2 * tf.reduce_sum(E)

    # mx
    # mx_E = mx_T * mx_Mt
    # mx_D = 2 * mx.symbol.sum(mx_E)

    # mx
    # return mx_D, mx_Mlam

    return D, Mlam


def mx_wasserstein(X, t, p, lam=10, its=10, sq=False, backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    # i0 = mx.sym.cast(mx.sym.where(t < 1), dtype='int32')

    # mx
    mx_it = mx.sym.where(t > 0)  # [ first_row:last_row , column_0 ]
    mx_ic = mx.sym.where(t < 1)
    mx_Xc = mx.sym.gather_nd(X, mx_ic)
    mx_Xt = mx.sym.gather_nd(X, mx_it)
    mx_nc = mx.sym.cast(mx_Xc, dtype='float32')
    mx_nt = mx.sym.cast(mx_Xt, dtype='float32')

    ''' Compute distance matrix'''
    if sq:
        M = mx_pdist2sq(mx_Xt, mx_Xc)
    else:
        M = mx_safe_sqrt(mx_pdist2sq(mx_Xt, mx_Xc))

    ''' Estimate lambda and delta '''
    # mx
    mx_M_mean = mx.sym.mean(M)
    mx_delta = mx.sym.stop_gradient(mx.symbol.max(M))
    mx_eff_lam = mx.sym.stop_gradient(lam / mx_M_mean)

    ''' Compute new distance matrix '''
    # mx
    mx_Mt = M
    mx_row = mx_delta * mx.sym.ones_like(mx.sym.slice_axis(M, axis=0, begin=0, end=1))
    mx_col = mx.symbol.Concat(mx_delta * mx.sym.ones_like(mx.sym.slice_axis(M, axis=1, begin=0, end=1)),
                              mx.sym.zeros((1, 1)), dim=0)
    mx_Mt = mx.symbol.Concat(M, mx_row, dim=0)
    mx_Mt = mx.symbol.Concat(mx_Mt, mx_col, dim=1)

    ''' Compute marginal vectors '''
    # mx

    mx_a = mx.symbol.Concat(
        p * mx.sym.ones_like(mx.sym.slice_axis(mx.sym.where(t > 0), axis=1, begin=0, end=1)) / mx_nt,
        (1 - p) * mx.sym.ones((1, 1)),
        dim=0)

    mx_b = mx.symbol.Concat(
        (1 - p) * mx.sym.ones_like(mx.sym.slice_axis(mx.sym.where(t < 1), axis=1, begin=0, end=1)) / mx_nc,
        p * mx.sym.ones((1, 1)),
        dim=0)

    ''' Compute kernel matrix'''
    # mx
    mx_Mlam = mx_eff_lam * mx_Mt
    mx_K = mx.symbol.exp(-mx_Mlam) + 1e-6  # added constant to avoid nan
    mx_U = mx_K * mx_Mt
    mx_ainvK = mx_K / mx_a

    # mx
    mx_u = mx_a
    for i in range(0, its):
        mx_u = 1.0 / (
            mx.symbol.dot(mx_ainvK, (mx_b / mx.symbol.transpose(mx.symbol.dot(mx.symbol.transpose(mx_u), mx_K)))))
    mx_v = mx_b / (mx.symbol.transpose(mx.symbol.dot(mx.symbol.transpose(mx_u), mx_K)))

    # mx
    mx_T = mx_u * (mx.symbol.transpose(mx_v) * mx_K)

    if not backpropT:
        # mx
        mx_T = mx.nd.stop_gradient(mx_T)

    # mx
    mx_E = mx_T * mx_Mt
    mx_D = 2 * mx.symbol.sum(mx_E)

    # mx
    return mx_D, mx_Mlam


def np_wasserstein(X, t, p, lam=10, its=10, sq=False, backpropT=False):  # TODO
    """ Returns the Wasserstein distance between treatment groups, via numpy operations. """

    it = np.where(t[:] == 1)[0]
    ic = np.where(t[:] == 0)[0]
    Xt = X[it]
    Xc = X[ic]
    nt = np.float(Xt.shape[0])
    nc = np.float(Xc.shape[0])

    ''' Compute distance matrix (opposite to clinicalml) '''
    if sq:
        M = np_safe_sqrt(np_pdist2sq(Xt, Xc))
    else:
        M = np_pdist2sq(Xt, Xc)

    ''' Estimate lambda and delta '''
    M_mean = mx.nd.mean(M)
    delta = mx.nd.stop_gradient(mx.nd.max(M))
    eff_lam = mx.nd.stop_gradient(lam / M_mean)

    ''' Compute new distance matrix '''
    row = delta * mx.nd.ones_like(mx.nd.slice_axis(M, axis=0, begin=0, end=1))
    col = mx.nd.Concat(delta * mx.nd.ones_like(mx.nd.slice_axis(M, axis=1, begin=0, end=1)),
                       mx.nd.zeros((1, 1)), dim=0)
    Mt = mx.nd.Concat(M, row, dim=0)
    Mt = mx.nd.Concat(Mt, col, dim=1)

    ''' Compute marginal vectors '''
    a = mx.nd.Concat(
        p * mx.nd.ones_like(mx.nd.slice_axis(Xt, axis=1, begin=0, end=1)) / nt,
        (1 - p) * mx.nd.ones((1, 1)),
        dim=0)

    b = mx.nd.Concat(
        (1 - p) * mx.nd.ones_like(mx.nd.slice_axis(Xc, axis=1, begin=0, end=1)) / nc,
        p * mx.nd.ones((1, 1)),
        dim=0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = mx.nd.exp(-Mlam) + 1e-6  # added constant to avoid nan
    ainvK = K / a

    u = a
    for i in range(0, its):
        u = 1.0 / (
            mx.nd.dot(ainvK, (b / mx.nd.transpose(mx.nd.dot(mx.nd.transpose(u), K)))))
    v = b / (mx.nd.transpose(mx.nd.dot(mx.nd.transpose(u), K)))

    T = u * (mx.nd.transpose(v) * K)

    if not backpropT:
        T = mx.nd.stop_gradient(T)

    E = T * Mt
    D = 2 * mx.nd.sum(E)

    return D, Mlam
