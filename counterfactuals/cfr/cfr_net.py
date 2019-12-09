import json

from counterfactuals.cfr.util import *


class cfr_net(object):
    """
    cfr_net implements the counterfactual regression neural network
    by F. Johansson, U. Shalit and D. Sontag: https://arxiv.org/abs/1606.03976

    This file contains the class cfr_net as well as helper functions.
    The network is implemented as a tensorflow graph. The class constructor
    creates an object containing relevant TF nodes as member variables.
    """

    def __init__(self, x, t, y_, p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims, mx_do_in=None, mx_do_out=None,
                 mx_x=None, mx_t=None, mx_y=None,
                 mx_p_t=None):
        self.variables = {}
        self.wd_loss = 0
        self.nonlin = tf.nn.relu

        # mx
        self.mx_variables = {}
        self.mx_wd_loss = 0
        self.mx_nonlin = mx.symbol.relu

        self._build_graph(x, t, y_, p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims, mx_do_in, mx_do_out, mx_x, mx_t,
                          mx_y, mx_p_t)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i)  # @TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.wd_loss += wd * tf.nn.l2_loss(var)
        return var

    def _build_graph(self, x, t, y_, p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims, mx_do_in, mx_do_out, mx_x,
                     mx_t, mx_y_, mx_p_t):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """

        self.x = x
        self.t = t
        self.y_ = y_
        self.p_t = p_t
        self.r_alpha = r_alpha
        self.r_lambda = r_lambda
        self.do_in = do_in
        self.do_out = do_out

        self.mx_x = mx_x
        self.mx_t = mx_t
        self.mx_y_ = mx_y_
        self.mx_p_t = mx_p_t
        self.mx_do_in = mx_do_in
        self.mx_do_out = mx_do_out

        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]

        weights_in = []
        biases_in = []

        # mx
        mx_weights_in = []
        mx_biases_in = []

        if FLAGS.rep_lay == 0:
            dim_in = dim_input
        if FLAGS.reg_lay == 0:
            if not FLAGS.split_output:
                dim_out = dim_in + 1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm:
            bn_biases = []
            bn_scales = []

        ''' Construct input/representation layers '''
        h_in = [x]

        # mx
        mx_h_in = [mx_x]

        for i in range(0, FLAGS.rep_lay):
            if i == 0:
                weights_in.append(
                    tf.Variable(
                        tf.random_normal([dim_input, dim_in], stddev=FLAGS.weight_init_scale / np.sqrt(dim_input))))
                # mx
                random_values = mx.symbol.Variable('random_values_weight')
                mx_weights_in.append(
                    mx.nd.random.normal(scale=FLAGS.weight_init_scale / np.sqrt(dim_input), shape=(dim_input, dim_in)))
            else:
                weights_in.append(
                    tf.Variable(tf.random_normal([dim_in, dim_in], stddev=FLAGS.weight_init_scale / np.sqrt(dim_in))))
                # mx
                random_values = mx.symbol.Variable('random_values_weight')
                mx_weights_in.append(
                    mx.nd.random.normal(scale=FLAGS.weight_init_scale / np.sqrt(dim_in), shape=(dim_in, dim_in)))

            biases_in.append(tf.Variable(tf.zeros([1, dim_in])))
            z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

            # mx
            dim_in_zeros = mx.symbol.Variable('dim_in_zeros')
            mx_biases_in.append(mx.nd.zeros(shape=(1, dim_in)))
            mx_z = mx.symbol.dot(mx_h_in[i], random_values) + dim_in_zeros

            if FLAGS.batch_norm:
                batch_mean, batch_var = tf.nn.moments(z, [0])

                if FLAGS.normalization == 'bn_fixed':
                    z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                else:
                    bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                    bn_scales.append(tf.Variable(tf.ones([dim_in])))
                    z = tf.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)

            h_in.append(self.nonlin(z))
            h_in[i + 1] = tf.nn.dropout(h_in[i + 1], do_in)

            # mx
            mx_h_in.append(self.mx_nonlin(mx_z))
            mx_h_in[i + 1] = mx.symbol.Dropout(mx_h_in[i + 1], p=1 - mx_do_in)  # TODO, again, watchout with keep prob

        h_rep = h_in[len(h_in) - 1]

        # mx
        mx_h_rep = mx_h_in[len(mx_h_in) - 1]

        if FLAGS.normalization == 'divide':
            h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))

            # mx
            mx_h_rep_norm = mx_h_rep / mx.symbol.sqrt(mx.symbol.sum(mx.symbol.square(mx_h_rep), axis=1, keepdims=True))
        else:
            h_rep_norm = 1.0 * h_rep

            # mx
            mx_h_rep_norm = 1.0 * mx_h_rep

        ''' Construct output layers '''
        y, weights_out, weights_pred = self._build_output_graph(h_rep_norm, t, dim_in, dim_out, do_out, FLAGS)

        # mx
        mx_y, mx_weights_out, mx_weights_pred = self.mx_build_output_graph(mx_h_rep_norm, mx_t, dim_in, dim_out,
                                                                           mx_do_out,
                                                                           FLAGS)

        ''' Compute sample reweighing '''
        if FLAGS.reweight_sample:
            w_t = t / (2 * p_t)
            w_c = (1 - t) / (2 * 1 - p_t)
            sample_weight = w_t + w_c

            # mx
            mx_w_t = mx_t / (2 * mx_p_t)
            mx_w_c = (1 - mx_t) / (2 * 1 - mx_p_t)
            mx_sample_weight = mx_w_t + mx_w_c
        else:
            sample_weight = 1.0

            # mx
            mx_sample_weight = 1.0

        self.sample_weight = sample_weight

        # mx
        self.mx_sample_weight = mx_sample_weight

        ''' Construct factual L2 loss function '''
        risk = tf.reduce_mean(sample_weight * tf.square(y_ - y))
        pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

        # mx
        mx_risk = mx.sym.mean(mx_sample_weight * mx.sym.square(mx_y_ - mx_y))
        mx_pred_error = mx.sym.sqrt(mx.sym.mean(mx.sym.square(mx_y_ - mx_y)))

        ''' Regularization '''
        if FLAGS.p_lambda > 0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.rep_lay):
                self.wd_loss += tf.nn.l2_loss(weights_in[i])

                # mx
                self.mx_wd_loss = self.mx_wd_loss + mx.symbol.LinearRegressionOutput(mx_weights_in[i])

        ''' Imbalance error '''
        p_ipm = 0.5

        imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                        sq=False, backpropT=FLAGS.wass_bpg)
        imb_error = r_alpha * imb_dist

        ''' Total error '''
        tot_error = risk

        # mx
        mx_tot_error = mx_risk

        if FLAGS.p_alpha > 0:
            tot_error = tot_error + imb_error

        if FLAGS.p_lambda > 0:
            tot_error = tot_error + r_lambda * self.wd_loss

        self.output = y
        self.tot_loss = tot_error
        self.imb_loss = imb_error
        self.imb_dist = imb_dist
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm

    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out] * FLAGS.reg_lay)

        weights_out = []
        biases_out = []

        for i in range(0, FLAGS.reg_lay):
            wo = self._create_variable_with_weight_decay(
                tf.random_normal([dims[i], dims[i + 1]],
                                 stddev=FLAGS.weight_init_scale / np.sqrt(dims[i])),
                'w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1, dim_out])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            h_out.append(self.nonlin(z))
            h_out[i + 1] = tf.nn.dropout(h_out[i + 1], do_out)

        weights_pred = self._create_variable(tf.random_normal([dim_out, 1],
                                                              stddev=FLAGS.weight_init_scale / np.sqrt(dim_out)),
                                             'w_pred')
        bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

        if FLAGS.reg_lay == 0:
            self.wd_loss += tf.nn.l2_loss(
                tf.slice(weights_pred, [0, 0], [dim_out - 1, 1]))  # don't penalize treatment coefficient
        else:
            self.wd_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.matmul(h_pred, weights_pred) + bias_pred

        return y, weights_out, weights_pred

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''

        if FLAGS.split_output:

            i0 = tf.to_int32(tf.where(t < 1)[:, 0])
            i1 = tf.to_int32(tf.where(t > 0)[:, 0])

            rep0 = tf.gather(rep, i0)
            rep1 = tf.gather(rep, i1)

            y0, weights_out0, weights_pred0 = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS)
            y1, weights_out1, weights_pred1 = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS)

            y = tf.dynamic_stitch([i0, i1], [y0, y1])
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
        else:
            h_input = tf.concat(1, [rep, t])
            y, weights_out, weights_pred = self._build_output(h_input, dim_in + 1, dim_out, do_out, FLAGS)

        return y, weights_out, weights_pred


class mx_cfr_net(object):
    """
    cfr_net implements the counterfactual regression neural network
    by F. Johansson, U. Shalit and D. Sontag: https://arxiv.org/abs/1606.03976

    This file contains the class cfr_net as well as helper functions.
    The network is implemented as a tensorflow graph. The class constructor
    creates an object containing relevant TF nodes as member variables.
    """

    def __init__(self, FLAGS, r_alpha, r_lambda, do_in, do_out, dims, mx_do_in=None, mx_do_out=None,
                 mx_x=None, mx_t=None, mx_y=None,
                 mx_p_t=None):

        self.mx_variables = {}
        self.mx_wd_loss = 0
        self.mx_nonlin = mx.symbol.relu

        self.mx_build_graph(FLAGS, r_alpha, r_lambda, do_in, do_out, dims, mx_do_in, mx_do_out, mx_x, mx_t,
                          mx_y, mx_p_t)

    def mx_add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.mx_variables:
            name = '%s_%d' % (basename, i)  # @TODO: not consistent with TF internally if changed
            i += 1

        self.mx_variables[name] = var

    def mx_create_variable(self, initializer, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = mx.sym.Variable(name=name, init=initializer)
        self.mx_add_variable(var, name)
        return var

    def mx_create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self.mx_create_variable(initializer, name)
        self.mx_wd_loss = self.mx_wd_loss + wd * mx.symbol.LinearRegressionOutput(var)

        return var

    def mx_build_graph(self, FLAGS, r_alpha, r_lambda, do_in, do_out, dims, mx_do_in, mx_do_out, mx_x,
                     mx_t, mx_y_, mx_p_t):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """

        self.r_alpha = r_alpha
        self.r_lambda = r_lambda
        self.do_in = do_in
        self.do_out = do_out

        self.x = mx_x
        self.t = mx_t
        self.y_ = mx_y_
        self.p_t = mx_p_t
        self.do_in = mx_do_in
        self.do_out = mx_do_out

        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]

        weights_in = []
        biases_in = []

        # mx
        mx_weights_in = []
        mx_biases_in = []

        if FLAGS.rep_lay == 0:
            dim_in = dim_input
        if FLAGS.reg_lay == 0:
            if not FLAGS.split_output:
                dim_out = dim_in + 1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm:
            bn_biases = []
            bn_scales = []

        ''' Construct input/representation layers '''
        # mx
        mx_h_in = [mx_x]

        for i in range(0, FLAGS.rep_lay):
            if i == 0:
                weights_in.append(
                    tf.Variable(
                        tf.random_normal([dim_input, dim_in], stddev=FLAGS.weight_init_scale / np.sqrt(dim_input))))
                # mx
                random_values = mx.symbol.Variable('random_values_weight')
                mx_weights_in.append(
                    mx.nd.random.normal(scale=FLAGS.weight_init_scale / np.sqrt(dim_input), shape=(dim_input, dim_in)))
            else:
                weights_in.append(
                    tf.Variable(tf.random_normal([dim_in, dim_in], stddev=FLAGS.weight_init_scale / np.sqrt(dim_in))))
                # mx
                random_values = mx.symbol.Variable('random_values_weight')
                mx_weights_in.append(
                    mx.nd.random.normal(scale=FLAGS.weight_init_scale / np.sqrt(dim_in), shape=(dim_in, dim_in)))

            # mx
            dim_in_zeros = mx.symbol.Variable('dim_in_zeros')
            mx_biases_in.append(mx.nd.zeros(shape=(1, dim_in)))
            mx_z = mx.symbol.dot(mx_h_in[i], random_values) + dim_in_zeros

            if FLAGS.batch_norm:
                batch_mean, batch_var = tf.nn.moments(z, [0])

                if FLAGS.normalization == 'bn_fixed':
                    z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                else:
                    bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                    bn_scales.append(tf.Variable(tf.ones([dim_in])))
                    z = tf.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)

            # mx
            mx_h_in.append(self.mx_nonlin(mx_z))
            mx_h_in[i + 1] = mx.symbol.Dropout(mx_h_in[i + 1], p=1 - mx_do_in)  # TODO, again, watchout with keep prob

        # mx
        mx_h_rep = mx_h_in[len(mx_h_in) - 1]

        if FLAGS.normalization == 'divide':
            # mx
            mx_h_rep_norm = mx_h_rep / mx.symbol.sqrt(mx.symbol.sum(mx.symbol.square(mx_h_rep), axis=1, keepdims=True))
        else:
            # mx
            mx_h_rep_norm = 1.0 * mx_h_rep

        ''' Construct output layers '''
        # mx
        mx_y, mx_weights_out, mx_weights_pred = self.mx_build_output_graph(mx_h_rep_norm, mx_t, dim_in, dim_out,
                                                                           mx_do_out,
                                                                           FLAGS)

        ''' Compute sample reweighing '''
        if FLAGS.reweight_sample:
            # mx
            mx_w_t = mx_t / (2 * mx_p_t)
            mx_w_c = (1 - mx_t) / (2 * 1 - mx_p_t)
            mx_sample_weight = mx_w_t + mx_w_c
        else:
            # mx
            mx_sample_weight = 1.0

        # mx
        self.mx_sample_weight = mx_sample_weight

        ''' Construct factual L2 loss function '''
        # mx
        mx_risk = mx.sym.mean(mx_sample_weight * mx.sym.square(mx_y_ - mx_y))
        mx_pred_error = mx.sym.sqrt(mx.sym.mean(mx.sym.square(mx_y_ - mx_y)))

        ''' Regularization '''
        if FLAGS.p_lambda > 0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.rep_lay):
                # mx
                self.mx_wd_loss = self.mx_wd_loss + mx.symbol.LinearRegressionOutput(mx_weights_in[i])

        ''' Imbalance error '''
        p_ipm = 0.5

        imb_dist, imb_mat = mx_wasserstein(mx_h_rep_norm, mx_t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                        sq=False, backpropT=FLAGS.wass_bpg)
        imb_error = r_alpha * imb_dist

        ''' Total error '''
        # mx
        mx_tot_error = mx_risk

        if FLAGS.p_alpha > 0:
            mx_tot_error = mx_tot_error + imb_error

        if FLAGS.p_lambda > 0:
            mx_tot_error = mx_tot_error + r_lambda * self.mx_wd_loss

        self.output = mx_y
        self.tot_loss = mx_tot_error
        self.imb_loss = imb_error
        self.imb_dist = imb_dist
        self.pred_loss = mx_pred_error
        self.weights_in = weights_in
        self.weights_out = mx_weights_out
        self.weights_pred = mx_weights_pred
        self.h_rep = mx_h_rep
        self.h_rep_norm = mx_h_rep_norm

    def mx_build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''

        if FLAGS.split_output:

            # i0 = mx.nd.cast(mx.sym.where(t < 1)[:, 0], dtype='int32')
            # i1 = mx.nd.cast(mx.sym.where(t > 0)[:, 0], dtype='int32')
            i0 = mx.sym.cast(mx.sym.where(t < 1), dtype='int32')
            i1 = mx.sym.cast(mx.sym.where(t > 0), dtype='int32')

            rep0 = mx.sym.gather_nd(rep, i0)
            rep1 = mx.sym.gather_nd(rep, i1)

            y0, weights_out0, weights_pred0 = self.mx_build_output(rep0, dim_in, dim_out, do_out, FLAGS)
            y1, weights_out1, weights_pred1 = self.mx_build_output(rep1, dim_in, dim_out, do_out, FLAGS)

            # y = tf.dynamic_stitch([i0, i1], [y0, y1])
            y = mx.sym.gather_nd(data=mx.sym.stack(y0, y1), indices=mx.sym.stack(i0, i1))
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
        else:
            h_input = tf.concat(1, [rep, t])
            y, weights_out, weights_pred = self._build_output(h_input, dim_in + 1, dim_out, do_out, FLAGS)

        return y, weights_out, weights_pred

    def mx_build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out] * FLAGS.reg_lay)

        weights_out = []
        biases_out = []

        for i in range(0, FLAGS.reg_lay):
            # initializer = mx.initializer.random.normal(scale=FLAGS.weight_init_scale / np.sqrt(dims[i]),
            #                                            shape=(dims[i], dims[i + 1]))
            initializer = mx.init.Normal(
                FLAGS.weight_init_scale / np.sqrt(dims[i]))  # TODO what about shape from above?

            wo = self.mx_create_variable_with_weight_decay(
                initializer,
                'w_out_%d' % i, 1.0)

            weights_out.append(wo)

            json_zeroes = json.dumps({"biases_out": mx.nd.zeros((1, dim_out)).asnumpy().tolist()})
            biases_out.append(mx.sym.Variable(name="biases_out" + str(i), init=mx.init.Constant(json_zeroes)))
            z = mx.symbol.dot(h_out[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            h_out.append(self.mx_nonlin(z))
            h_out[i + 1] = mx.symbol.Dropout(h_out[i + 1], p=1 - do_out)  # TODO, again, watchout with keep prob

        # initializer = mx.initializer.random.normal(scale=FLAGS.weight_init_scale / np.sqrt(dim_out), shape=(dim_out, 1))
        initializer = mx.init.Normal(FLAGS.weight_init_scale / np.sqrt(dim_out))  # TODO what about shape from above?
        weights_pred = self.mx_create_variable(initializer, 'w_pred')

        # bias_pred = self.mx_create_variable(tf.zeros([1]), 'b_pred')
        json_zeroes = json.dumps({"biases_out": mx.nd.zeros((1)).asnumpy().tolist()})
        bias_pred = self.mx_create_variable(mx.init.Constant(json_zeroes), 'b_pred')

        if FLAGS.reg_lay == 0:
            # self.mx_wd_loss += tf.nn.l2_loss(tf.slice(weights_pred, [0, 0], [dim_out - 1, 1]))  # don't penalize treatment coefficient
            # l2_loss = mx.gluon.loss.L2Loss()
            # TODO watchout tf size vs mx end
            # TODO watchout tf.nn.l2_loss only computes tensor
            self.mx_wd_loss += tf.nn.l2_loss(mx.symbol.slice(weights_pred,
                                                             begin=(0, 0),
                                                             end=(dim_out - 1, 1)))
        else:
            self.mx_wd_loss = self.mx_wd_loss + mx.symbol.LinearRegressionOutput(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = mx.sym.dot(h_pred, weights_pred) + bias_pred

        return y, weights_out, weights_pred
