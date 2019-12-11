from counterfactuals import utilities
from counterfactuals.cfr.loader import *


class NaNException(Exception):
    pass


def pdist2(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * X.dot(Y.T)
    nx = np.sum(np.square(X), 1, keepdims=True)
    ny = np.sum(np.square(Y), 1, keepdims=True)
    D = (C + ny.T) + nx

    return np.sqrt(D + 1e-8)


def cf_nn(x, t):
    It = np.array(np.where(t == 1))[0, :]
    Ic = np.array(np.where(t == 0))[0, :]

    x_c = x[Ic, :]
    x_t = x[It, :]

    D = pdist2(x_c, x_t)

    nn_t = Ic[np.argmin(D, 0)]
    nn_c = It[np.argmin(D, 1)]

    return nn_t, nn_c


def pehe_nn(yf_p, ycf_p, y, x, t, nn_t=None, nn_c=None):
    if nn_t is None or nn_c is None:
        nn_t, nn_c = cf_nn(x, t)

    It = np.array(np.where(t == 1))[0, :]
    Ic = np.array(np.where(t == 0))[0, :]

    ycf_t = 1.0 * y[nn_t]
    eff_nn_t = ycf_t - 1.0 * y[It]
    eff_pred_t = ycf_p[It] - yf_p[It]

    eff_pred = eff_pred_t
    eff_nn = eff_nn_t

    '''
    ycf_c = 1.0*y[nn_c]
    eff_nn_c = ycf_c - 1.0*y[Ic]
    eff_pred_c = ycf_p[Ic] - yf_p[Ic]

    eff_pred = np.vstack((eff_pred_t, eff_pred_c))
    eff_nn = np.vstack((eff_nn_t, eff_nn_c))
    '''

    pehe_nn = np.sqrt(np.mean(np.square(eff_pred - eff_nn)))

    return pehe_nn


def evaluate_ate(predictions, data, i_exp, I_subset=None, nn_t=None, nn_c=None):
    x = data['x'][:, :, i_exp]
    t = data['t'][:, i_exp]
    yf = data['yf'][:, i_exp]
    ycf = data['ycf'][:, i_exp]
    mu0 = data['mu0'][:, i_exp]
    mu1 = data['mu1'][:, i_exp]
    yf_p = predictions[:, 0]
    ycf_p = predictions[:, 1]

    if not I_subset is None:
        x = x[I_subset,]
        t = t[I_subset]
        yf_p = yf_p[I_subset]
        ycf_p = ycf_p[I_subset]
        yf = yf[I_subset]
        ycf = ycf[I_subset]
        mu0 = mu0[I_subset]
        mu1 = mu1[I_subset]

    eff = mu1 - mu0

    rmse_fact = np.sqrt(np.mean(np.square(yf_p - yf)))
    rmse_cfact = np.sqrt(np.mean(np.square(ycf_p - ycf)))

    eff_pred = ycf_p - yf_p
    eff_pred[t > 0] = -eff_pred[t > 0]

    ite_pred = ycf_p - yf
    ite_pred[t > 0] = -ite_pred[t > 0]
    rmse_ite = np.sqrt(np.mean(np.square(ite_pred - eff)))

    ate_pred = np.mean(eff_pred)
    bias_ate = ate_pred - np.mean(eff)

    att_pred = np.mean(eff_pred[t > 0])
    bias_att = att_pred - np.mean(eff[t > 0])

    atc_pred = np.mean(eff_pred[t < 1])
    bias_atc = atc_pred - np.mean(eff[t < 1])

    pehe = np.sqrt(np.mean(np.square(eff_pred - eff)))

    pehe_appr = pehe_nn(yf_p, ycf_p, yf, x, t, nn_t, nn_c)

    return {'ate_pred': ate_pred, 'att_pred': att_pred,
            'atc_pred': atc_pred, 'bias_ate': bias_ate,
            'bias_att': bias_att, 'bias_atc': bias_atc,
            'rmse_fact': rmse_fact, 'rmse_cfact': rmse_cfact,
            'pehe': pehe, 'rmse_ite': rmse_ite, 'pehe_nn': pehe_appr}


def evaluate_result(result, data, validation=False, multiple_exps=False):
    predictions = result['pred']

    if validation:
        I_valid = result['val']

    n_units, _, n_rep, n_outputs = predictions.shape

    eval_results = []
    # Loop over output_times
    for i_out in range(n_outputs):
        eval_results_out = []

        if not multiple_exps and not validation:
            nn_t, nn_c = cf_nn(data['x'][:, :, 0], data['t'][:, 0])

        # Loop over repeated experiments
        for i_rep in range(n_rep):

            if validation:
                I_valid_rep = I_valid[i_rep, :]
            else:
                I_valid_rep = None

            if multiple_exps:
                i_exp = i_rep
                if validation:
                    nn_t, nn_c = cf_nn(data['x'][I_valid_rep, :, i_exp], data['t'][I_valid_rep, i_exp])
                else:
                    nn_t, nn_c = cf_nn(data['x'][:, :, i_exp], data['t'][:, i_exp])
            else:
                i_exp = 0

            if validation and not multiple_exps:
                nn_t, nn_c = cf_nn(data['x'][I_valid_rep, :, i_exp], data['t'][I_valid_rep, i_exp])

            eval_result = evaluate_ate(predictions[:, :, i_rep, i_out], data, i_exp, I_valid_rep, nn_t=nn_t,
                                       nn_c=nn_c)

            eval_results_out.append(eval_result)

        eval_results.append(eval_results_out)

    # Reformat into dict
    eval_dict = {}
    keys = eval_results[0][0].keys()
    for k in keys:
        v = np.array([[eval_results[i][j][k] for i in range(n_outputs)] for j in range(n_rep)])
        eval_dict[k] = v

    # Gather loss
    # Shape [times, types, reps]
    # Types: obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj
    if 'loss' in result.keys() and result['loss'].shape[1] >= 6:
        losses = result['loss']
        n_loss_outputs = losses.shape[0]

        if validation:
            objective = np.array([losses[int((n_loss_outputs * i) / n_outputs), 6, :] for i in range(n_outputs)]).T
        else:
            objective = np.array([losses[int((n_loss_outputs * i) / n_outputs), 0, :] for i in range(n_outputs)]).T

        eval_dict['objective'] = objective

    return eval_dict


def evaluate(output_dir, data_path_train, data_path_test=None):
    print('\nEvaluating experiment %s...' % output_dir)

    # Load results for all configurations
    results = load_results(output_dir)

    if len(results) == 0:
        raise Exception('No finished results found.')

    # Separate configuration files
    configs = [r['config'] for r in results]

    # Test whether multiple experiments (different data)
    multiple_exps = (configs[0]['experiments'] > 1)
    if multiple_exps:
        print('Multiple data (experiments) detected')

    # Load training data
    print('Loading TRAINING data %s...' % data_path_train)
    data_train = utilities.load_data(data_path_train)

    # Load test data
    if data_path_test is not None:
        print('Loading TEST data %s...' % data_path_test)
        data_test = utilities.load_data(data_path_test)
    else:
        data_test = None

    # Evaluate all results
    eval_results = []
    configs_out = []
    i = 0
    print('Evaluating result (out of %d): ' % len(results))
    for result in results:
        print('Evaluating %d...' % (i + 1))

        try:
            eval_train = evaluate_result(result['train'], data_train, validation=False, multiple_exps=multiple_exps)

            eval_valid = evaluate_result(result['train'], data_train, validation=True, multiple_exps=multiple_exps)

            if data_test is not None:
                eval_test = evaluate_result(result['test'], data_test, validation=False, multiple_exps=multiple_exps)
            else:
                eval_test = None

            eval_results.append({'train': eval_train, 'valid': eval_valid, 'test': eval_test})
            configs_out.append(configs[i])
        except NaNException as e:
            print('WARNING: Encountered NaN exception. Skipping.')
            print(e)

        i += 1

    # Reformat into dict
    eval_dict = {'train': {}, 'test': {}, 'valid': {}}
    keys = eval_results[0]['train'].keys()
    for k in keys:
        v = np.array([eval_results[i]['train'][k] for i in range(len(eval_results))])
        eval_dict['train'][k] = v

        v = np.array([eval_results[i]['valid'][k] for i in range(len(eval_results))])
        eval_dict['valid'][k] = v

        if eval_test is not None and k in eval_results[0]['test']:
            v = np.array([eval_results[i]['test'][k] for i in range(len(eval_results))])
            eval_dict['test'][k] = v

    return eval_dict, configs_out
