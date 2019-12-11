import os

import numpy as np


def load_result_file(file):
    arr = np.load(file, allow_pickle=True)

    D = dict([(k, arr[k]) for k in arr.keys()])

    return D


def load_config(cfgfile):
    """ Parses a configuration file """

    cfgf = open(cfgfile, 'r')
    cfg = {}
    for l in cfgf:
        ps = [p.strip() for p in l.split(':')]
        if len(ps) == 2:
            try:
                cfg[ps[0]] = float(ps[1])
            except ValueError:
                cfg[ps[0]] = ps[1]
                if cfg[ps[0]] == 'False':
                    cfg[ps[0]] = False
                elif cfg[ps[0]] == 'True':
                    cfg[ps[0]] = True
    cfgf.close()
    return cfg


def load_single_result(result_dir):
    print('Loading %s...' % result_dir)

    config_path = '%s/config.txt' % result_dir
    has_config = os.path.isfile(config_path)
    if not has_config:
        print('WARNING: Could not find config.txt for %s. Skipping.' % os.path.basename(result_dir))
        config = None
    else:
        config = load_config(config_path)

    train_path = '%s/result.npz' % result_dir
    test_path = '%s/result.test.npz' % result_dir

    has_test = os.path.isfile(test_path)

    try:
        train_results = load_result_file(train_path)
    except:
        'WARNING: Could not load result file. Skipping'
        return None

    n_exp = config['experiments']

    if len(train_results['pred'].shape) < 4 or train_results['pred'].shape[2] < n_exp:
        print('WARNING: Experiment %s appears not to have finished. Skipping.' % result_dir)
        return None

    if has_test:
        test_results = load_result_file(test_path)
    else:
        test_results = None

    return {'train': train_results, 'test': test_results, 'config': config}


def load_results(output_dir):
    print('Loading results from %s...' % output_dir)

    ''' Detect results structure '''
    # Single result
    if os.path.isfile('%s/results.npz' % output_dir):
        # @TODO: Implement
        pass

    # Multiple results
    files = ['%s/%s' % (output_dir, f) for f in os.listdir(output_dir)]
    exp_dirs = [f for f in files if os.path.isdir(f)
                if os.path.isfile('%s/result.npz' % f)]

    print('Found %d experiment configurations.' % len(exp_dirs))

    # Load each result folder
    results = []
    for dir in exp_dirs:
        dir_result = load_single_result(dir)
        if dir_result is not None:
            results.append(dir_result)

    return results
