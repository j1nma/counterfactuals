import pickle

import counterfactuals.cfr.evaluation as evaluation
from counterfactuals.cfr.plotting import *


def load_config(config_file):
    with open(config_file, 'r') as f:
        cfg = [l.split('=') for l in f.read().split('\n') if '=' in l]
        cfg = dict([(kv[0], eval(kv[1])) for kv in cfg])
    return cfg


def evaluate(config_file, overwrite=False, filters=None):
    if not os.path.isfile(config_file):
        raise Exception('Could not find config file at path: %s' % config_file)

    cfg = load_config(config_file)
    output_dir = cfg['outdir']

    if not os.path.isdir(output_dir):
        raise Exception('Could not find output at path: %s' % output_dir)

    data_train = cfg['data_dir'] + cfg['data_train']
    data_test = cfg['data_dir'] + cfg['data_test']

    # Evaluate results
    eval_path = '%s/evaluation.npz' % output_dir
    if overwrite or (not os.path.isfile(eval_path)):
        eval_results, configs = evaluation.evaluate(output_dir,
                                                    data_path_train=data_train,
                                                    data_path_test=data_test)
        # Save evaluation
        pickle.dump((eval_results, configs), open(eval_path, "wb"))
    else:
        # Load evaluation
        print('Loading evaluation results from %s...' % eval_path)
        eval_results, configs = pickle.load(open(eval_path, "rb"))

    # Print evaluation results
    plot_evaluation(eval_results, configs, output_dir, data_train, data_test, filters)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python evaluate.py <config_file> <overwrite (default 0)> <filters (optional)>')
    else:
        config_file = sys.argv[1]

        overwrite = False
        if len(sys.argv) > 2 and sys.argv[2] == '1':
            overwrite = True

        filters = None
        if len(sys.argv) > 3:
            filters = eval(sys.argv[3])

        evaluate(config_file, overwrite, filters=filters)
