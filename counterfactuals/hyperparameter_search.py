# This code is based from this repository:
# https://github.com/clinicalml/cfrnet
# for the purpose of parsing configuration files and running tests
# with different hyperparameter configurations

import os
import sys
from subprocess import call

import numpy as np


def load_config(config_file):
    config = {}

    with open(config_file, 'r') as file:

        for line in file:
            line = line.strip()

            if len(line) > 0:
                values = line.split('=')

                if len(values) > 0 and not line[0] == '#':
                    key, value = (values[0], eval(values[1]))

                    if not isinstance(value, list):
                        value = [value]

                    config[key] = value

    return config


def get_sample_config(configs):
    config_sample = {}

    for k in configs.keys():
        options = configs[k]
        idx = np.random.choice(len(options), 1)[0]
        config_sample[k] = options[idx]

    return config_sample


def config_to_string(config):
    keys = sorted(config.keys())
    string = ','.join(['%s:%s' % (k, str(config[k])) for k in keys])
    return string.lower()


def is_used_config(config, used_config_file):
    return config_to_string(config) in get_used_configs(used_config_file)


def get_used_configs(used_configs_file):
    used_configs = set()

    with open(used_configs_file, 'r') as file:
        for line in file:
            used_configs.add(line.strip())

    return used_configs


def save_used_config(config, used_config_file):
    with open(used_config_file, 'a') as file:
        cfg_str = config_to_string(config)
        file.write('%s\n' % cfg_str)


def run(config_file, num_runs):
    configs = load_config(config_file)

    outdir = configs['outdir'][0]
    used_configs_file = '%s/used_configs.txt' % outdir

    if not os.path.isfile(used_configs_file):
        file = open(used_configs_file, 'w')
        file.close()

    for i in range(num_runs):
        config = get_sample_config(configs)

        if is_used_config(config, used_configs_file):
            print('Configuration used, skipping')
            continue

        save_used_config(config, used_configs_file)

        print('------------------------------')
        print('Run %d of %d:' % (i + 1, num_runs))
        print('------------------------------')
        print('\n'.join(['%s: %s' % (str(key), str(value)) for key, value in config.items()]))
        print('\n'.join(['%s: %s' % (str(key), str(value)) for key, value in config.items() if len(configs[key]) > 1]))

        flags = ' '.join('--%s %s' % (key, str(value)) for key, value in config.items())
        call('python3 net_train.py %s' % flags, shell=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python hyperparameter_search.py <configuration file> <num runs>')
    else:
        run(sys.argv[1], int(sys.argv[2]))
