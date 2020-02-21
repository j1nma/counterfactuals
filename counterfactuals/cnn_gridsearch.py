import datetime
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from counterfactuals import nn4, cnn
from counterfactuals.net_train import get_nn_args_parser


def gridsearch(config_file):
    # Parse arguments
    args = get_nn_args_parser().parse_args(['@' + config_file])
    outdir = args.outdir
    architecture = args.architecture

    # (1) Grid search hyperparameters pool_size
    kernel_size_range = np.linspace(2, 2, num=1, dtype='int')
    strides_range = np.linspace(2, 3, num=2, dtype='int')
    pool_size_range = np.linspace(2, 2, num=1, dtype='int')

    # (2) Construct combinations
    param_grid = {
        'kernel_size': kernel_size_range,
        'strides': strides_range,
        'pool_size': pool_size_range
    }
    combinations = list(ParameterGrid(param_grid))

    # (3) Initialize hyperparameters list
    kernel_size_list = []
    strides_list = []
    pool_size_list = []

    rmse_ite_list = []
    ate_list = []
    pehe_list = []
    mean_duration_list = []

    outdir = outdir + architecture + '/'
    outdir_path = Path(outdir)
    if not outdir_path.is_dir():
        os.makedirs(outdir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = outdir + timestamp + '/'
    os.mkdir(outdir)

    i = 0

    for combination in combinations:
        print('Evaluating ' + str(i + 1) + ' out of ' + str(len(combinations)) + ' combinations...')

        # (4) Overwrite hyperparameters into args
        # args.learning_rate_factor = combination['learning_rate_factor']

        # (5) Create results directory with hyperparameters
        combination_outdir = outdir \
                             + 'k-' + str(combination['kernel_size']) \
                             + '-s-' + str(combination['strides']) \
                             + '-p-' + str(combination['pool_size']) \
                             + '/'
        os.mkdir(combination_outdir)

        if args.architecture == 'nn4':
            result = nn4.run(args, combination_outdir)
        elif args.architecture == 'cnn':

            try:
                result = cnn.run(args, combination_outdir, combination['kernel_size'], combination['strides'],
                                 combination['pool_size'])
            except:
                traceback.print_exc()
                continue
        else:
            return "Architecture not found."

        # (6) Append used hyperparameters into list
        kernel_size_list.append(str(combination['kernel_size']))
        strides_list.append(str(combination['strides']))
        pool_size_list.append(str(combination['pool_size']))

        rmse_ite_list.append(result['ite'])
        ate_list.append(result['ate'])
        pehe_list.append(result['pehe'])
        mean_duration_list.append(result['mean_duration'])

        i += 1

    df = pd.DataFrame()

    # (7) Save hyperparameters list into dataframe
    df['kernel_size'] = kernel_size_list
    df['strides'] = strides_list
    df['pool_size'] = pool_size_list

    df['RMSE_ITE'] = np.array(rmse_ite_list)
    df['ATE'] = np.array(ate_list)
    df['PEHE'] = np.array(pehe_list)
    df['MEAN_DURATION'] = np.array(mean_duration_list)

    df.to_csv(outdir + architecture.lower() + '_gridsearch_results.csv', index=False)


if __name__ == "__main__":
    gridsearch(config_file=sys.argv[1])
