import datetime
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from counterfactuals import nn4, cnn
from counterfactuals.net_train import get_args_parser


def gridsearch(config_file):
    # Parse arguments
    args = get_args_parser().parse_args(['@' + config_file])
    outdir = args.outdir
    architecture = args.architecture

    # (1) Grid search hyperparameters
    learning_rate_factor_range = np.linspace(0.95, 0.99, num=4, dtype='float')

    # (2) Construct combinations
    param_grid = {
        'learning_rate_factor': learning_rate_factor_range,
    }
    combinations = list(ParameterGrid(param_grid))

    # (3) Initialize hyperparameters list
    learning_rate_factor_list = []

    rmse_ite_list = []
    ate_list = []
    pehe_list = []
    mean_duration_list = []

    outdir = outdir + architecture + '/'
    my_file = Path(outdir)
    if not my_file.is_dir():
        os.mkdir(outdir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = outdir + timestamp + '/'
    os.mkdir(outdir)

    i = 0

    for combination in combinations:
        print('Evaluating ' + str(i + 1) + ' out of ' + str(len(combinations)) + ' combinations...')

        # (4) Overwrite hyperparameters into args
        args.learning_rate_factor = combination['learning_rate_factor']

        # (5) Create results directory with hyperparameters
        combination_outdir = outdir + 'lrf-' + str(combination['learning_rate_factor']) + '/'
        os.mkdir(combination_outdir)

        if args.architecture == 'nn4':
            result = nn4.run(args, combination_outdir)
        elif args.architecture == 'cnn':
            result = cnn.run(args, combination_outdir)
        else:
            return "Architecture not found."

        # (6) Append used hyperparameters into list
        learning_rate_factor_list.append(str(combination['learning_rate_factor']))

        rmse_ite_list.append(result['ite'])
        ate_list.append(result['ate'])
        pehe_list.append(result['pehe'])
        mean_duration_list.append(result['mean_duration'])

        i += 1

    df = pd.DataFrame()

    # (7) Save hyperparameters list into dataframe
    df['learning_rate_factor'] = learning_rate_factor_list

    df['RMSE_ITE'] = np.array(rmse_ite_list)
    df['ATE'] = np.array(ate_list)
    df['PEHE'] = np.array(pehe_list)
    df['MEAN_DURATION'] = np.array(mean_duration_list)

    df.to_csv(outdir + 'gridsearch_results.csv', index=False)


if __name__ == "__main__":
    gridsearch(config_file=sys.argv[1])
