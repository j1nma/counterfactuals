import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from counterfactuals import nn4
from counterfactuals.net_train import get_args_parser

num_workers = 2
seed = 1
outdir = 'results/ihdp/gridsearch/'
data_dir = 'data/'
data_train = 'ihdp_npci_1-100.train.npz'
data_test = 'ihdp_npci_1-100.test.npz'

architecture = 'nn4'
epochs = 100
learning_rate = 0.001
weight_decay = 0.0001  # l2_weight_decay_lambda
input_size = 26  # 25 features + 1 treatment
hidden_size = 25
train_experiments = 100
learning_rate_factor = 0.96
learning_rate_steps = 2000  # changes the learning rate for every n updates
batch_size = batch_size_per_unit = 32  # mini-batch size

# Grid search parameters
epochs_range = np.linspace(200, 300, num=2, dtype='int')
learning_rate_steps_range = np.linspace(2000, 3000, num=2, dtype='int')

param_grid = {
    'epochs': epochs_range,
    'learning_rate_steps': learning_rate_steps_range,
}
combinations = list(ParameterGrid(param_grid))

epochs_list = []
learning_rate_steps_list = []

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

    args = get_args_parser().parse_args()

    args.train_experiments = train_experiments
    args.batch_size_per_unit = batch_size

    args.epochs = combination['epochs']
    args.learning_rate_steps = combination['learning_rate_steps']

    # Create results directory
    combination_outdir = outdir + str(combination['epochs']) + '-' + str(combination['learning_rate_steps'])
    os.mkdir(combination_outdir)

    result = nn4.run(args, combination_outdir)

    epochs_list.append(str(combination['epochs']))
    learning_rate_steps_list.append(str(combination['learning_rate_steps']))

    rmse_ite_list.append(result['ite'])
    ate_list.append(result['ate'])
    pehe_list.append(result['pehe'])
    mean_duration_list.append(result['mean_duration'])

    i += 1

myDf = pd.DataFrame()
myDf['epochs'] = epochs_list
myDf['learning_rate_steps'] = learning_rate_steps_list

myDf['RMSE_ITE'] = np.array(rmse_ite_list)
myDf['ATE'] = np.array(ate_list)
myDf['PEHE'] = np.array(pehe_list)
myDf['MEAN_DURATION'] = np.array(mean_duration_list)

myDf.to_csv(outdir + 'gridsearch_results.csv', index=False)
