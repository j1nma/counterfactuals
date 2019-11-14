import numpy as np


def load_data(filename):
    if filename[-3:] != 'npz':
        raise Exception("Data file format is not npz.")

    data_in = np.load(filename)
    data = {'x': data_in['x'],
            't': data_in['t'],
            'yf': data_in['yf'],
            'ycf': data_in['ycf'],
            'mu0': data_in['mu0'],
            'mu1': data_in['mu1']}

    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]

    # adjust binary feature at index 13: {1, 2} -> {0, 1}
    data['x'][:, 13] -= 1

    return data


# def normalize(features_training, y_training):
#     features_mean, features_std = np.mean(features_training, axis=0), np.std(features_training, axis=0)
#     normalized_features_training = (features_training - features_mean) / features_std
#
#     y_mean, y_std = np.mean(y_training), np.std(y_training)
#     normalized_y_training = (y_training - y_mean) / y_std
#
#     # TODO: what about validation
#     # features_validation = (features_validation - features_mean) / features_std
#     # features_full_training_set = (features_full_training_set - features_mean) / features_std
#     # y_validation = (y_validation - y_mean) / y_std
#
#     return normalized_features_training, normalized_y_training
