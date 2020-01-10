from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from counterfactuals.utilities import load_data


def tsne_plot_pca10(data, label, learned_label=None, outdir='/results'):
    X = data
    y = label
    print(X.shape, y.shape)

    feat_cols = ['feature' + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    print('Size of the dataframe: {}'.format(df.shape))

    n_components = 15
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df)
    print('Cumulative explained variation for {} principal components: {}'.format(n_components,
                                                                                  np.sum(
                                                                                      pca.explained_variance_ratio_)))

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000)
    tsne_pca_results = tsne.fit_transform(pca_result)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    df['tsne-pca-one'] = tsne_pca_results[:, 0]
    df['tsne-pca-two'] = tsne_pca_results[:, 1]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    sns.scatterplot(
        x="tsne-pca-one",
        y="tsne-pca-two",
        data=df,
        legend=None,
        alpha=0.5,
        ax=ax
    )

    # START LEARNED SECTION
    if learned_label is not None:
        y_l = label
        print(X.shape, y_l.shape)

        df_l = pd.DataFrame(X, columns=feat_cols)
        df_l['y_l'] = y_l
        df_l['label_l'] = df_l['y_l'].apply(lambda i: str(i))
        print('Size of the dataframe: {}'.format(df_l.shape))

        pca_result_l = pca.fit_transform(df_l)
        print('Cumulative explained variation for {} principal components: {}'.format(n_components,
                                                                                      np.sum(
                                                                                          pca.explained_variance_ratio_)))

        time_start = time.time()
        tsne_pca_results_l = tsne.fit_transform(pca_result_l)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        df_l['tsne-pca-one-l'] = tsne_pca_results_l[:, 0]
        df_l['tsne-pca-two-l'] = tsne_pca_results_l[:, 1]
        sns.scatterplot(
            x="tsne-pca-one-l",
            y="tsne-pca-two-l",
            data=df_l,
            ax=ax
        )
        fig.add_subplot(ax)

    plt.savefig(outdir + '/tsne_c={}.svg'.format(n_components), format="svg")


def main():
    """ Main entry point for TSNE plotting. """

    data_dir = '../data/'
    data_train = 'ihdp_npci_1-100.train.npz'
    train_experiment = 0

    ihdp = load_data('../' + data_dir + data_train)

    data = ihdp['x'][:, :, train_experiment]
    label = ihdp['yf'][:, train_experiment]

    tsne_plot_pca10(data=data, label=label)


if __name__ == '__main__':
    main()
