from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from counterfactuals.utilities import load_data

data_dir = '../data/'
data_train = 'ihdp_npci_1-100.train.npz'
train_experiment = 0

ihdp = load_data('../' + data_dir + data_train)
X = ihdp['x'][:, :, train_experiment]
y = ihdp['yf'][:, train_experiment]
print(X.shape, y.shape)

feat_cols = ['feature' + str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

# For reproducibility of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

plt.gray()
fig = plt.figure(figsize=(16, 7))
for i in range(0, 15):
    ax = fig.add_subplot(3, 5, i + 1, title="Feature: {}".format(str(df.loc[rndperm[i], 'label'])))
    ax.matshow(df.loc[rndperm[i], feat_cols].values.reshape((25, 1)).astype(float))
# plt.show()

N = 300
df_subset = df.loc[rndperm, :].copy()
data_subset = df_subset[feat_cols].values

n_components = 10
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(data_subset)
print('Cumulative explained variation for {} principal components: {}'.format(n_components,
                                                                              np.sum(pca.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

df_subset['tsne-pca-one'] = tsne_pca_results[:, 0]
df_subset['tsne-pca-two'] = tsne_pca_results[:, 1]
plt.figure(figsize=(6, 6))
ax3 = plt.subplot(1, 1, 1)
sns.scatterplot(
    x="tsne-pca-one",
    y="tsne-pca-two",
    data=df_subset,
    legend=None,
    alpha=0.3,
    ax=ax3
)

plt.savefig("test.svg", format="svg")
