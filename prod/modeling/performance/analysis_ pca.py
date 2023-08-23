import numpy as np
from sklearn.decomposition import PCA
from prod.utils import config as cfg
import matplotlib.pyplot as plt

data = ['D1', 'D2', 'D3', 'D4', 'D5']
"""
def analyse_pca():
    pca_results = {}
    for i, d in enumerate(data):
        source = f"data {cfg.data_configurations[d]['second_dim_name'].lower()}".strip()
        data_path = f'../../../{source}/{cfg.data_configurations[d]["name"]}'

        X = np.load(f'{data_path}/x.npz')['item'][:, :cfg.data_configurations[d]['first_dim']]

        pca = PCA(n_components=X.shape[1])
        pca.fit(X)
        pca_results[d] = np.cumsum(pca.explained_variance_ratio_)

        np.savez_compressed(f"../../database/pca analysis.npz", item=pca_results)
"""

def analyse_pca():
    pca_results = {}
    for i, d in enumerate(data):
        source = f"data {cfg.data_configurations[d]['second_dim_name'].lower()}".strip()
        data_path = f'../../../{source}/{cfg.data_configurations[d]["name"]}'

        X = np.load(f'{data_path}/x.npz')['item']
        Y = np.load(f'{data_path}/y.npz')['item']
        Z = np.hstack([X, Y])
        pca = PCA(n_components=cfg.data_configurations[d]['first_dim'])
        pca.fit(Z)
        pca_results[d] = np.cumsum(pca.explained_variance_ratio_)

        np.savez_compressed(f"../../database/pca analysis.npz", item=pca_results)

def plot_analysis():
    pca_results = np.load(f"../../database/pca analysis.npz", allow_pickle=True)['item'].item()
    names = list(cfg.data_configuration_selection.keys())
    labels = list(cfg.data_configuration_selection.values())
    colors = ['r', 'b', 'g', 'y']
    for i, k in enumerate(names):
        dim = cfg.data_configurations[k]['first_dim']
        plt.plot((np.arange(dim) + 1)/dim, pca_results[k], '--', color=colors[i], label=labels[i])
    plt.xlabel(r'$\dfrac{Number\ of\ components}{Primary\ dimension}$')
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("PCA Analysis")
    plt.legend()
    plt.show()

#analyse_pca()
plot_analysis()
