#Models Accuracy
from prod.utils import config as cfg
import numpy as np

modeling_performance = np.load(f'../../database/modeling perf.npz', allow_pickle=True)['item'].item()
dim_change_performance = np.load(f'../../database/dim change perf.npz', allow_pickle=True)['item'].item()
decoder_performance = np.load(f"../../database/decoder perf.npz", allow_pickle=True)['item'].item()

error_name = 'AMSE'
error_label = 'Average MSE'
configuration_label = 'Configuration'
dimension_label = 'Lattent dimension'
Z_NORM = 2  # np.inf
Z_NORM_LABEL = r'$||Z||_2$'  # r'$||Z||_\infty$'
DIMENSION_LABEL = 'Dimension change rate'  # 'Lattent dimension'
ERROR_LABEL = 'Backward Parametrization Error'


def modeling_accuracy():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots()
    names = ['D1', 'D3', 'D2', 'D4', 'D5']
    indices = [1, 2, 3, 4, 5]
    types = ['RRN', 'RRNLike', 'Surrogate']
    colors = ['white', 'r', 'b']

    values = [modeling_performance['RRNLike'][n][error_name] for n in names]
    p1, = axes.plot(indices, values, 'x', markersize=10, color=colors[types.index('RRNLike')], label='RRNLike')

    values = [modeling_performance['Surrogate'][n][error_name] for n in names]
    p2, = axes.plot(indices, values, '+', markersize=10, color=colors[types.index('Surrogate')], label='Surrogate')

    values = [np.array([dim_change_performance[n][k][error_name] for k in dim_change_performance[n].keys()]) for n in names]
    axes.boxplot(values, labels=names, patch_artist=True, boxprops={'facecolor':colors[types.index('RRN')]})

    axes.set_xlabel(configuration_label)
    axes.set_ylabel(error_label)
    plt.yscale("log")
    import matplotlib.patches as mpatches
    rrn_patch = mpatches.Patch(edgecolor='k', facecolor=colors[types.index('RRN')], label='RRN')

    plt.legend(handles=[p1, p2, rrn_patch], ncol=3, title='Model type:')
    plt.show()

def dim_change_accuracy():
    import matplotlib.pyplot as plt
    names = ['D1', 'D3', 'D2', 'D4', 'D5']
    indices = [1, 2, 3, 4, 5]
    types = ['RRN', 'RRNLike', 'Surrogate']
    colors = ['r', 'g', 'b', 'c', 'm']

    fig, axes = plt.subplots()
    for n in names:
        x = [int(k) for k in dim_change_performance[n].keys()]
        y = [v[error_name] for v in dim_change_performance[n].values()]
        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)
        axes.plot(x, y, 'x', markersize=5, color=colors[names.index(n)], label=f'{n}')
        axes.plot(x, poly1d_fn(x), linestyle=(0, (5, 2)), color=colors[names.index(n)])
    axes.set_xlabel(dimension_label)
    axes.set_ylabel(error_label)

    plt.legend(title=configuration_label, ncol=5)
    plt.show()

def decoder_accuracy_by_dim():
    vmin = 0.06#2
    vmax = 2.#48
    names = ['D1', 'D2']
    markers = ['x', '.']

    import matplotlib.pyplot as plt
    from matplotlib import colors
    cmap = plt.get_cmap('jet', 48)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    for n in names:
        errors = np.vstack([v for v in decoder_performance['error'][n].values()])
        norms = np.vstack([np.linalg.norm(v, ord=Z_NORM, axis=1) for v in decoder_performance['z'][n].values()])
        dims = np.array([int(k) for k in decoder_performance['error'][n].keys()]).reshape(-1, 1)
        dims = dims / cfg.data_configurations[n]['first_dim']
        dims = np.repeat(dims, errors.shape[1], axis=1)
        plt.scatter(norms, errors, c=dims, cmap=cmap, vmin=vmin, vmax=vmax, s=30, marker=markers[names.index(n)], label=n)

    cbar = plt.colorbar()
    cbar.set_label(DIMENSION_LABEL)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel(Z_NORM_LABEL)
    plt.ylabel(ERROR_LABEL)
    plt.show()

def decoder_accuracy_by_error():
    vmin = 0
    vmax = 1.
    names = ['D1', 'D2']
    markers = ['x', '.']

    import matplotlib.pyplot as plt
    from matplotlib import colors
    cmap = plt.get_cmap('jet', 48)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    for n in names:
        errors = np.vstack([v for v in decoder_performance['error'][n].values()])
        norms = np.vstack([np.linalg.norm(v, ord=Z_NORM, axis=1) for v in decoder_performance['z'][n].values()])
        dims = np.array([int(k) for k in decoder_performance['error'][n].keys()]).reshape(-1, 1)
        dims = dims / cfg.data_configurations[n]['first_dim']
        dims = np.repeat(dims, errors.shape[1], axis=1)
        plt.scatter(dims, norms, c=errors, cmap=cmap, vmin=vmin, vmax=vmax, s=30, marker=markers[names.index(n)], label=n)

    cbar = plt.colorbar()
    cbar.set_label(ERROR_LABEL)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel(DIMENSION_LABEL)
    plt.ylabel(Z_NORM_LABEL)
    plt.show()

modeling_accuracy()
dim_change_accuracy()
decoder_accuracy_by_dim()
decoder_accuracy_by_error()