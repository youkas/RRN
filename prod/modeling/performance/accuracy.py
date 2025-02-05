#Models Accuracy
from prod.utils import config as cfg
import numpy as np

modeling_performance = np.load(f'../../database/modeling perf.npz', allow_pickle=True)['item'].item()
decoder_performance = np.load(f"../../database/decoder perf.npz", allow_pickle=True)['item'].item()
nn_decoder_performance = np.load(f"../../database/NN decoder perf.npz", allow_pickle=True)['item'].item()

error_name = 'AMSE'
error_label = 'Average MSE'
configuration_label = 'Configuration'
dimension_label = 'Lattent dimension'
Z_NORM = 2  # np.inf
Z_NORM_LABEL = r'$||Z||_2$'  # r'$||Z||_\infty$'
DIMENSION_LABEL = 'Dimension change rate'  # 'Lattent dimension'
ERROR_LABEL = 'Decoder Error'


def modeling_accuracy():
    import matplotlib.pyplot as plt

    names = list(cfg.data_configuration_selection.keys())
    labels = list(cfg.data_configuration_selection.values())
    indices = np.arange(len(names)) + 1
    types = ['kRRN', 'RRN', 'RRNLike', 'Surrogate']
    colors = ['green', 'orange', 'b', 'r']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 1, 5]})
    fig.subplots_adjust(hspace=0.0)


    values = [np.array([modeling_performance['RRNLike'][n][k][error_name] for k in modeling_performance['RRNLike'][n].keys()]) for n in names]
    ax1.boxplot(values, labels=labels, patch_artist=True,
                 boxprops={'facecolor':'white'}, medianprops={'color':colors[types.index('RRNLike')]})

    values = [modeling_performance['Surrogate'][n][error_name] for n in names]
    p2, = ax1.plot(indices, values, '+', markersize=10, color=colors[types.index('Surrogate')], label=cfg.model_types['Surrogate'])

    values = [np.array([modeling_performance['RRN'][n][k][error_name] for k in modeling_performance['RRN'][n].keys()]) for n in names]
    ax3.boxplot(values, labels=labels, patch_artist=True,
                 boxprops={'facecolor':'white'}, medianprops={'color':colors[types.index('RRN')]})

    values = [np.array([modeling_performance['kRRN'][n][k][error_name] for k in modeling_performance['kRRN'][n].keys()]) for n in names]
    ax3.boxplot(values, labels=labels, patch_artist=True,
                 boxprops={'facecolor':'white'}, medianprops={'color':colors[types.index('kRRN')]})

    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)

    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(right=False, left=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax2.spines['left'].set_linestyle((0,(3,6)))
    ax2.spines['right'].set_linestyle((0,(3,6)))

    ax3.spines['top'].set_visible(False)
    ax3.xaxis.tick_bottom()

    ax3.set_xlabel(configuration_label)
    ax1.set_ylabel(error_label)

    import matplotlib.lines as mlines
    rrn_like_patch =  mlines.Line2D([], [], color=colors[types.index('RRNLike')], label=cfg.model_types['RRNLike'])
    rrn_patch =  mlines.Line2D([], [], color=colors[types.index('RRN')], label=cfg.model_types['RRN'])
    krrn_patch =  mlines.Line2D([], [], color=colors[types.index('kRRN')], label=cfg.model_types['kRRN'])

    ax1.legend(handles=[p2, rrn_like_patch, rrn_patch, krrn_patch], ncol=3, title='Model type:')
    plt.show()

def dim_change_accuracy():
    import matplotlib.pyplot as plt
    names = list(cfg.data_configuration_selection.keys())
    labels = list(cfg.data_configuration_selection.values())
    indices = np.arange(len(names)) + 1
    types = ['kRRN', 'RRN']
    marker = ['+', 'x']
    line_style = ['-', (0, (5, 2))]
    colors = ['r', 'g', 'b', 'c', 'm']

    fig, axes = plt.subplots()
    for j, t in enumerate(types):
        for i, n in enumerate(names):
            x = [int(k) for k in modeling_performance[t][n].keys()]
            y = [v[error_name] for v in modeling_performance[t][n].values()]
            coef = np.polyfit(x, y, 1)
            poly1d_fn = np.poly1d(coef)
            axes.plot(x, y, marker[j], markersize=5, color=colors[names.index(n)], label=f'{t} {labels[i]}')
            axes.plot(x, poly1d_fn(x), linestyle=line_style[j], color=colors[names.index(n)])
    axes.set_xlabel(dimension_label)
    axes.set_ylabel(error_label)

    plt.legend(title=configuration_label, ncol=int(len(names) / 2))
    plt.show()

def decoder_accuracy_by_dim(performance, title):
    vmin = 0.06#2
    vmax = 2.#48

    names = list(k for k in cfg.data_configuration_selection.keys() if cfg.data_configurations[k]['second_dim'] == 0)
    labels = list(cfg.data_configuration_selection[n] for n in names)
    markers = ['x', '.', '+', 'o']

    import matplotlib.pyplot as plt
    from matplotlib import colors
    cmap = plt.get_cmap('jet', 48)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    for n in names:
        errors = np.vstack([v for v in performance['error'][n].values()])
        norms = np.vstack([np.linalg.norm(v, ord=Z_NORM, axis=1) for v in performance['z'][n].values()])
        dims = np.array([int(k) for k in performance['error'][n].keys()]).reshape(-1, 1)
        dims = dims / cfg.data_configurations[n]['first_dim']
        dims = np.repeat(dims, errors.shape[1], axis=1)
        plt.scatter(norms, errors, c=dims, cmap=cmap, vmin=vmin, vmax=vmax, s=30, marker=markers[names.index(n)], label=labels[names.index(n)])

    cbar = plt.colorbar()
    cbar.set_label(DIMENSION_LABEL)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel(Z_NORM_LABEL)
    plt.ylabel(ERROR_LABEL)
    plt.title(title)
    plt.show()

def decoder_accuracy_by_error(performance, title):
    vmin = 0
    vmax = 1.

    names = list(k for k in cfg.data_configuration_selection.keys() if cfg.data_configurations[k]['second_dim'] == 0)
    labels = list(cfg.data_configuration_selection[n] for n in names)
    markers = ['x', '.', '+', 'o']

    import matplotlib.pyplot as plt
    from matplotlib import colors
    cmap = plt.get_cmap('jet', 48)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    for n in names:
        errors = np.vstack([v for v in performance['error'][n].values()])
        norms = np.vstack([np.linalg.norm(v, ord=Z_NORM, axis=1) for v in performance['z'][n].values()])
        dims = np.array([int(k) for k in performance['error'][n].keys()]).reshape(-1, 1)
        dims = dims / cfg.data_configurations[n]['first_dim']
        dims = np.repeat(dims, errors.shape[1], axis=1)
        plt.scatter(dims, norms, c=errors, cmap=cmap, vmin=vmin, vmax=vmax, s=30, marker=markers[names.index(n)], label=labels[names.index(n)])

    cbar = plt.colorbar()
    cbar.set_label(ERROR_LABEL)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel(DIMENSION_LABEL)
    plt.ylabel(Z_NORM_LABEL)
    plt.title(title)
    plt.show()

modeling_accuracy()
dim_change_accuracy()
decoder_accuracy_by_dim(decoder_performance, 'Optimization-based Decoder')
decoder_accuracy_by_dim(nn_decoder_performance, 'Neural netwaork-based Decoder')
decoder_accuracy_by_error(decoder_performance, 'Optimization-based Decoder')
decoder_accuracy_by_error(nn_decoder_performance, 'Neural netwaork-based Decoder')
