import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def plot_t_y(t, y, t_labels=(r'$t_0$', r'$t_1$'), y_labels=(r'$C_l$', r'$C_d$')):
    count = y.shape[1]
    fig, ax = plt.subplots(1, count)

    for i in range(count):
        scatter1 = ax[i].scatter(t[:, 0], t[:, 1], c=y[:, i], cmap="jet")
        legend1 = ax[i].legend(*scatter1.legend_elements(num=5), title=y_labels[i])
        ax[i].add_artist(legend1)
        ax[i].set_xlabel(t_labels[0])
        if not i:
            ax[i].set_ylabel(t_labels[1])
        else:
            ax[i].set_yticks([])
    fig.subplots_adjust(wspace=0)
    plt.show()

def plot_params_distribution(t, labels=None):
    import seaborn as sns
    if t.shape[1] == 2:
        g = sns.jointplot(t[:, 0], t[:, 1], kind='kde')
        g.plot_joint(sns.scatterplot, color='b', marker="+", s=50, )
        axis_labels = labels if labels is not None else (r"$t_0$", r"$t_1$")
        g.set_axis_labels(*axis_labels)
    else:
        sns.displot({labels[i] if labels is not None else r'$t_' + str(i) + '$': t[:, i] for i in range(t.shape[1])}, kind="kde")
    plt.show()

def plot_features(t, y, marker_size=2., t_label='', y_label='', labels=()):
    count = y.shape[1]
    params = t.shape[1]
    fig, axes = plt.subplots(1, count)
    for i in range(count):
        for j in range(params):
            axes[i].scatter(t[:, j], y[:, i], s=marker_size, label=str(j))
        legend1 = axes[i].legend(title=labels[i])
        axes[i].add_artist(legend1)
        axes[i].set_xlabel(t_label)
        axes[i].set_ylabel(y_label)
    plt.show()

def triangular_contours(t, y, levels=30, filled=False, marker_size=2., titles=[], t_labels=('Mach number', 'Angle of attack'), bar_labels='Fitness'):
    count = y.shape[1]

    if len(titles) == 0:
        titles = ['' for i in range(count)]
    assert len(titles) == count

    fig, axes = plt.subplots(1, count)
    vmin = np.min(y)
    vmax = np.max(y)
    cmap = plt.get_cmap('jet', levels)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    for i in range(count):
        if filled:
            axes[i].tricontourf(t[:, 0], t[:, 1], y[:, i], levels=levels, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        else:
            axes[i].tricontour(t[:, 0], t[:, 1], y[:, i], levels=levels, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        axes[i].scatter(t[:, 0], t[:, 1], s=marker_size, c=y[:, i], cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        axes[i].set_xlabel(t_labels[0])
        axes[i].set_title(titles[i], fontsize = 16)
        if not i:
            axes[i].set_ylabel(t_labels[1])

    cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), extend='both')
    cbar.set_label(bar_labels)
    plt.show()

def contours(x, y, z, levels=50, xlabel='', ylabel='', zlabel='', title='', selection=None, filled=True, grid=False, xticks=None, yticks=None):
    vmin = np.min(z)
    vmax = np.max(z)
    cmap = plt.get_cmap('jet')
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    #norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig, ax = plt.subplots()
    if filled:
        ax.contourf(x, y, z, levels=levels, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    else:
        ax.contour(x, y, z, levels=levels, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)

    ax.set_xlabel(r'$'+xlabel+'$')
    ax.set_ylabel(r'$'+ylabel+'$')
    ax.set_title(title)
    if grid:
        ax.grid()
    if xticks is not None:
        ax.xaxis.set_ticks(xticks)
    if yticks is not None:
        ax.yaxis.set_ticks(yticks)
    cbar = plt.colorbar(sm, ax=ax, extend='both')
    cbar.set_label(r'$'+zlabel+'$')
    if selection is not None:
        points = np.array(list(selection.values()))
        ax.scatter(points[:, 0], points[:, 1], s=15, marker='s', c='k')
        for i, txt in enumerate(selection.keys()):
            ax.annotate(r'$'+txt+'$', (selection[txt][0], selection[txt][1]), fontsize=15)
    plt.show()

def curve(t, y, marker_size=2., t_label='Mach number', y_label='Fitness', labels=(), selection=None, scatter=True, grid=False, xticks=None, yticks=None):
    count = y.shape[1]
    fig, axes = plt.subplots(1, 1)
    for i in range(count):
        if scatter:
            axes.scatter(t, y[:, i], s=marker_size, label=labels[i])
        else:
            axes.plot(t, y[:, i], label=labels[i])
    if selection is not None:
        axes.scatter(selection[:, 0], selection[:, 1], s=50, marker='s', c='k')

    axes.set_xlabel(t_label)
    axes.set_ylabel(y_label)
    if grid:
        axes.grid()
    if xticks is not None:
        axes.xaxis.set_ticks(xticks)
    if yticks is not None:
        axes.yaxis.set_ticks(yticks)
    plt.legend()
    plt.show()

