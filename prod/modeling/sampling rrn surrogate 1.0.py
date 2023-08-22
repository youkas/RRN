import numpy as np
from prod.utils import rrn
from prod.utils import config as cfg
import matplotlib.pyplot as plt


def plot_contours(x, y, levels=100, xlabel='', ylabel='', zlabel='', titles=('', ''), filled=True):
    from matplotlib import colors
    def contours(axe, _x, _y, _z, title, yticks=True):

        if filled:
            axe.contourf(_x, _y, _z, levels=levels, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        else:
            axe.contour(_x, _y, _z, levels=levels, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)

        axe.set_xlabel(xlabel, fontsize = 13)
        if not yticks:
            axe.set_yticks([])
            axe.set_ylabel('')
        else:
            axe.set_ylabel(ylabel, fontsize = 13)
        axe.set_title(title, fontsize = 16)

    vmin = np.min(y)
    vmax = np.max(y)
    cmap = plt.get_cmap('jet')
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    #norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig, ax = plt.subplots(1, 2)
    contours(ax[0], x[:, :, 0], x[:, :, 1], y[:, :, 0], titles[0])
    contours(ax[1], x[:, :, 0], x[:, :, 1], y[:, :, 1], titles[1], yticks=False)

    cbar = plt.colorbar(sm, ax=ax, extend='both')
    #cbar.set_label(r'$'+zlabel+'$')
    plt.show()


model_type = 'RRN'
data = ['D5']
l, u = [-1.0, 1.0]
number_of_points = 1000

def sample_surrogate():
    samples = {}
    for i, d in enumerate(data):
        lt = 2
        _model_path = f'../models/{model_type}/{d}/{lt}'

        model = rrn.RRN()
        model.load(_model_path)
        x = np.meshgrid(*[np.linspace(l, u, number_of_points) for _ in range(model.dimension)])
        x = np.hstack([np.reshape(xi, (-1, 1)) for xi in x])

        if cfg.data_configurations[d]['second_dim'] == 2:
            x = np.hstack([x,
                           cfg.MACH_NOMINAL * np.ones((number_of_points ** 2, 1)),
                           cfg.INCIDENCE_NOMINAL * np.ones((number_of_points ** 2, 1))])
        y = model.predict(x)

        x = np.reshape(x[:, :-2], (number_of_points, number_of_points, 2))
        y = np.reshape(y, (number_of_points, number_of_points, 2))

        samples[d] = {'X':x, 'Y':y}
    np.savez_compressed(f"../database/surrogate samples.npz", item=samples)

def plot_samples():
    samples = np.load(f"../database/surrogate samples.npz", allow_pickle=True)['item'].item()
    for i, d in enumerate(data):
        x = samples[d]['X']
        y = samples[d]['Y']
        plot_contours(x, y, levels=100,
                      filled=True, xlabel=r'$\mathcal{Z}_0$', ylabel=r'$\mathcal{Z}_1$', zlabel=f'',
                      titles=(r'$\rho_{Cl}$', r'$\rho_{Cd}$'))

#sample_surrogate()
plot_samples()

