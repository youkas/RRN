import numpy as np
from prod.utils import ann
from prod.utils import config as cfg


def create_model(data_path, primary_dim, secondary_dim, lattent_dim, model_path, epochs=50, learning_rate=0.001, patience=5):
    split = np.load(f'{data_path}/split_indices.npz', allow_pickle=True)['item']
    sp = split[0]

    X = np.load(f'{data_path}/x.npz')['item']
    Y = np.load(f'{data_path}/y.npz')['item']

    Y = Y / np.array([[cfg.CL0, cfg.CD0]])

    Y_DISPERSION = np.std(Y, axis=0)
    Y_MIDDLE = np.mean(Y, axis=0)

    #vi = np.concatenate((sp['test'], sp['validation']))
    #ti = sp['training']
    ti = np.concatenate((sp['training'], sp['validation']))
    vi = sp['test']
    val = [X[vi, :], Y[vi, :]]
    tr = [X[ti, :], Y[ti, :]]

    model = ann.RRNLike(primary_dim, secondary_dim, lattent_dim, Y.shape[1],
                          x_dispersion=(np.max(X, axis=0) - np.min(X, axis=0)) / 2,
                          x_middle=(np.max(X, axis=0) + np.min(X, axis=0)) / 2,
                          y_dispersion=Y_DISPERSION,
                          y_middle=Y_MIDDLE)

    trainer = ann.Trainer(model)
    trainer.compile(learning_rate=learning_rate)
    trainer.train(tr[0], (tr[1] - Y_MIDDLE) / Y_DISPERSION, val[0], (val[1] - Y_MIDDLE) / Y_DISPERSION,
                  epochs=epochs, verbose=0, batch_size=50, model_path=model_path, patience=patience)

    print(trainer.test(X, Y))

model_type = 'RRNLike'
data = ['D5']

for index, d in enumerate(data):
    _epochs = 10000
    _learning_rate = 0.01
    _patience = 5
    _latent_dim = cfg.data_configurations[d]['first_dim']
    _model_path = f'../models/{model_type}/{d}/{_latent_dim}'

    full_dimension = cfg.data_configurations[d]['first_dim'] + cfg.data_configurations[d]['second_dim']

    source = f"data {cfg.data_configurations[d]['second_dim_name'].lower()}".strip()
    _data_path = f'../../{source}/{cfg.data_configurations[d]["name"]}'

    create_model(_data_path,
                       cfg.data_configurations[d]['first_dim'],
                       cfg.data_configurations[d]['second_dim'],
                       cfg.data_configurations[d]['first_dim'],
                       _model_path,
                       epochs=_epochs, learning_rate=_learning_rate, patience=_patience)

    print(_model_path)
