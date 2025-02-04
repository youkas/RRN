import tensorflow as tf
import numpy as np
from prod.utils import rrn
from prod.utils import config as cfg

def create_model(data_path, primary_dim, secondary_dim, lattent_dim, model_path,
                 training_mode='Fit', epochs=50, calibration_epochs=25, learning_rate=0.001):

    split = np.load(f'{data_path}/split_indices.npz', allow_pickle=True)['item']

    kpis = {'MPE': [], 'MSE': [], 'stop_epoch': []}
    for sp in split:
        X = np.load(f'{data_path}/x.npz')['item']
        Y = np.load(f'{data_path}/y.npz')['item']

        Y = Y / np.array([[cfg.CL0, cfg.CD0]])

        Y_DISPERSION = np.std(Y, axis=0)
        Y_MIDDLE = np.mean(Y, axis=0)

        tst = [X[sp['test'], :], Y[sp['test'], :]]
        val = [X[sp['validation'], :], Y[sp['validation'], :]]
        tr = [X[sp['training'], :], Y[sp['training'], :]]

        model = rrn.Surrogate(primary_dim, secondary_dim, Y.shape[1], lattent_dim,
                              x_dispersion=(np.max(X, axis=0) - np.min(X, axis=0)) / 2,
                              x_middle=(np.max(X, axis=0) + np.min(X, axis=0)) / 2,
                              y_dispersion=Y_DISPERSION,
                              y_middle=Y_MIDDLE)

        trainer = rrn.Trainer(model, primary_dim, secondary_dim, training_mode=training_mode)
        trainer.compile(learning_rate=learning_rate, kernel="RBF")
        trainer.train(tr[0], (tr[1] - Y_MIDDLE) / Y_DISPERSION, val[0], (val[1] - Y_MIDDLE) / Y_DISPERSION,
                      epochs=epochs, calibration_epochs=calibration_epochs, verbose=0, batch_size=50,
                      model_path=model_path)
        error = trainer.test(tst[0], tst[1])

        for k in error.keys():
            kpis[k].append(error[k])
        kpis['stop_epoch'].append(trainer.stop_epoch)
    return kpis

results = {'kRRN':{}}
data = ['D1', 'D2', 'D3', 'D4', 'D5']
dimensions = np.arange(cfg.MIN_LATTENT_DIM, cfg.MAX_LATTENT_DIM + 1)

for i, d in enumerate(data):
    results['kRRN'][d] = {}
    for lt in dimensions:
        _lattent_dim = int(lt)
        _epochs = 10000
        _learning_rate = 0.01
        _model_path = None

        source = f"data {cfg.data_configurations[d]['second_dim_name'].lower()}".strip()
        _data_path = f'../../../{source}/{cfg.data_configurations[d]["name"]}'

        out = create_model(_data_path,
                             cfg.data_configurations[d]['first_dim'],
                             cfg.data_configurations[d]['second_dim'],
                             _lattent_dim,
                             _model_path,
                             training_mode='Fit',
                             epochs=_epochs,
                             calibration_epochs=0,
                             learning_rate=_learning_rate)
        results['kRRN'][d][_lattent_dim] = {'AESE': np.average(out['stop_epoch']),
                                           'AMSE': np.average(out['MSE']),
                                           'AMPE': np.average(out['MPE'])}

np.savez_compressed(f'../../database/modeling perf.npz', item=results)
