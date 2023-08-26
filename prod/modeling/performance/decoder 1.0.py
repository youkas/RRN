from prod.utils import rrn
from prod.utils import config as cfg
from prod.utils.rrn import RRN
from prod.utils.sampling import PLHS
import numpy as np


def create_model(data_path, primary_dim, secondary_dim, lattent_dim, model_path,
                 training_mode='Fit', epochs=50, calibration_epochs=25, learning_rate=0.001):

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

    model = rrn.Surrogate(primary_dim, secondary_dim, Y.shape[1], lattent_dim,
                          x_dispersion=(np.max(X, axis=0) - np.min(X, axis=0)) / 2,
                          x_middle=(np.max(X, axis=0) + np.min(X, axis=0)) / 2,
                          y_dispersion=Y_DISPERSION,
                          y_middle=Y_MIDDLE)

    trainer = rrn.Trainer(model, primary_dim, secondary_dim, training_mode=training_mode)
    trainer.compile(learning_rate=learning_rate)
    trainer.train(tr[0], (tr[1] - Y_MIDDLE) / Y_DISPERSION, val[0], (val[1] - Y_MIDDLE) / Y_DISPERSION,
                  epochs=epochs, calibration_epochs=calibration_epochs, verbose=0, batch_size=50,
                  model_path=model_path)

data = ['D1', 'D2']

def create_models():
    for i, d in enumerate(data):
        lts = cfg.dimensions[d]
        for lt in lts:
            _lattent_dim = int(lt)
            _epochs = 10000
            _learning_rate = 0.01
            _model_path = f'../../models/RRN/{d}/{lt}'

            source = f"data {cfg.data_configurations[d]['second_dim_name'].lower()}".strip()
            _data_path = f'../../../{source}/{cfg.data_configurations[d]["name"]}'

            create_model(_data_path,
                         cfg.data_configurations[d]['first_dim'],
                         cfg.data_configurations[d]['second_dim'],
                         _lattent_dim,
                         _model_path,
                         training_mode='Fit',
                         epochs=_epochs,
                         calibration_epochs=0,
                         learning_rate=_learning_rate)

            print(_model_path)

def get_master_z():
    errors = {'z':{}}
    for i, d in enumerate(data):
        errors['z'][d] = {}
        lts = cfg.dimensions[d]
        for lt in lts:
            z = PLHS(1., lt, 10)
            #norm = np.linalg.norm(z, axis=1)
            #print(z.shape)
            #print(np.min(norm), np.max(norm), np.mean(norm), np.std(norm))
            #print(z.shape)
            #print(70*"*")
            errors['z'][d][str(lt)] = z

    np.savez_compressed(f"../../database/decoder perf.npz", item=errors)

def decode_master_z():
    errors = np.load(f"../../database/decoder perf.npz", allow_pickle=True)['item'].item()
    if not 'z_hat' in errors:
        errors['z_hat'] = {}
    for d in data:
        if not d in errors['z_hat']:
            errors['z_hat'][d] = {}
        lts = cfg.dimensions[d]
        for lt in lts:
            model = RRN()
            model.load(f'../../models/RRN/{d}/{lt}')

            first_dim = cfg.data_configurations[d]['first_dim']
            z = errors['z'][d][str(lt)]
            x = np.array([model.decode(zi.reshape((1, -1)),
                                       -200 * np.ones(first_dim),
                                       200 * np.ones(first_dim), verbose=True).flatten() for zi in z])
            errors['z_hat'][d][str(lt)] = model.encode(x)

    np.savez_compressed(f"../../database/decoder perf.npz", item=errors)

def get_decoder_fitness():
    errors = np.load(f"../../database/decoder perf.npz", allow_pickle=True)['item'].item()
    if not 'error' in errors:
        errors['error'] = {}

    for d in data:
        if not d in errors['error']:
            errors['error'][d] = {}
        lts = cfg.dimensions[d]
        for lt in lts:
            z = errors['z'][d][str(lt)]
            z_hat = errors['z_hat'][d][str(lt)]

            errors['error'][d][str(lt)] = np.linalg.norm(z - z_hat, axis=1)

    np.savez_compressed(f"../../database/decoder perf.npz", item=errors)

#create_models()
#get_master_z()
#decode_master_z()
#get_decoder_fitness()
