import numpy as np

CL0 = 7.5860545461673146E-002
CD0 = 6.1904918213793615E-003

MACH_NOMINAL = 0.83
MACH_SCALE = 0.04
INCIDENCE_NOMINAL = 2.
INCIDENCE_SCALE = 0.04
MACH_LOWER = MACH_NOMINAL - MACH_SCALE
MACH_UPPER = MACH_NOMINAL + MACH_SCALE
INCIDENCE_LOWER = INCIDENCE_NOMINAL - INCIDENCE_SCALE
INCIDENCE_UPPER = INCIDENCE_NOMINAL + INCIDENCE_SCALE

MIN_LATTENT_DIM = 2
MAX_LATTENT_DIM = 48

optimization = {'Det': 'Determinist', 'Rob1':'Robust Alpha', 'Rob2':'Robust Mach', 'Rob3':'Robust Alpha and Mach'}
objectives = {'J1':"Drag to Lift ratio",
              'J2':"Drag minimization under constraint on Lift  (1%)",
              'J3':"Drag minimization under constraint on Lift (5%)",
              'J4':"Drag minimization under constraint on Lift (10%)",
              'J5':"Drag minimization under constraint on Lift (20%)"}
radius = {'s':0.5, 'm':0.8, 'l':1.}
data_configurations = {
    'D1': {'name': '425 12 +-200 1000', 'size':1000, 'first_dim': 12, 'second_dim': 0, 'out_dim': 2, 'second_dim_name': '', 'out': ['rho_Cl', 'rho_Cd']},
    'D2': {'name': '625 24 +-200 1000', 'size':1000, 'first_dim': 24, 'second_dim': 0, 'out_dim': 2, 'second_dim_name': '', 'out': ['rho_Cl', 'rho_Cd']},
    'D3': {'name': '425 12 +-200 1000', 'size':1000, 'first_dim': 12, 'second_dim': 2, 'out_dim': 2, 'second_dim_name': 'MI', 'out': ['rho_Cl', 'rho_Cd']},
    'D4': {'name': '625 24 +-200 1000', 'size':1000, 'first_dim': 24, 'second_dim': 2, 'out_dim': 2, 'second_dim_name': 'MI', 'out': ['rho_Cl', 'rho_Cd']},
    'D5': {'name': '425 12 +-200 3000', 'size':3000, 'first_dim': 12, 'second_dim': 2, 'out_dim': 2, 'second_dim_name': 'MI', 'out': ['rho_Cl', 'rho_Cd']}
}

data_configuration_selection = {
    'D1': 'Dataset 1',
    'D3': 'Dataset 2',
    'D5': 'Dataset 3',
    'D2': 'Dataset 4'
}
model_types = {'RRN': 'RRN',
               'kRRN': 'kRRN',
               'RRNLike': 'RRN Like',
               'Surrogate': 'Surrogate Like'}


dimensions = {k:np.arange(2, 2 * v['first_dim'] + 1) for k, v in data_configurations.items()}

functions = {'J1': lambda y: y[:, 1]/y[:, 0],
             'J2': lambda y: y[:, 1] + 10**4 * np.maximum((1. - y[:, 0]) - 0.01, 0.),
             'J3': lambda y: y[:, 1] + 10**4 * np.maximum((1. - y[:, 0]) - 0.05, 0.),
             'J4': lambda y: y[:, 1] + 10**4 * np.maximum((1. - y[:, 0]) - 0.10, 0.),
             'J5': lambda y: y[:, 1] + 10**4 * np.maximum((1. - y[:, 0]) - 0.20, 0.)}

database = {'data configuration':data_configurations,
            'model type':model_types,
            'optimization':optimization,
            'objectives':objectives,
            'design space radius':radius,
            'parametrization dimension':dimensions,
            'results':{}}

def split_indices(size, portions):
    indices = np.arange(size)
    np.random.shuffle(indices)
    splits = np.array_split(indices, portions)
    conbinations = []
    for i in range(portions):
        _splits = list(splits)
        further_splits = np.array_split(_splits.pop(0), 2)
        conbinations.append({'test':further_splits[0], 'validation':further_splits[1], 'training':np.concatenate(_splits)})
    return conbinations

def get_id(M, D, d, r, O, f):
    return '_'.join([str(M), str(D), str(d), str(r), str(O), str(f)])

def get_ids(M=None, D=None, d=None, r=None, O=None, f=None):
    M = M if M is not None else model_types.keys()
    D = D if D is not None else data_configurations.keys()
    r = r if r is not None else radius.keys()
    O = O if O is not None else optimization.keys()
    f = f if f is not None else objectives.keys()
    IDS = []
    for _M in M:
        for _D in D:
            for _d in d if d is not None else dimensions[_D]:
                for _r in r:
                    for _O in O:
                        for _f in f:
                            IDS.append(get_id(_M, _D, _d, _r, _O, _f))
    return IDS

def load_model(keys, root='./models'):
    from prod.utils import rrn, ann
    models = {}
    for k in keys:
        M, D, d, r, O, f = k.split(sep='_')
        models[k] = rrn.RRN() if M in ['kRRN', 'RRN'] else ann.ANN()
        models[k].load(f'{root}/{M}/{D}/{d}')
    return models
