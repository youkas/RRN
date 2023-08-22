import numpy as np
from pymoo.algorithms.so_pso import PSO
from pymoo.optimize import minimize

from prod.utils import opt
from prod.utils import config as cfg

IDS = cfg.get_ids(M=['RRNLike', 'Surrogate'], D=['D5'], d=[12], O=['Det'], f=['J1', 'J2'])
models = cfg.load_model(IDS, root='../../models')
database = np.load(f"../../database/results.npz", allow_pickle=True)['item'].item()
for _id in IDS:
    m, c, d, r, o, f = _id.split(sep='_')
    dim_x = cfg.data_configurations[c]['first_dim']
    dim_w = cfg.data_configurations[c]['second_dim']

    nominal = np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]) if dim_w == 2 else None

    evaluator = opt.Evaluator(models[_id], cfg.functions[f], nominal=nominal)
    prob = opt.DeterministProblem(dim_x,
                                  evaluator,
                                  [-200 * np.ones(dim_x),
                                   200 * np.ones(dim_x)])
    res = minimize(prob,
                   PSO(),
                   seed=1,
                   verbose=False)

    mapper = opt.Mapper(models[_id], nominal=nominal)
    x = res.X.reshape((1, -1))
    Y_x = mapper.evaluate(x).flatten()
    ratio_x = Y_x[1]/Y_x[0]
    J_x = res.F.flatten()[0]
    database['results'][_id] = {"x":x,
                                "Y_x":Y_x,
                                "ratio_x": ratio_x,
                                "J_x": J_x,
                                'effective':{},
                                'robust analysis':{},
                                'artefact':{"lattice":0, "shape":0, "mesh":0}}

np.savez_compressed(f"../../database/results.npz", item=database)