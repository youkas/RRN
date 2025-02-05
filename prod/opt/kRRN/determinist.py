import numpy as np
from pymoo.algorithms.so_pso import PSO
from pymoo.optimize import minimize

from prod.utils import opt
from prod.utils import config as cfg

IDS = cfg.get_ids(M=['kRRN'], D=['D5'], d=[2], O=['Det'], f=['J1', 'J2'])
models = cfg.load_model(IDS, root='../../models')
database = np.load(f"../../database/results.npz", allow_pickle=True)['item'].item()

for _id in IDS:
    m, c, d, r, o, f = _id.split(sep='_')
    dim_x = cfg.data_configurations[c]['first_dim']
    dim_w = cfg.data_configurations[c]['second_dim']

    nominal = np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]) if dim_w == 2 else None
    evaluator = opt.Evaluator(models[_id], cfg.functions[f], nominal=nominal)
    prob = opt.DeterministProblem(models[_id].dimension,
                                  evaluator,
                                  [-cfg.radius[r] * np.ones(models[_id].dimension),
                                   cfg.radius[r] * np.ones(models[_id].dimension)])

    res = minimize(prob, PSO(), seed=1, verbose=False)

    mapper = opt.Mapper(models[_id], nominal=nominal)

    z = res.X.reshape((1, -1))
    Y_z = mapper.evaluate(z).flatten()
    ratio_z = Y_z[1]/Y_z[0]
    J_z = res.F.flatten()[0]

    x = models[_id].decode(z, x_lower=-200 * np.ones(dim_x), x_upper=200 * np.ones(dim_x), verbose=True)
    z_hat = models[_id].encode(x)
    decoder_error = np.linalg.norm(z - z_hat, axis=1).flatten()[0]
    Y_x = mapper.evaluate(z_hat).flatten()
    ratio_x = Y_x[1] / Y_x[0]
    J_x = evaluator.evaluate(z_hat).flatten()[0]

    database["results"][_id] = {"x":x, "z":z, "decoder_error":decoder_error,
                                "Y_x":Y_x, "Y_z":Y_z,
                                "ratio_x": ratio_x, "ratio_z": ratio_z,
                                "J_x": J_x, "J_z": J_z,
                                'effective':{},
                                'robust analysis':{},
                                'artefact':{"lattice":0, "shape":0, "mesh":0}}

np.savez_compressed(f"../../database/results.npz", item=database)