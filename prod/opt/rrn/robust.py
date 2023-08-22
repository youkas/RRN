import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.so_pso import PSO
from pymoo.model.callback import Callback
from pymoo.model.problem import Problem
from pymoo.optimize import minimize

from prod.utils import rrn
from prod.utils import opt
from prod.utils.misc import triangular_contours, curve
from prod.utils import config as cfg

IDS = cfg.get_ids(M=['RRN'], D=['D5'], d=[2], O=['Rob1', 'Rob2', 'Rob3'], f=['J1'])
models = cfg.load_model(IDS, root='../../models')

def robust_optimization():
    database = np.load(f"../../database/results.npz", allow_pickle=True)['item'].item()
    NUMBER_OF_SAMPLES = 300
    def get_evaluator(optim, func, mod):
        if optim == 'Rob3':
            return opt.Evaluator(mod,
                              func,
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_LOWER, cfg.INCIDENCE_LOWER]),
                              np.array([cfg.MACH_UPPER, cfg.INCIDENCE_UPPER]),
                              NUMBER_OF_SAMPLES,
                              resample=False)
        if optim == 'Rob2':
            return opt.Evaluator(mod,
                              func,
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_LOWER, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_UPPER, cfg.INCIDENCE_NOMINAL]),
                              NUMBER_OF_SAMPLES,
                              resample=False)
        return opt.Evaluator(mod,
                              func,
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_LOWER]),
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_UPPER]),
                              NUMBER_OF_SAMPLES,
                              resample=False)

    for _id in IDS:
        m, c, d, r, o, f = _id.split(sep='_')
        evaluator = get_evaluator(o, cfg.functions[f], models[_id])

        prob = opt.RobustProblem(models[_id].dimension,
                                 evaluator,
                                 [-cfg.radius[r] * np.ones(models[_id].dimension),
                                  cfg.radius[r] * np.ones(models[_id].dimension)])

        res = minimize(prob,
                       NSGA2(pop_size=500),
                       ('n_gen', 20),
                       seed=1,
                       verbose=True)

        z = res.X
        J_z = res.F

        database["results"][_id] = {"x":None, "z":None, "decoder_error":None,
                                    "Y_x":None, "Y_z":None,
                                    "ratio_x": None, "ratio_z": None,
                                    "J_x": None, "J_z": None,
                                    'effective':{},
                                    'robust analysis':{"z":z, "J_z":J_z, "selection":None},
                                    'artefact':{"lattice":0, "shape":0, "mesh":0}}

    np.savez_compressed(f"../../database/results.npz", item=database)

def sample_selection():
    database = np.load(f"../../database/results.npz", allow_pickle=True)['item'].item()
    for _id in IDS:
        import matplotlib.pyplot as plt
        selection = database["results"][_id]['robust analysis']["selection"]
        plt.scatter(database["results"][_id]['robust analysis']['J_z'][:, 0],
                    database["results"][_id]['robust analysis']['J_z'][:, 1], s=1, label='X')
        if selection is not None:
            plt.scatter(database["results"][_id]['robust analysis']['J_z'][selection, 0],
                        database["results"][_id]['robust analysis']['J_z'][selection, 1], s=15, label='X', c='red')

        for i in np.arange(len(database["results"][_id]['robust analysis']['J_z']), step=10):
            plt.annotate(str(i), (database["results"][_id]['robust analysis']['J_z'][i, 0],
                                  database["results"][_id]['robust analysis']['J_z'][i, 1]))

        plt.xlabel('Mean fitness')
        plt.ylabel('Fitness standard deviation')
        plt.legend()
        plt.title(_id)
        plt.show()
        selection = int(input('Selected index:'))
        database["results"][_id]['robust analysis']["selection"] = selection
        database["results"][_id]["z"] = database["results"][_id]['robust analysis']["z"][selection].reshape((1, -1))
    np.savez_compressed(f"../../database/results.npz", item=database)

def process_selection():
    database = np.load(f"../../database/results.npz", allow_pickle=True)['item'].item()
    for _id in IDS:
        m, c, d, r, o, f = _id.split(sep='_')
        dim_x = cfg.data_configurations[c]['first_dim']
        nominal = np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL])
        evaluator = opt.Evaluator(models[_id], cfg.functions[f], nominal=nominal)

        mapper = opt.Mapper(models[_id], nominal=nominal)
        z = database["results"][_id]["z"]
        Y_z = mapper.evaluate(z).flatten()
        ratio_z = Y_z[1] / Y_z[0]
        J_z = evaluator.evaluate(z).flatten()[0]

        x = models[_id].decode(z, -200 * np.ones(dim_x), 200 * np.ones(dim_x), verbose=True)
        z_hat = models[_id].encode(x)
        decoder_error = np.linalg.norm(z - z_hat, axis=1).flatten()[0]
        Y_x = mapper.evaluate(z_hat).flatten()
        ratio_x = Y_x[1] / Y_x[0]
        J_x = evaluator.evaluate(z_hat).flatten()[0]

        database["results"][_id] = {"x": x, "z": z, "decoder_error": decoder_error,
                                    "Y_x": Y_x, "Y_z": Y_z,
                                    "ratio_x": ratio_x, "ratio_z": ratio_z,
                                    "J_x": J_x, "J_z": J_z,
                                    'effective': {},
                                    'robust analysis': database["results"][_id]['robust analysis'],
                                    'artefact': {"lattice": 0, "shape": 0, "mesh": 0}}
    np.savez_compressed(f"../../database/results.npz", item=database)

def evaluate_determinist():
    database = np.load(f"../../database/results.npz", allow_pickle=True)['item'].item()
    NUMBER_OF_SAMPLES = 300
    def get_evaluator(optim, func, mod):
        if optim == 'Rob3':
            return opt.Evaluator(mod,
                              func,
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_LOWER, cfg.INCIDENCE_LOWER]),
                              np.array([cfg.MACH_UPPER, cfg.INCIDENCE_UPPER]),
                              NUMBER_OF_SAMPLES,
                              resample=False)
        if optim == 'Rob2':
            return opt.Evaluator(mod,
                              func,
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_LOWER, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_UPPER, cfg.INCIDENCE_NOMINAL]),
                              NUMBER_OF_SAMPLES,
                              resample=False)
        return opt.Evaluator(mod,
                              func,
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_LOWER]),
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_UPPER]),
                              NUMBER_OF_SAMPLES,
                              resample=False)


    DET_IDS = cfg.get_ids(M=['RRN'], D=['D5'], d=[2], O=['Det'], f=['J1'])

    for _id in DET_IDS:
        m, c, d, r, o, f = _id.split(sep='_')
        for op in ['Rob1', 'Rob2', 'Rob3']:
            rob_id = '_'.join([m, c, d, r, op, f])

            evaluator = get_evaluator(op, cfg.functions[f], models[rob_id])

            prob = opt.RobustProblem(models[rob_id].dimension,
                                     evaluator,
                                     [-cfg.radius[r] * np.ones(models[rob_id].dimension),
                                      cfg.radius[r] * np.ones(models[rob_id].dimension)])

            det_z = database["results"][_id]['z']
            J_det = prob.evaluate(det_z)
            selected_z = database["results"][rob_id]['z']
            J_selected = prob.evaluate(selected_z)

            ev_both = opt.Evaluator(models[rob_id],
                              cfg.functions[f],
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_LOWER, cfg.INCIDENCE_LOWER]),
                              np.array([cfg.MACH_UPPER, cfg.INCIDENCE_UPPER]),
                              NUMBER_OF_SAMPLES,
                              resample=False)
            Y_det = np.hstack(ev_both.evaluate(det_z))
            master_points = np.vstack(([[cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]], ev_both.master_points))
            Y_selected = np.hstack(ev_both.evaluate(selected_z))

            ev_mach = opt.Evaluator(models[rob_id],
                              cfg.functions[f],
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_LOWER, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_UPPER, cfg.INCIDENCE_NOMINAL]),
                              NUMBER_OF_SAMPLES,
                              resample=False)
            Y_det_mach = np.hstack(ev_mach.evaluate(det_z))
            master_points_mach = np.vstack(([[cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]], ev_mach.master_points))
            Y_selected_mach = np.hstack(ev_mach.evaluate(selected_z))

            ev_alpha = opt.Evaluator(models[rob_id],
                              cfg.functions[f],
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_LOWER]),
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_UPPER]),
                              NUMBER_OF_SAMPLES,
                              resample=False)
            Y_det_alpha = np.hstack(ev_alpha.evaluate(det_z))
            master_points_alpha = np.vstack(([[cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]], ev_alpha.master_points))
            Y_selected_alpha = np.hstack(ev_alpha.evaluate(selected_z))

            database["results"][rob_id]['robust analysis']['z_det'] = det_z
            database["results"][rob_id]['robust analysis']['J_det'] = J_det
            database["results"][rob_id]['robust analysis']['z_selected'] = selected_z
            database["results"][rob_id]['robust analysis']['J_selected'] = J_selected

            database["results"][rob_id]['robust analysis']['Y_det'] = Y_det
            database["results"][rob_id]['robust analysis']['Y_selected'] = Y_selected
            database["results"][rob_id]['robust analysis']['master_points'] = master_points

            database["results"][rob_id]['robust analysis']['Y_det_mach'] = Y_det_mach
            database["results"][rob_id]['robust analysis']['Y_selected_mach'] = Y_selected_mach
            database["results"][rob_id]['robust analysis']['master_points_mach'] = master_points_mach

            database["results"][rob_id]['robust analysis']['Y_det_alpha'] = Y_det_alpha
            database["results"][rob_id]['robust analysis']['Y_selected_alpha'] = Y_selected_alpha
            database["results"][rob_id]['robust analysis']['master_points_alpha'] = master_points_alpha

    np.savez_compressed(f"../../database/results.npz", item=database)

robust_optimization()
sample_selection()
process_selection()
evaluate_determinist()