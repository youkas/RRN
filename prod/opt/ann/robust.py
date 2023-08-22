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


IDS = cfg.get_ids(M=['RRNLike', 'Surrogate'], D=['D5'], d=[12], O=['Rob1', 'Rob2', 'Rob3'], f=['J1'])
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
        dimension = int(d)
        prob = opt.RobustProblem(dimension,
                                 evaluator,
                                 [-200 * np.ones(dimension),
                                  200 * np.ones(dimension)])

        res = minimize(prob,
                       NSGA2(pop_size=500),
                       ('n_gen', 20),
                       seed=1,
                       verbose=True)

        x = res.X
        J_x = res.F

        database['results'][_id] = {"x":None,
                                    "Y_x":None,
                                    "ratio_x": None,
                                    "J_x": None,
                                    'effective':{},
                                    'robust analysis':{"x":x, "J_x":J_x, "selection":None},
                                    'artefact':{"lattice":0, "shape":0, "mesh":0}}

    np.savez_compressed(f"../../database/results.npz", item=database)

def sample_selection():
    database = np.load(f"../../database/results.npz", allow_pickle=True)['item'].item()
    for _id in IDS:
        import matplotlib.pyplot as plt
        selection = database['results'][_id]['robust analysis']["selection"]
        plt.scatter(database['results'][_id]['robust analysis']['J_x'][:, 0],
                    database['results'][_id]['robust analysis']['J_x'][:, 1], s=1, label='X')
        if selection is not None:
            plt.scatter(database['results'][_id]['robust analysis']['J_x'][selection, 0],
                        database['results'][_id]['robust analysis']['J_x'][selection, 1], s=15, label='X', c='red')

        for i in np.arange(len(database['results'][_id]['robust analysis']['J_x']), step=10):
            plt.annotate(str(i), (database['results'][_id]['robust analysis']['J_x'][i, 0],
                                  database['results'][_id]['robust analysis']['J_x'][i, 1]))

        plt.xlabel('Mean fitness')
        plt.ylabel('Fitness standard deviation')
        plt.legend()
        plt.title(_id)
        plt.show()
        selection = int(input('Selected index:'))
        database['results'][_id]['robust analysis']["selection"] = selection
        database['results'][_id]["x"] = database['results'][_id]['robust analysis']["x"][selection].reshape((1, -1))
    np.savez_compressed(f"../../database/results.npz", item=database)

def process_selection():
    database = np.load(f"../../database/results.npz", allow_pickle=True)['item'].item()
    for _id in IDS:
        m, c, d, r, o, f = _id.split(sep='_')
        nominal = np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL])
        evaluator = opt.Evaluator(models[_id], cfg.functions[f], nominal=nominal)

        mapper = opt.Mapper(models[_id], nominal=nominal)
        x = database['results'][_id]["x"]

        Y_x = mapper.evaluate(x).flatten()
        ratio_x = Y_x[1] / Y_x[0]
        J_x = evaluator.evaluate(x).flatten()[0]

        database['results'][_id] = {"x": x,
                                    "Y_x": Y_x,
                                    "ratio_x": ratio_x,
                                    "J_x": J_x,
                                    'effective': {},
                                    'robust analysis': database['results'][_id]['robust analysis'],
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

    DET_IDS = cfg.get_ids(M=['RRNLike', 'Surrogate'], D=['D5'], d=[12], O=['Det'], f=['J1'])

    for _id in DET_IDS:
        m, c, d, r, o, f = _id.split(sep='_')
        for op in ['Rob1', 'Rob2', 'Rob3']:
            rob_id = '_'.join([m, c, d, r, op, f])

            evaluator = get_evaluator(op, cfg.functions[f], models[rob_id])
            dimension = int(d)
            prob = opt.RobustProblem(dimension,
                                     evaluator,
                                     [-200 * np.ones(dimension),
                                      200 * np.ones(dimension)])

            det_x = database['results'][_id]['x']
            J_det = prob.evaluate(det_x)
            selected_x = database['results'][rob_id]['x']
            J_selected = prob.evaluate(selected_x)

            ev_both = opt.Evaluator(models[rob_id],
                              cfg.functions[f],
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_LOWER, cfg.INCIDENCE_LOWER]),
                              np.array([cfg.MACH_UPPER, cfg.INCIDENCE_UPPER]),
                              NUMBER_OF_SAMPLES,
                              resample=False)
            Y_det = np.hstack(ev_both.evaluate(det_x))
            master_points = np.vstack(([[cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]], ev_both.master_points))
            Y_selected = np.hstack(ev_both.evaluate(selected_x))

            ev_mach = opt.Evaluator(models[rob_id],
                              cfg.functions[f],
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_LOWER, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_UPPER, cfg.INCIDENCE_NOMINAL]),
                              NUMBER_OF_SAMPLES,
                              resample=False)
            Y_det_mach = np.hstack(ev_mach.evaluate(det_x))
            master_points_mach = np.vstack(([[cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]], ev_mach.master_points))
            Y_selected_mach = np.hstack(ev_mach.evaluate(selected_x))

            ev_alpha = opt.Evaluator(models[rob_id],
                              cfg.functions[f],
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]),
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_LOWER]),
                              np.array([cfg.MACH_NOMINAL, cfg.INCIDENCE_UPPER]),
                              NUMBER_OF_SAMPLES,
                              resample=False)
            Y_det_alpha = np.hstack(ev_alpha.evaluate(det_x))
            master_points_alpha = np.vstack(([[cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL]], ev_alpha.master_points))
            Y_selected_alpha = np.hstack(ev_alpha.evaluate(selected_x))

            database['results'][rob_id]['robust analysis']['x_det'] = det_x
            database['results'][rob_id]['robust analysis']['J_det'] = J_det
            database['results'][rob_id]['robust analysis']['x_selected'] = selected_x
            database['results'][rob_id]['robust analysis']['J_selected'] = J_selected

            database['results'][rob_id]['robust analysis']['Y_det'] = Y_det
            database['results'][rob_id]['robust analysis']['Y_selected'] = Y_selected
            database['results'][rob_id]['robust analysis']['master_points'] = master_points

            database['results'][rob_id]['robust analysis']['Y_det_mach'] = Y_det_mach
            database['results'][rob_id]['robust analysis']['Y_selected_mach'] = Y_selected_mach
            database['results'][rob_id]['robust analysis']['master_points_mach'] = master_points_mach

            database['results'][rob_id]['robust analysis']['Y_det_alpha'] = Y_det_alpha
            database['results'][rob_id]['robust analysis']['Y_selected_alpha'] = Y_selected_alpha
            database['results'][rob_id]['robust analysis']['master_points_alpha'] = master_points_alpha

    np.savez_compressed(f"../../database/results.npz", item=database)

#robust_optimization()
#sample_selection()
#process_selection()
#evaluate_determinist()