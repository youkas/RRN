import numpy as np
import msof.loader as ml
from prod.utils import config as cfg
from prod.utils import sim
import msof.mesh as msh

mesh = ml.load(f"../database/mesh/initial.npz", asType=msh.Mesh)
solution, Cl, Cd = sim.solve(mesh, cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL)
ml.save(solution, f"../database/mesh/initial.npz", asType=msh.Mesh)

IDS = cfg.get_ids()
database = np.load(f"../database/results.npz", allow_pickle=True)['item'].item()
for _id in IDS:
    results = database["results"]
    if _id not in results:
        continue
    if not results[_id]['artefact']["mesh"]:
        continue
    m, c, d, r, o, f = _id.split(sep='_')

    mesh = ml.load(f"../database/mesh/{_id}.npz", asType=msh.Mesh)
    solution, Cl, Cd = sim.solve(mesh, cfg.MACH_NOMINAL, cfg.INCIDENCE_NOMINAL)
    rho_cl = Cl / cfg.CL0
    rho_cd = Cd / cfg.CD0
    ClCd = np.array([Cl, Cd])
    ratio = rho_cd/rho_cl
    Y = np.array([rho_cl, rho_cd])
    J = cfg.functions[f](np.array([[rho_cl, rho_cd]])).flatten()[0]

    results[_id]['effective']["ClCd"] = ClCd
    results[_id]['effective']["Y"] = Y
    results[_id]['effective']["ratio"] = ratio
    results[_id]['effective']["J"] = J
    ml.save(solution, f"../database/mesh/{_id}.npz", asType=msh.Mesh)
    np.savez_compressed(f"../database/results.npz", item=database)