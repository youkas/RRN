import msof.loader as ml
import msof.mesh as msh
from msof.geometry.ffd import Lattice
from prod.utils import sim
import numpy as np
from prod.utils import config as cfg

shape = ml.load(f"../../data shape/Shape.npz", asType=msh.Mesh)
mesh = ml.load(f"../../data shape/CFD.npz", asType=msh.Mesh)
ml.save(shape, f"../database/shape/initial.npz", asType=msh.Mesh)
ml.save(mesh, f"../database/mesh/initial.npz", asType=msh.Mesh)
shape_generators = {}
mesh_generators = {}
for c, v in cfg.data_configurations.items():
    shape_generators[c] = sim.get_parametrization(v['name'], shape, shape)
    mesh_generators[c] = sim.get_parametrization(v['name'], shape, mesh)
    ml.save(shape_generators[c].lattice, f"../database/lattice/{c}.npz", asType=Lattice)

IDS = cfg.get_ids()

database = np.load(f"../database/results.npz", allow_pickle=True)['item'].item()
for _id in IDS:
    results = database["results"]
    if _id not in results:
        continue
    if results[_id]['artefact']["lattice"] and \
            results[_id]['artefact']["shape"] and \
            results[_id]['artefact']["mesh"]:
        continue

    m, c, d, r, o, f = _id.split(sep='_')

    sh = sim.get_meshes(shape_generators[c], results[_id]["x"].flatten(), show=False)
    ms = sim.get_meshes(mesh_generators[c], results[_id]["x"].flatten(), show=False)
    ltc = shape_generators[c].lattice
    ml.save(ltc, f"../database/lattice/{_id}.npz", asType=Lattice)
    ml.save(sh, f"../database/shape/{_id}.npz", asType=msh.Mesh)
    ml.save(ms, f"../database/mesh/{_id}.npz", asType=msh.Mesh)

    results[_id]['artefact'] = {"lattice":1, "shape":1, "mesh":1}
    np.savez_compressed(f"../database/results.npz", item=database)


