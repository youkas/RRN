import os

import numpy as np
from prod.utils import config as cfg
import msof.loader as ml
import msof.mesh as msh
from msof.geometry.ffd import Lattice
from msof.viz import Visu
from msof.viz.actors import ShapeCutActor, FFDActor, MeshActor, ScalarsActor, \
    ScalarProjectectionActor, ScalarIsoLineActor, \
    ScalarBarActor, ScalarCutterActor

#convert to vtk
#ml.save(initial_shape, '../temp/initial_shape.vtk', category='mesh', loader='vtk')
#ml.save(initial_mesh, '../temp/initial_mesh.vtk', category='mesh', loader='vtk')

def plot(_lattice, _shape, _mesh, _path):
    global initial_shape, initial_mesh, min_v, max_v
    pressure = _mesh.get('Pressure')

    root_p0 = np.array([-18500 / 2, -8500 / 2, 0])
    root_p1 = root_p0 + np.array([18500, 0, 0])
    root_p2 = root_p0 + np.array([0, 8500, 0])

    p0 = np.array([-14500 / 2, -6500 / 2, 5800])
    p1 = p0 + np.array([14500, 0, 0])
    p2 = p0 + np.array([0, 6500, 0])

    actors = [FFDActor(_lattice, radius=100, scale_factor=500),
              MeshActor(_shape, name='Shape', scalars='Pressure', edges=False, points=False, min_value=min_v, max_value=max_v),
              ScalarIsoLineActor(_mesh, 'Pressure', pressure, root_p0, root_p1, root_p2, 60, name='Root_Pressure_Isolines', min_value=min_v, max_value=max_v),
              ScalarIsoLineActor(_mesh, 'Pressure', pressure, p0, p1, p2, 60, name='Pressure_Isolines', min_value=min_v, max_value=max_v),
              ScalarBarActor('Pressure', pressure, min_value=min_v, max_value=max_v)
        ]
    #Visu.Show(*actors, title='', background1=(1., 1., 1.), background2=None)
    Visu.Write(_path, *actors)

initial_shape = ml.load(f"../database/shape/initial.npz", asType=msh.Mesh)
initial_mesh = ml.load(f"../database/mesh/initial.npz", asType=msh.Mesh)
initial_shape.import_point_data(initial_mesh)
ml.save(initial_shape, f"../database/shape/initial.npz", asType=msh.Mesh)
print(initial_shape.get('Pressure'))
min_v, max_v = np.min(initial_mesh.get('Pressure')), np.max(initial_mesh.get('Pressure'))
print(min_v, max_v)
"""
for c, v in cfg.data_configurations.items():
    lattice = ml.load(f"../database/lattice/{c}.npz", asType=Lattice)
    path = f'../database/vtk/{c}/'
    if not os.path.isdir(path):
        os.mkdir(path)
    plot(lattice, initial_shape, initial_mesh, path)
"""
IDS = cfg.get_ids()

for _id in IDS:
    try:
        lattice = ml.load(f"../database/lattice/{_id}.npz", asType=Lattice)
        shape = ml.load(f"../database/shape/{_id}.npz", asType=msh.Mesh)
        mesh = ml.load(f"../database/mesh/{_id}.npz", asType=msh.Mesh)
        shape.import_point_data(mesh)
        ml.save(shape, f"../database/shape/{_id}.npz", asType=msh.Mesh)

        path = f'../database/vtk/{_id}/'
        if not os.path.isdir(path):
            os.mkdir(path)
        plot(lattice, shape, mesh, path)
        print(f"{_id} Processed successfully")
    except FileNotFoundError as e:
        print(str(e))