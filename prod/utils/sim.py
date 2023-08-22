import numpy as np

import msof.geometry as gm
import msof.geometry.bbox as bbox
import msof.geometry.ffd as ffd
import msof.analysis.cfd.casolver as ca

def get_config(name):
    c = name.split(' ')
    order = [int(o) for o in c[0]]
    dof = int(c[1])
    interval = float(c[2].replace('+', '').replace('-', ''))
    if len(c) == 3:
        return order, dof, interval
    size = int(c[3])
    return order, dof, interval, size

def get_lattice(shape, order, interval):
    deformation_frame = bbox.buildHull(shape.nodes, offset=np.array([1500., 3000., 0.]), trim='Y')
    script = dict(shape=[*order],
                  hull=list(deformation_frame),
                  limits={':': [0., 0.], f'1:{order[0] - 1}, :, 1:{order[2] - 1}, 1': [-interval, +interval]})
    builder = ffd.LatticeBuilder(**script)
    lattice = builder.build()
    return lattice

def get_parametrization(config, shape, mesh):
    order, dof, interval, size = get_config(config)
    lattice = get_lattice(shape, order, interval)
    return gm.BezierFFD(lattice, mesh)

def get_meshes(ffd_generator, point, show=False, name=''):
    from msof.viz import Visu
    from msof.viz.actors import FFDActor, MeshActor

    sh = ffd_generator.generate(point)
    if show:
        Visu.Show(MeshActor(sh, surface=False),
                  FFDActor(ffd_generator.lattice, radius=60, scale_factor=400),
                  title=name)
    return sh


def solve(mesh, mach, alpha, cl0=1., cd0=1.):
    solver = ca.CASolver()
    solver.xmach = mach
    solver.tetadeg = alpha
    solution = solver.solve(mesh)
    return solution, solver.Cl/cl0, solver.Cd/cd0
