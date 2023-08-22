import os

import numpy as np
from prod.utils import config as cfg

if not os.path.isdir(f"./database"):
    os.mkdir(f"./database")

if not os.path.isdir(f"./database/lattice"):
    os.mkdir(f"./database/lattice")

if not os.path.isdir(f"./database/mesh"):
    os.mkdir(f"./database/mesh")

if not os.path.isdir(f"./database/shape"):
    os.mkdir(f"./database/shape")

if not os.path.isdir(f"./database/vtk"):
    os.mkdir(f"./database/vtk")

np.savez_compressed(f"./database/results.npz", item=cfg.database)
