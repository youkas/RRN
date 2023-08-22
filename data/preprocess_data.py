import numpy as np

def shuffle(path):
    x = np.load(f'../data/{path}/training_points.npz')['item']
    y = np.load(f'../data/{path}/training_features.npz')['item'][:, :2]

    d = np.hstack((x, y))
    np.random.shuffle(d)

    np.savez_compressed(f'./{path}/x.npz', item=d[:, :-2])
    np.savez_compressed(f'./{path}/y.npz', item=d[:, -2:])

paths = ['425 12 +-200 1000', '625 24 +-20 1000', '625 24 +-100 1000', '625 24 +-200 1000']

for p in paths:
    shuffle(p)
