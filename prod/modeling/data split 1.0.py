import numpy as np
from prod.utils import config as cfg

data = ['D1', 'D2', 'D3', 'D4', 'D5']
chunks = 10 #4
for i, d in enumerate(data):
    source = f"data {cfg.data_configurations[d]['second_dim_name'].lower()}".strip()
    data_path = f'../../{source}/{cfg.data_configurations[d]["name"]}'

    sp = cfg.split_indices(cfg.data_configurations[d]["size"], chunks)
    np.savez_compressed(f'{data_path}/split_indices.npz', item=sp)
    for i in range(len(sp)):
        print(len(sp[i]['test']), len(sp[i]['validation']), len(sp[i]['training']))



