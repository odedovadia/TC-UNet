import numpy as np
import os


def preprocess_shallow_water(configs):
    sw1 = np.load(os.path.join('..', 'data', 'shallow_water', 'shallow-water-256x256x72_1.npz'))
    sw2 = np.load(os.path.join('..', 'data', 'shallow_water', 'shallow-water-256x256x72_2.npz'))
    x = np.concatenate([sw1['inputs'], sw2['inputs']])
    y = np.moveaxis(np.concatenate([sw1['outputs'], sw2['outputs']]), 1, -1)
    t = np.tile(np.linspace(0, 1, configs.T)[:, None], 300).transpose()
    n_x = sw1['inputs'].shape[1]
    n_y = sw1['inputs'].shape[2]
    path = os.path.join('..', 'data', 'shallow_water',
                    f'Shallow_Water_N_{configs.N}_T_{int(configs.T)}_size_' + 
                    f'nx_{n_x}_ny_{n_y}')
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, 'x.npy'), x)
    np.save(os.path.join(path, 'y.npy'), y)
    np.save(os.path.join(path, 't.npy'), t)
