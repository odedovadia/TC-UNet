import numpy as np
import os


def preprocess_hypersonics(configs):

    ### Structured grid
    data = np.load(os.path.join('..', 'data', 'hypersonics', 'hypersonic_data_256_256.npz'))
    x_train = data['x_train'][..., 0]
    y_train = data['y_train']
    t_train = np.moveaxis(data['t_train'], -1, 0)
    x_test = x_train
    y_test = data['y_test']
    t_test = np.moveaxis(data['t_test'], -1, 0)

    n_x = configs.nx
    n_y = configs.ny
    assert x_train.shape[1] == n_x and x_train.shape[2] == n_y
    assert y_train.shape[1] == n_x and y_train.shape[2] == n_y
    assert x_test.shape[1] == n_x and x_test.shape[2] == n_y
    assert y_test.shape[1] == n_x and y_test.shape[2] == n_y
    path = os.path.join('..', 'data', 'hypersonics',
                    f'Hypersonics_N_{configs.N}_T_{int(configs.T)}_' + 
                    f'nx_{n_x}_ny_{n_y}')
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, 'raw_data.npy'), 
                {
                    'x_train': x_train,
                    'y_train': y_train,
                    't_train': t_train,
                    'x_val': x_test,
                    'y_val': y_test,
                    't_val': t_test,
                    'x_test': x_test,
                    'y_test': y_test,
                    't_test': t_test,
                }
            )
    

    ### Unstructured grid
    data = np.load(os.path.join('..', 'data', 'hypersonics', 'hypersonic_data_65536.npz'))
    x_train = data['x_train'][..., 0]
    y_train = data['y_train']
    t_train = np.moveaxis(data['t_train'], -1, 0)
    x_test = x_train
    y_test = data['y_test']
    t_test = np.moveaxis(data['t_test'], -1, 0)

    n_x = configs.nx
    n_y = configs.ny
    path = os.path.join('..', 'data', 'hypersonics',
                    f'Hypersonics_N_{configs.N}_T_{int(configs.T)}_' + 
                    f'nx_{n_x}_ny_{n_y}')

    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, 'raw_data_points.npy'), 
                {
                    'x_train': x_train,
                    'y_train': y_train,
                    't_train': t_train,
                    'x_val': x_test,
                    'y_val': y_test,
                    't_val': t_test,
                    'x_test': x_test,
                    'y_test': y_test,
                    't_test': t_test,
                }
            )

