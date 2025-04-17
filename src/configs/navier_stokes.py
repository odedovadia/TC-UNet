import ml_collections
import numpy as np


def get_navier_stokes_configs():
    config = ml_collections.ConfigDict()

    # Data configs
    config.scenario = "navier_stokes"
    config.viscosity = 1e-03
    config.reynolds_number = int(
        np.sqrt(0.1) / ((2 * np.pi) ** (3/2) * config.viscosity))
    config.N = 1000
    config.T = 50
    config.noise_level = 0.

    # Grid configs
    config.nx = 64 * 1
    config.ny = 64 * 1
    config.output_shape = (config.nx, config.ny)
    config.time_steps_train = 50
    config.time_steps_inference = 200
    config.time_steps_inference_list = [10, 20, 50, 100, 200]
    config.sr_factors = [1]

    # Data loading configs
    config.ntrain = None
    config.nval = None
    config.ntest = None
    config.use_same_validation_and_test = False
    config.normalize = True
    config.dtype = 'float32'
    config.subsample = 10

    config.batch_size_train = {'ditto': 80,
                               'ditto_point': 80,
                               f'ditto_subsampled_X{config.subsample}': 800,
                               f'ditto_point_subsampled_X{config.subsample}': 800,
                               f'ditto_point_subsampled_X{config.subsample}_gate': 800,
                               'FNO': 1,
                               'UNet': 2,
                               'RNN': 10,
                               'DeepONet': 400}
    config.batch_size_val = {'ditto': 10,
                             'ditto_point': 400,
                             f'ditto_subsampled_X{config.subsample}': 100,
                             f'ditto_point_subsampled_X{config.subsample}': 400,
                             f'ditto_point_subsampled_X{config.subsample}_gate': 400,
                             'FNO': 1,
                             'UNet': 2,
                             'RNN': 50,
                             'DeepONet': 400}
    config.batch_size_test = {'ditto': 100,
                              'ditto_point': 100,
                              f'ditto_subsampled_X{config.subsample}': 100,
                              f'ditto_point_subsampled_X{config.subsample}': 100,
                              f'ditto_point_subsampled_X{config.subsample}_gate': 100,
                              'FNO': 10,
                              'UNet': 2,
                              'RNN': 10,
                              'DeepONet': 400}

    # Model configs
    config.models = [
        'ditto',
        'ditto_point',
        f'ditto_subsampled_X{config.subsample}',
        f'ditto_point_subsampled_X{config.subsample}_gate',
        f'ditto_point_subsampled_X{config.subsample}',
        'FNO',
        'UNet',
        'RNN',
    ]

    config.pi_loss = False
    config.switch_epoch = None
    config.unet_dim = 16
    config.lr = 0.001 * 0.2
    config.weight_decay = 0.0001
    config.epochs = {'ditto': 500,
                     'ditto_point': 500,
                     f'ditto_subsampled_X{config.subsample}': 500,
                     f'ditto_point_subsampled_X{config.subsample}': 500,
                     f'ditto_point_subsampled_X{config.subsample}_gate': 500,
                     'FNO': 500,
                     'UNet': 500,
                     'RNN': 500,
                     'DeepONet': 50000}
    config.train = {'ditto': False,
                    'ditto_point': False,
                    f'ditto_subsampled_X{config.subsample}': False,
                    f'ditto_point_subsampled_X{config.subsample}': False,
                    f'ditto_point_subsampled_X{config.subsample}_gate': False,
                    'FNO': False,
                    'UNet': False,
                    'RNN': False,
                    'DeepONet': False}

    # Extrapolation configs
    config.extrapolation = False
    config.look_back = 1
    config.look_forward = 20 + 1

    # Plotting configs
    config.plot = False
    config.save_every = 1
    config.plot_nt = {'200': True, '50': True}
    return config
