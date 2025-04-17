import ml_collections
import numpy as np


def get_climate_3d_configs():
    config = ml_collections.ConfigDict()

    # Data configs
    config.scenario = "climate_3d"
    config.N = 260
    config.T = 1
    config.pressure_levels = [10, 20, 30, 50, 70, 
                              100, 150, 200, 250, 
                              300, 400, 500, 600, 
                              700, 850, 925]
    config.noise_level = 0.

    # Grid configs
    config.nx = 72
    config.ny = 144
    config.nz = 16
    config.output_shape = (config.nx, config.ny, config.nz)
    config.time_steps_train = 30
    config.time_steps_inference = 30
    config.time_steps_inference_list = [30]
    config.sr_factors = [1]

    # Data loading configs
    config.ntrain = None
    config.nval = None
    config.ntest = None
    config.use_same_validation_and_test = False
    config.normalize = True
    config.dtype = 'float32'
    config.subsample = 10

    config.batch_size_train = {'ditto': 1,
                               'ditto_point': 1,
                               f'ditto_subsampled_X{config.subsample}': 10,
                               f'ditto_point_subsampled_X{config.subsample}': 1,
                               f'ditto_point_subsampled_X{config.subsample}_gate': 1,
                               'DeepONet': 1}
    config.batch_size_val = {'ditto': 1,
                             'ditto_point': 1,
                             f'ditto_subsampled_X{config.subsample}': 10,
                             f'ditto_point_subsampled_X{config.subsample}': 1,
                             f'ditto_point_subsampled_X{config.subsample}_gate': 1,
                             'DeepONet': 1}
    config.batch_size_test = {'ditto': 10,
                              'ditto_point': 1,
                              f'ditto_subsampled_X{config.subsample}': 10,
                              f'ditto_point_subsampled_X{config.subsample}': 1,
                              f'ditto_point_subsampled_X{config.subsample}_gate': 1,
                              'DeepONet': 1}

    # Model configs
    config.models = [
                     f'ditto_subsampled_X{config.subsample}',
                     'ditto',
                     'ditto_point',
                    #  f'ditto_point_subsampled_X{config.subsample}',
                    #  f'ditto_point_subsampled_X{config.subsample}_gate',
                     ]

    config.pi_loss = False
    config.switch_epoch = None
    config.unet_dim = 16
    config.lr = 0.001
    config.weight_decay = 0.0001
    config.epochs = {'ditto': 100,
                     'ditto_point': 100,
                     f'ditto_subsampled_X{config.subsample}': 100,
                     f'ditto_point_subsampled_X{config.subsample}': 100,
                     f'ditto_point_subsampled_X{config.subsample}_gate': 100,
                     'FNO': 100,
                     'UNet': 100,
                     'DeepONet': 50000}
    config.train = {'ditto': False,
                    'ditto_point': False,
                    f'ditto_subsampled_X{config.subsample}': False,
                    f'ditto_point_subsampled_X{config.subsample}': False,
                    f'ditto_point_subsampled_X{config.subsample}_gate': False,
                    'FNO': False,
                    'UNet': False,
                    'DeepONet': False}

    # Extrapolation configs
    config.extrapolation = False
    config.look_back = 1
    config.look_forward = 20 + 1

    # Plotting configs
    config.plot = False
    config.save_every = 1
    config.plot_nt = {'30': True}
    return config
