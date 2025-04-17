import ml_collections
import numpy as np


def get_climate_2d_configs():
    config = ml_collections.ConfigDict()

    # Data configs
    config.scenario = "climate_2d"
    config.N = 260
    config.T = 1
    config.pressure_level = 10
    config.noise_level = 0.

    # Grid configs
    config.nx = 72
    config.ny = 144
    config.output_shape = (config.nx, config.ny)
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

    config.batch_size_train = {'ditto': 80,
                               'ditto_point': 80,
                               f'ditto_subsampled_X{config.subsample}': 800,
                               f'ditto_point_subsampled_X{config.subsample}': 800,
                               f'ditto_point_subsampled_X{config.subsample}_gate': 800,
                               'FNO': 10,
                               'UNet': 2,
                               'DeepONet': 400}
    config.batch_size_val = {'ditto': 80,
                             'ditto_point': 80,
                             f'ditto_subsampled_X{config.subsample}': 80,
                             f'ditto_point_subsampled_X{config.subsample}': 80,
                             f'ditto_point_subsampled_X{config.subsample}_gate': 80,
                             'FNO': 10,
                             'UNet': 2,
                             'DeepONet': 400}
    config.batch_size_test = {'ditto': 1,
                              'ditto_point': 1,
                              f'ditto_subsampled_X{config.subsample}': 1,
                              f'ditto_point_subsampled_X{config.subsample}': 1,
                              f'ditto_point_subsampled_X{config.subsample}_gate': 1,
                              'FNO': 1,
                              'UNet': 2,
                              'DeepONet': 400}

    # Model configs
    config.models = [f'ditto_subsampled_X{config.subsample}',
                     'ditto',
                     'ditto_point',
                     'FNO',
                     f'ditto_point_subsampled_X{config.subsample}',
                     f'ditto_point_subsampled_X{config.subsample}_gate',
                     'UNet'
                     ]

    config.pi_loss = False
    config.switch_epoch = None
    config.unet_dim = 16
    config.lr = 0.001
    config.weight_decay = 0.0001
    config.epochs = {'ditto': 500,
                     'ditto_point': 500,
                     f'ditto_subsampled_X{config.subsample}': 500,
                     f'ditto_point_subsampled_X{config.subsample}': 500,
                     f'ditto_point_subsampled_X{config.subsample}_gate': 500,
                     'FNO': 500,
                     'UNet': 500,
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
    config.plot = True
    config.save_every = 1
    config.plot_nt = {'30': True}
    return config
