import ml_collections
import numpy as np


def get_shallow_water_configs():
    config = ml_collections.ConfigDict()

    # Data configs
    config.scenario = "shallow_water"
    config.N = 300
    config.T = 72
    config.noise_level = 0.

    # Grid configs
    config.nx = 256 * 1
    config.ny = 256 * 1
    config.output_shape = (config.nx, config.ny)
    config.time_steps_train = 18
    config.time_steps_inference = 72
    config.time_steps_inference_list = [4, 8, 18, 36, 72]
    config.sr_factors = [1]

    # Data loading configs
    config.ntrain = None
    config.nval = None
    config.ntest = None
    config.use_same_validation_and_test = False
    config.normalize = True
    config.dtype = 'float32'
    config.subsample = 10

    config.batch_size_train = {'ditto': 20,
                               'ditto_point': 20,
                               f'ditto_subsampled_X{config.subsample}': 20,
                               f'ditto_point_subsampled_X{config.subsample}': 20,
                               f'ditto_point_subsampled_X{config.subsample}_gate': 20,
                               'FNO': 5,
                               'UNet': 1,
                               'DeepONet': 400}
    config.batch_size_val = {'ditto': 20,
                             'ditto_point': 20,
                             f'ditto_subsampled_X{config.subsample}': 20,
                             f'ditto_point_subsampled_X{config.subsample}': 20,
                             f'ditto_point_subsampled_X{config.subsample}_gate': 20,
                             'FNO': 5,
                             'UNet': 1,
                             'DeepONet': 400}
    config.batch_size_test = {'ditto': 20,
                              'ditto_point': 20,
                              f'ditto_subsampled_X{config.subsample}': 20,
                              f'ditto_point_subsampled_X{config.subsample}': 20,
                              f'ditto_point_subsampled_X{config.subsample}_gate': 20,
                              'FNO': 5,
                              'UNet': 1,
                              'DeepONet': 400}

    # Model configs
    config.models = [
        'ditto',
        'ditto_point',
        f'ditto_subsampled_X{config.subsample}',
        f'ditto_point_subsampled_X{config.subsample}',
        f'ditto_point_subsampled_X{config.subsample}_gate',
        'FNO',
        # 'UNet'
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
    config.plot_nt = {'200': True, '50': True}
    return config
