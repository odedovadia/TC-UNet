import ml_collections
import numpy as np


def get_hypersonics_time_configs():
    config = ml_collections.ConfigDict()

    # Data configs
    config.scenario = "hypersonics_time"
    config.N = 61
    config.T = 0.04
    config.nx = 256
    config.ny = 256
    config.noise_level = 0

    # Grid configs
    config.points = 65536
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

    config.batch_size_train = {'ditto': 10,
                               'ditto_point': 5,
                               f'ditto_subsampled_X{config.subsample}': 100,
                               f'ditto_point_subsampled_X{config.subsample}': 50,
                               f'ditto_point_subsampled_X{config.subsample}_gate': 50,
                               'FNO': 1,
                               'UNet': 1,
                               'RNN': 1,
                               'DeepONet': 1}
    config.batch_size_val = {'ditto': 5,
                             'ditto_point': 5,
                             f'ditto_subsampled_X{config.subsample}': 5,
                             f'ditto_point_subsampled_X{config.subsample}': 5,
                             f'ditto_point_subsampled_X{config.subsample}_gate': 5,
                             'FNO': 1,
                             'UNet': 1,
                             'RNN': 1,
                             'DeepONet': 1}
    config.batch_size_test = {'ditto': 5,
                              'ditto_point': 5,
                              f'ditto_subsampled_X{config.subsample}': 5,
                              f'ditto_point_subsampled_X{config.subsample}': 5,
                              f'ditto_point_subsampled_X{config.subsample}_gate': 5,
                              'FNO': 1,
                              'UNet': 1,
                              'RNN': 1,
                              'DeepONet': 1}

    # Model configs
    config.models = [
        'ditto',
        'ditto_point',
        f'ditto_subsampled_X{config.subsample}',
        f'ditto_point_subsampled_X{config.subsample}',
        f'ditto_point_subsampled_X{config.subsample}_gate',
        'RNN',
        # 'FNO',
        # 'UNet'  Failed on a6000 due to OEM with batch 1
    ]

    config.pi_loss = False
    config.switch_epoch = None
    config.unet_dim = 16
    config.lr = 5e-03 / 10
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
    config.plot_nt = {'10': True, '50': True, '200': True}
    return config
