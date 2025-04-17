import ml_collections
import numpy as np


def get_hypersonics_configs():
    config = ml_collections.ConfigDict()

    # Data configs
    config.scenario = "hypersonics"
    config.N = 1
    config.T = 10
    config.nx = 256
    config.ny = 256
    config.noise_level = 0.

    # Grid configs
    config.points = 65536
    config.output_shape = (config.nx, config.ny)
    config.time_steps_train = 21
    config.time_steps_inference = 20
    config.time_steps_inference_list = [20]
    config.sr_factors = [1]

    # Data loading configs
    config.ntrain = None
    config.nval = None
    config.ntest = None
    config.use_same_validation_and_test = False
    config.normalize = True
    config.dtype = 'float32'
    config.subsample = 1

    config.batch_size_train = {'ditto': 1,
                               'ditto_point': 1,
                               f'ditto_point_subsampled_X{config.subsample}_gate': 1,
                               'FNO': 1,
                               'UNet': 1,
                               'DeepONet': 1}
    config.batch_size_val = {'ditto': 1,
                             'ditto_point': 1,
                             f'ditto_point_subsampled_X{config.subsample}_gate': 1,
                             'FNO': 1,
                             'UNet': 1,
                             'DeepONet': 1}
    config.batch_size_test = {'ditto': 1,
                              'ditto_point': 1,
                              f'ditto_point_subsampled_X{config.subsample}_gate': 1,
                              'FNO': 1,
                              'UNet': 1,
                              'DeepONet': 1}

    # Model configs
    config.models = [
                    'ditto',
                    'ditto_point',
                    #  f'ditto_point_subsampled_X{config.subsample}_gate',
                    'FNO',
                    #  'UNet'
                     ]

    config.pi_loss = False
    config.switch_epoch = None
    config.unet_dim = 16
    config.lr = 0.001
    config.weight_decay = 0.0001
    config.epochs = {'ditto': 5000,
                     'ditto_point': 5000,
                     f'ditto_point_subsampled_X{config.subsample}_gate': 500,
                     'FNO': 5000,
                     'UNet': 500,
                     'DeepONet': 50000}
    config.train = {'ditto': False,
                    'ditto_point': False,
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
    config.plot_nt = {'20': True}
    return config
