import ml_collections


def get_wave_equation_2d_configs():
    config = ml_collections.ConfigDict()

    # Data configs
    config.scenario = "wave_equation_2d"
    config.velocity = 1.0
    config.N = 1000
    config.T = 2
    config.noise_level = 0.

    # Grid configs
    config.nx = 64
    config.ny = 64
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
                               'FNO': 40,
                               'UNet': 2,
                               'RNN': 40,
                               'DeepONet': 400}
    config.batch_size_val = {'ditto': 400,
                             'ditto_point': 400,
                             f'ditto_subsampled_X{config.subsample}': 400,
                             f'ditto_point_subsampled_X{config.subsample}': 400,
                             f'ditto_point_subsampled_X{config.subsample}_gate': 400,
                             'FNO': 10,
                             'UNet': 2,
                             'RNN': 10,
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
    config.models = ['ditto',
                     'ditto_point',
                     f'ditto_subsampled_X{config.subsample}',
                     f'ditto_point_subsampled_X{config.subsample}',
                     f'ditto_point_subsampled_X{config.subsample}_gate',
                     'FNO',
                     'RNN',
                     'UNet'
                     ]
    config.pi_loss = False
    config.switch_epoch = 10
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
