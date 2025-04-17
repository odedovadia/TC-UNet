import ml_collections


def get_wave_equation_3d_configs():
    config = ml_collections.ConfigDict()

    # Data configs
    config.scenario = "wave_equation_3d"
    config.velocity = 1.0
    config.N = 1000
    config.T = 2
    config.noise_level = 0.

    # Grid configs
    config.nx = 32
    config.ny = 32
    config.nz = 32
    config.output_shape = (config.nx, config.ny, config.nz)
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

    config.batch_size_train = {'ditto': 1,
                               'ditto_point': 10,
                               f'ditto_subsampled_X{config.subsample}': 100,
                               f'ditto_point_subsampled_X{config.subsample}': 100,
                               f'ditto_point_subsampled_X{config.subsample}_gate': 100,
                               'DeepONet': 1}
    config.batch_size_val = {'ditto': 1,
                             'ditto_point': 10,
                             f'ditto_subsampled_X{config.subsample}': 10,
                             f'ditto_point_subsampled_X{config.subsample}': 10,
                             f'ditto_point_subsampled_X{config.subsample}_gate': 10,
                             'DeepONet': 1}
    config.batch_size_test = {'ditto': 10,
                              'ditto_point': 10,
                              f'ditto_subsampled_X{config.subsample}': 10,
                              f'ditto_point_subsampled_X{config.subsample}': 10,
                              f'ditto_point_subsampled_X{config.subsample}_gate': 10,
                              'DeepONet': 1}

    # Model configs
    config.models = [  # 'ditto',
        # 'ditto_point',
        f'ditto_point_subsampled_X{config.subsample}',
        f'ditto_point_subsampled_X{config.subsample}_gate',
        f'ditto_subsampled_X{config.subsample}',
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
                     'DeepONet': 50000}
    config.train = {'ditto': False,
                    'ditto_point': False,
                    f'ditto_subsampled_X{config.subsample}': False,
                    f'ditto_point_subsampled_X{config.subsample}': False,
                    f'ditto_point_subsampled_X{config.subsample}_gate': False,
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
