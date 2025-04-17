import ml_collections


def get_burgers_configs():
    config = ml_collections.ConfigDict()

    # Data configs
    config.scenario = "burgers"
    config.viscosity = 1e-02
    config.N = 1000
    config.T = 1
    config.noise_level = 0.

    # Grid configs
    config.nx = 128
    config.output_shape = (config.nx,)
    config.time_steps_train = 50
    config.time_steps_inference = 200
    config.time_steps_inference_list = [10, 20, 50, 100, 200]
    config.sr_factors = [1, 2, 4, 8]

    # Data loading configs
    config.ntrain = None
    config.nval = None
    config.ntest = None
    config.use_same_validation_and_test = False
    config.normalize = True
    config.dtype = 'float32'
    config.subsample = 10

    config.batch_size_train = {'ditto': 800,
                               'ditto_point': 800,
                               'ditto_no_attention': 800,
                               f'ditto_subsampled_X{config.subsample}': 800,
                               f'ditto_point_subsampled_X{config.subsample}': 800,
                               f'ditto_point_subsampled_X{config.subsample}_gate': 800,
                               'FNO': 100,
                               'UNet': 100,
                               'DeepONet': 8000}
    config.batch_size_val = {'ditto': 2000,
                             'ditto_point': 2000,
                             'ditto_no_attention': 2000,
                             f'ditto_subsampled_X{config.subsample}': 2000,
                             f'ditto_point_subsampled_X{config.subsample}': 2000,
                             f'ditto_point_subsampled_X{config.subsample}_gate': 2000,
                             'FNO': 100,
                             'UNet': 100,
                             'DeepONet': 80}
    config.batch_size_test = {'ditto': 100,
                              'ditto_point': 100,
                              'ditto_no_attention': 100,
                              f'ditto_subsampled_X{config.subsample}': 100,
                              f'ditto_point_subsampled_X{config.subsample}': 100,
                              f'ditto_point_subsampled_X{config.subsample}_gate': 100,
                              'FNO': 100,
                              'UNet': 100,
                              'DeepONet': 80}

    # Model configs
    config.models = [
                     'ditto',
                     # 'ditto_point',
                     f'ditto_subsampled_X{config.subsample}',
                     # f'ditto_point_subsampled_X{config.subsample}_gate',
                     # f'ditto_point_subsampled_X{config.subsample}',
                     'FNO',
                     'UNet'
                    ]

    config.pi_loss = False
    config.switch_epoch = None
    config.unet_dim = 16
    config.lr = 0.001
    config.weight_decay = 0.0001
    config.epochs = {'ditto': 500,
                     'ditto_point': 500,
                     'ditto_no_attention': 500,
                     f'ditto_subsampled_X{config.subsample}': 500,
                     f'ditto_point_subsampled_X{config.subsample}': 500,
                     f'ditto_point_subsampled_X{config.subsample}_gate': 500,
                     'FNO': 500,
                     'UNet': 500,
                     'DeepONet': 20000}
    config.train = {'ditto': True,
                    'ditto_point': False,
                    'ditto_no_attention': False,
                    f'ditto_subsampled_X{config.subsample}': True,
                    f'ditto_point_subsampled_X{config.subsample}': False,
                    f'ditto_point_subsampled_X{config.subsample}_gate': False,
                    'FNO': True,
                    'UNet': True,
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
