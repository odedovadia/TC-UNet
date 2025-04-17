import torch
from neuralop.models import TFNO2d, TFNO3d
from modulus.models.rnn.rnn_one2many import One2ManyRNN
from src.ditto.unet_1d import Unet1D
from src.ditto.unet_2d import Unet2D
from src.ditto.unet_3d import Unet3D


def build_archtitecture(model_type, configs, logger, mode='train'):
    if 'point' in model_type:
        dim_head = 32
        attention_heads = 4
        use_gate = True if 'gate' in model_type else False
        if configs.scenario == 'burgers':
            model = Unet1D(dim=configs.unet_dim,
                           dim_mults=(1, 2, 4, 8),
                           channels=1,
                           out_dim=1,
                           attention_heads=attention_heads,
                           dim_head=dim_head,
                           pointwise=True,
                           seq_len=configs.nx,
                           latent_dim=128,
                           space_dim=1,
                           use_gate=use_gate)
            logger.info('Using PointNet')

        elif configs.scenario in ['navier_stokes', 'wave_equation_2d',
                                  'shallow_water', 'hypersonics',
                                  'hypersonics_time', 'climate_2d']:
            model = Unet1D(dim=configs.unet_dim,
                           dim_mults=(1, 2, 4, 8),
                           channels=1,
                           out_dim=1,
                           attention_heads=attention_heads,
                           dim_head=dim_head,
                           pointwise=True,
                           seq_len=configs.nx * configs.ny,
                           latent_dim=128,
                           space_dim=2,
                           use_gate=use_gate)
            logger.info('Using PointNet')
        elif configs.scenario in ['wave_equation_3d', 'climate_3d']:
            model = Unet1D(dim=configs.unet_dim,
                           dim_mults=(1, 2, 4, 8),
                           channels=1,
                           out_dim=1,
                           attention_heads=attention_heads,
                           dim_head=dim_head,
                           pointwise=True,
                           seq_len=configs.nx * configs.ny * configs.nz,
                           latent_dim=128,
                           space_dim=3,
                           use_gate=use_gate)
            logger.info('Using PointNet')

    elif 'ditto' in model_type:
        dim_head = None if model_type == 'ditto_no_attention' else 32
        attention_heads = None if model_type == 'ditto_no_attention' else 4
        if configs.scenario == 'burgers':
            model = Unet1D(dim=configs.unet_dim,
                           dim_mults=(1, 2, 4, 8),
                           channels=1,
                           out_dim=1,
                           attention_heads=attention_heads,
                           dim_head=dim_head)
            logger.info('Using Unet1D')

        elif configs.scenario in ['navier_stokes', 'wave_equation_2d',
                                  'shallow_water', 'hypersonics',
                                  'hypersonics_time', 'climate_2d']:
            model = Unet2D(dim=configs.unet_dim,
                           dim_mults=(1, 2, 4, 8),
                           channels=1,
                           out_dim=1,
                           attention_heads=4,
                           dim_head=32)
            logger.info('Using Unet2D')

        elif configs.scenario in ['wave_equation_3d', 'climate_3d']:
            model = Unet3D(dim=configs.unet_dim,
                           dim_mults=(1, 2, 4, 8),
                           channels=1,
                           out_dim=1)
            logger.info('Using Unet3D')

    elif model_type == 'UNet':
        if configs.scenario == 'burgers':
            model = Unet2D(dim=configs.unet_dim,
                           dim_mults=(1, 2, 4, 8),
                           channels=3,
                           out_dim=1,
                           attention_heads=4,
                           dim_head=32)
            logger.info('Using Unet2D')
        elif configs.scenario in ['navier_stokes', 'wave_equation_2d',
                                  'shallow_water', 'hypersonics',
                                  'hypersonics_time', 'climate_2d']:
            model = Unet3D(dim=configs.unet_dim,
                           dim_mults=(1, 2, 4, 8),
                           channels=4,
                           out_dim=1)
            logger.info('Using Unet3D')

    elif model_type == 'RegUNet':
        if configs.scenario == 'burgers':
            model = Unet2D(dim=configs.unet_dim,
                           dim_mults=(1, 2, 4, 8),
                           channels=3,
                           out_dim=1,
                           attention_heads=None,
                           dim_head=None)
            logger.info('Using Unet2D')
    
    elif 'RNN' in model_type:
        if configs.scenario in ['navier_stokes', 'wave_equation_2d',
                                'shallow_water', 'hypersonics', 'hypersonics_time',
                                'climate_2d']:      
            model = One2ManyRNN(
                input_channels=1,
                dimension=2,
                nr_tsteps=configs.time_steps_inference,
                nr_downsamples=3,
                nr_residual_blocks=2,
                nr_latent_channels=32,
            )
            logger.info('Using One2Many RNN2D')

    elif model_type == 'FNO':
        if configs.scenario == 'burgers':
            model = TFNO2d(18, 18, hidden_channels=32, in_channels=3, out_channels=1,
                           lifting_channels=128, projection_channels=128)
            logger.info('Using FNO2D')
        elif configs.scenario in ['navier_stokes', 'wave_equation_2d',
                                  'shallow_water', 'hypersonics',
                                  'climate_2d']:
            model = TFNO3d(12, 12, 12, hidden_channels=20, in_channels=4, out_channels=1)
            logger.info('Using FNO3D')
        elif configs.scenario == 'hypersonics_time':
            model = TFNO3d(18, 18, 18, hidden_channels=32, in_channels=4, out_channels=1)
            logger.info('Using FNO3D')
        elif configs.scenario == 'wave_equation_3d':
            # TODO add (3 + 1)d FNO
            raise NotImplementedError
        elif configs.scenario == 'hypersonics':
            model = TFNO2d(18, 18, hidden_channels=32, in_channels=configs.T, out_channels=1,
                           lifting_channels=128, projection_channels=128)
            logger.info('Using FNO2D')

    if model_type == 'FNO':
        model = model.cuda()
    else:
        dtype = eval('torch.' + configs.dtype)
        model = model.cuda().to(dtype)
    return model
