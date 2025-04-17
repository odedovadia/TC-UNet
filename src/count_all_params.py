from src.utils.utils import (get_logger, count_params)
from src.utils.architecture_builder import build_archtitecture
from src.configs import (BURGERS_CONFIGS, NAVIER_STOKES_CONFIGS,
                         WAVE_EQUATION_2D_CONFIGS, WAVE_EQUATION_3D_CONFIGS,
                         SHALLOW_WATER_CONFIGS, HYPERSONICS_CONFIGS,
                         HYPERSONICS_TIME_CONFIGS,
                         CLIMATE_2D_CONFIGS, CLIMATE_3D_CONFIGS)
import numpy as np
import pandas as pd
import torch
torch.manual_seed(1)


def burgers_1():
    return BURGERS_CONFIGS


def burgers_2():
    configs = BURGERS_CONFIGS
    configs.viscosity = 1e-03
    configs.nx = 256
    configs.output_shape = (configs.nx,)
    configs.sr_factors = [0.5, 1, 2, 4]
    return configs


def burgers_3():
    configs = BURGERS_CONFIGS
    configs.viscosity = 1e-03
    configs.nx = 256
    configs.output_shape = (configs.nx,)
    configs.T = 2
    return configs


def ns_20():
    return NAVIER_STOKES_CONFIGS


def ns_2007():
    configs = NAVIER_STOKES_CONFIGS
    configs.viscosity = 1e-5
    configs.T = 20
    configs.reynolds_number = int(np.sqrt(0.1) / ((2 * np.pi) ** (3/2) * configs.viscosity))
    configs.train['FNO'] = False
    return configs


def ns_2007_large():
    configs = NAVIER_STOKES_CONFIGS
    configs.viscosity = 1e-5
    configs.T = 20
    configs.reynolds_number = int(np.sqrt(0.1) / ((2 * np.pi) ** (3/2) * configs.viscosity))
    configs.N = 5000

    configs.train['FNO'] = True
    return configs


def wave2d():
    return WAVE_EQUATION_2D_CONFIGS


def wave3d():
    return WAVE_EQUATION_3D_CONFIGS


def shallow_water():
    return SHALLOW_WATER_CONFIGS


def hypersonics():
    return HYPERSONICS_CONFIGS


def hypersonics_time():
    return HYPERSONICS_TIME_CONFIGS


def climate_2d_level_10():
    configs = CLIMATE_2D_CONFIGS
    configs.pressure_level = 10
    return configs


def climate_2d_level_1000():
    configs = CLIMATE_2D_CONFIGS
    configs.pressure_level = 1000
    return configs


def climate_3d():
    return CLIMATE_3D_CONFIGS


CONFIG_RUNS = [burgers_1, burgers_2, burgers_3, ns_20, ns_2007, ns_2007_large,
               wave2d, wave3d, shallow_water, hypersonics,
               climate_2d_level_10, climate_2d_level_1000, climate_3d]


if __name__ == "__main__":
    logger = get_logger()

    full_params = {}
    for cfg in CONFIG_RUNS:
        params = {}
        configs = cfg()
        configs.models = [
            'ditto',
            'ditto_point',
            f'ditto_point_subsampled_X{configs.subsample}_gate',
            'FNO',
            'UNet'
        ]

        # Make sure that all configs.train models are set to False
        for model in configs.models:
            configs.train[model] = False

        for model_name in configs.models:
            try:
                model = build_archtitecture(model_type=model_name, configs=configs, logger=logger, mode='train')
                params[model_name] = count_params(model)
            except:
                params[model_name] = None
            torch.cuda.empty_cache()
        full_params[cfg.__name__] = params

    df = pd.DataFrame(full_params)
    mini_df = df[['burgers_1', "ns_20", "wave3d"]].drop("ditto_point_subsampled_X1_gate")
    cols_rename_dict = {
        "burgers_1": "1-D",
        "ns_20": "2-D",
        "wave3d": "3-D"
    }
    mini_df = mini_df.rename(columns=cols_rename_dict)

    rows_rename_dict = {
        "ditto": "DiTTO",
        "ditto_point": "DiTTO-Point",
        "ditto_point_subsampled_X10_gate": "DiTTO-Point-gate",
        "FNO": "FNO",
        "UNet": "UNet"
    }
    mini_df = mini_df.rename(index=rows_rename_dict)
    mini_df = mini_df.fillna("N/A")
    print(mini_df.to_latex(index=True, float_format="{:0.4f}".format,  column_format="ccccc"))

