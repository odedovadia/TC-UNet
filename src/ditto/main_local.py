from src.utils.utils import (save_metrics_to_text, get_logger, save_sr_metrics_to_text, count_params)
from src.utils.architecture_builder import build_archtitecture
from src.ditto.data_loader import DataHandler
from src.ditto.trainer import Trainer
from src.ditto.tester import Tester
from src.configs import (BURGERS_CONFIGS, NAVIER_STOKES_CONFIGS,
                         WAVE_EQUATION_2D_CONFIGS, WAVE_EQUATION_3D_CONFIGS,
                         SHALLOW_WATER_CONFIGS, HYPERSONICS_CONFIGS,
                         HYPERSONICS_TIME_CONFIGS,
                         CLIMATE_2D_CONFIGS, CLIMATE_3D_CONFIGS)
import os
from copy import copy
import wandb
import numpy as np
import torch
torch.manual_seed(1)


def train(configs, logger):
    for model_name in configs.models:
        logger.info(f'Running {model_name} model')
        if configs.train[model_name]:            
            
            if "RNN" in model_name:
                time_steps_train = copy(configs.time_steps_train)
                configs.time_steps_train = configs.time_steps_inference

            reader = DataHandler(configs=configs, model_name=model_name, logger=logger)
            train_loader, val_loader, _ = reader.load_data()

            model = build_archtitecture(model_type=model_name, configs=configs, logger=logger, mode='train')
            logger.info(f'Training {model_name} model')
            logger.info(f'Number of parameters: {count_params(model)}')
            group = configs.scenario
            if 'navier_stokes' in configs.scenario:
                group += '_Re_' + str(configs.reynolds_number)
                if configs.N > 1000:
                    group += "_large"
            elif 'burgers' in configs.scenario:
                group += '_v_' + str(configs.viscosity) + '_T_' + str(int(configs.T))
            
            if configs.extrapolation:
                group += '_extrapolation'
                model_name += '_extrapolation'

            if configs.noise_level > 0:
                group += '_noise_' + str(configs.noise_level)
                model_name += '_noise_' + str(configs.noise_level)

            wandb.init(project="DiTTO-RNN",
                       config=configs,
                       group=group,
                       name=model_name)

            trainer = Trainer(model=model,
                              configs=configs,
                              logger=logger,
                              model_name=model_name,
                              y_normalizer=reader.y_normalizer,
                              grid=reader.grid_train)
            trainer.train(train_loader, val_loader)
            logger.info(f'Done training {model_name} model')

            if "RNN" in model_name:
                configs.time_steps_train = time_steps_train
            wandb.finish()
        torch.cuda.empty_cache()


def test(configs, logger, zero_shot_noise=False):
    final_metrics = {}
    for inference_steps in configs.time_steps_inference_list:
        logger.info(f'Running inference for {inference_steps} steps')
        configs.time_steps_inference = inference_steps
        for model_name in configs.models:
            logger.info(f'Testing {model_name} model')

            reader = DataHandler(configs=configs, model_name=model_name, logger=logger)
            test_loader = reader.load_data()[2]

            model = build_archtitecture(model_type=model_name, configs=configs, logger=logger, mode='test')
            if "RNN" in model_name:
                model.nr_tsteps = configs.time_steps_inference_list[-1]
                time_steps_train = copy(configs.time_steps_train)
                configs.time_steps_train = model.nr_tsteps

            if not zero_shot_noise:
                if configs.noise_level > 0:
                    model_name += '_noise_' + str(configs.noise_level)
            
            if configs.extrapolation:
                model_name += '_extrapolation'

            tester = Tester(model, configs=configs, logger=logger,
                            y_normalizer=reader.y_normalizer, model_name=model_name, grid=reader.grid_test)

            metric = tester.evaluate(test_loader)
            key = list(metric.keys())[0]
            val = list(metric.values())[0]
            val_key = list(val.keys())[0]

            if key not in final_metrics:
                final_metrics[key] = {}
            final_metrics[key][val_key] = val[val_key]

            if "RNN" in model_name:
                configs.time_steps_train = time_steps_train
            
            torch.cuda.empty_cache()
    metrics = save_metrics_to_text(final_metrics, path_model=tester.path_model, noise=configs.noise_level)
    logger.info(metrics)


def test_sr(configs, logger):
    final_metrics = {}
    sr_metrics = {model_name:
                  {step: {} for step in configs.time_steps_inference_list}
                  for model_name in configs.models}

    for inference_steps in configs.time_steps_inference_list:
        logger.info(f'Running inference for {inference_steps} steps')
        configs.time_steps_inference = inference_steps
        for model_name in configs.models:
            logger.info(f'Testing {model_name} model')

            model = build_archtitecture(model_type=model_name, configs=configs, logger=logger,  mode='test')
            original_size = copy(configs.nx)
            for sr in configs.sr_factors:
                configs.nx = int(original_size * sr)
                reader = DataHandler(configs=configs, model_name=model_name, logger=logger)
                test_loader = reader.load_data()[2]

                configs.nx = original_size
                tester = Tester(model, configs=configs, logger=logger,
                                y_normalizer=reader.y_normalizer, model_name=model_name)

                tester.path_figs = tester.path_figs.replace(f'{original_size}', f'{original_size * sr}')

                metric = tester.evaluate(test_loader)
                key = list(metric.keys())[0]
                val = list(metric.values())[0]
                val_key = list(val.keys())[0]

                if key not in final_metrics:
                    final_metrics[key] = {}
                final_metrics[key][val_key] = val[val_key]
                sr_metrics[model_name][inference_steps][original_size * sr] = val[val_key]

                torch.cuda.empty_cache()
    save_sr_metrics_to_text(sr_metrics, tester.path_model, model_name)
    metrics = save_metrics_to_text(final_metrics, path_model=tester.path_model)
    logger.info(metrics)


def run(configs, logger, zero_shot_noise=False):
    logger.info(f'Running {configs.scenario} scenario')
    train(configs=configs, logger=logger)
    test(configs=configs, logger=logger, zero_shot_noise=zero_shot_noise)
    # test_sr(configs=configs, logger=logger)
    logger.info('Done')


# RUN_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
# RUN_ID = -1


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
    configs.reynolds_number = int(np.sqrt(0.1) / ((2     * np.pi) ** (3/2) * configs.viscosity))
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

    configs = ns_20()
    run(configs=configs, logger=logger)
