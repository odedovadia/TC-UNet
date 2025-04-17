import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from torchmetrics.functional import r2_score
from tqdm import tqdm

from src.utils.utils import (LpLoss, count_params, rmse_loss)
from src.utils.plotter import Plotter


class Tester:
    def __init__(self, model, configs, logger, y_normalizer=None, model_name="ditto", grid=None) -> None:
        self.model = model.cuda()
        self.configs = configs
        self.logger = logger
        self.y_normalizer = y_normalizer
        self.model_name = model_name
        self.grid = grid['grid_space'].permute(1, 0)[None, ...].cuda()

        self.path_figs = os.path.join('..', 'outputs', 'figs', configs.scenario)

        self.path_model = os.path.join('..', 'outputs', 'models', configs.scenario)
        if not os.path.exists(self.path_model):
            os.makedirs(self.path_model)

        loaded_model_name = self._get_model_name(configs)

        self.path_figs = os.path.join(
            self.path_figs,
            loaded_model_name + f'_to_{configs.time_steps_inference}',
            model_name)
        self.logger.info(f'Saving figures to {self.path_figs}')

        loaded_model_name = os.path.join(model_name, loaded_model_name)

        self.path_model = os.path.join(
            self.path_model, loaded_model_name + '.pth')
        if not os.path.exists(self.path_figs) and configs.plot:
            os.makedirs(self.path_figs)

        self.logger.info(f'Loading model from {self.path_model}')
        self.model.load_state_dict(torch.load(self.path_model))
        if configs.scenario == 'climate_3d':
            self.plotter = None
        else:
            self.plotter = Plotter(
                configs=configs, logger=logger, path_figs=self.path_figs, save_every=configs.save_every)

    def _get_model_name(self, configs):
        if configs.scenario == 'navier_stokes':
            loaded_model_name = f'NavierStokes_v_{configs.viscosity}_N_{configs.N}_T_{configs.T}_nx_{configs.nx}_ny_{configs.ny}'
        elif configs.scenario == 'wave_equation_2d':
            loaded_model_name = f'WaveEq_N_{configs.N}_T_{configs.T}_velocity_{configs.velocity}_nx_{configs.nx}_ny_{configs.ny}'
        elif configs.scenario == 'wave_equation_3d':
            loaded_model_name = (f'WaveEq_N_{configs.N}_T_{configs.T}_velocity_{configs.velocity}' +
                                 f'_nx_{configs.nx}_ny_{configs.ny}_nz_{configs.nz}')
        elif configs.scenario == 'burgers':
            loaded_model_name = f'Burgers_v_{configs.viscosity}_N_{configs.N}_T_{configs.T}_size_{configs.nx}'
        elif configs.scenario == 'climate_2d':
            loaded_model_name = f"Climate_2d_N_{configs.N}_nx_{configs.nx}_ny_{configs.ny}_level_{configs.pressure_level}"
        elif configs.scenario == 'climate_3d':
            loaded_model_name = f"Climate_3d_N_{configs.N}_nx_{configs.nx}_ny_{configs.ny}_nz_{configs.nz}"
        elif configs.scenario == 'shallow_water':
            loaded_model_name = f'Shallow_Water_N_{configs.N}_T_{configs.T}_nx_{configs.nx}_ny_{configs.ny}'
        elif configs.scenario == 'hypersonics':
            loaded_model_name = f'Hypersonics_N_{configs.N}_M_{configs.T}_nx_{configs.nx}_ny_{configs.ny}'
        elif configs.scenario == 'hypersonics_time':
            loaded_model_name = f'Hypersonics_time_N_{configs.N}_T_{configs.T}_nx_{configs.nx}_ny_{configs.ny}'
        else:
            raise NotImplementedError

        loaded_model_name += f'_time_steps_{configs.time_steps_train}'
        return loaded_model_name
    
    def rollout(self, model, x, t, nt):
        # x - [bs, 1, nx,ny]
        # t - [lf,]
        # NT - length of target time-series

        fno_repeat_shape = (1, 1, x.shape[2]) + (1,) * (len(x.shape) - 3)

        configs = self.configs
        y_pred_ls = []

        bs = 1
        end = bs
        for end in range(bs, x.shape[0] + 1, bs):
            start = end - bs
            out_ls = []
            
            temp_x1 = x[start:end]  # [BS, 1, nx,ny]
            while (len(out_ls) - 1) * (configs.look_forward - 1) < nt:
                model.eval()
                with torch.no_grad():
                    if 'FNO' in self.model_name:
                        with autocast(): 
                            out = model(temp_x1.cuda())
                            out_lf = out.reshape(-1, configs.look_forward,*configs.output_shape)[:, 1:]
                        out_ls.append(out_lf)

                        grids = x[None, 0 , 1:, ...]
                        next_step = out_lf[:, -1:, ...]
                        next_step = next_step[:, None, ...]
                        next_step = next_step.repeat(fno_repeat_shape)
                        temp_x1 = torch.cat([next_step, grids], dim=1)
                    else:
                        temp_x = torch.repeat_interleave(temp_x1, configs.look_forward, dim=0) #[BS*lf, 1, nx,ny]
                        temp_t = t[0, 0].repeat(bs) #[BS*lf, ]
                        with autocast(): 
                            out = model(temp_x.cuda(), temp_t.cuda())
                            out = out.reshape(-1, configs.look_forward,*configs.output_shape)[:, 1:] #[BS, lf, nx,ny]
                        out_ls.append(out)
                        
                        temp_x1 = out[:,-1:] #[BS, 1, nx,ny]
                        
            pred = torch.cat(out_ls, dim=1)[:, :nt] #[BS, 200, nx, ny]
            y_pred_ls.append(pred)

        y_pred = torch.cat(y_pred_ls, dim=0) #.reshape(1,-1,Par['nz'],Par['ny'],Par['nx'])
        return y_pred

    def evaluate(self, test_loader):
        self.logger.info(f'Evaluating model {self.model_name} ...')

        matplotlib.rcParams['image.cmap'] = 'twilight'
        model = self.model
        model.eval()
        l2_loss = LpLoss(size_average=False)
        l2_abs_loss = LpLoss(size_average=False, use_relative=False)
        l_inf_loss = LpLoss(size_average=False, p=np.inf)
        l_inf_loss_abs = LpLoss(size_average=False, p=np.inf, use_relative=False)
        mse_loss = torch.nn.MSELoss(reduction='sum')
        mae_loss = torch.nn.L1Loss(reduction='sum')

        losses_fn = {
            'l2_loss': l2_loss,
            'l2_abs_loss': l2_abs_loss,
            'l_inf_loss': l_inf_loss,
            'l_inf_loss_abs': l_inf_loss_abs,
            'mse_loss': mse_loss,
            'mae_loss': mae_loss,
            'r2_score': r2_score,
            'rmse_loss': rmse_loss
        }

        losses_values = {k: 0. for k in losses_fn.keys()}

        y_pred_full, y_true_full = None, None
        test_loss = 0.

        with torch.no_grad():
            for x, t, y_true in tqdm(test_loader):
                x, t, y_true = x.cuda(), t.cuda(), y_true.cuda()

                loss = 0.

                if self.configs.extrapolation:
                    y_pred = self.rollout(model, x, t, y_true.shape[1])
                elif 'point' in self.model_name:
                    y_pred = model(x, time=t, grid=self.grid)
                elif 'ditto' in self.model_name:
                    y_pred = model(x, t)
                elif self.model_name == 'DeepONet':
                    y_pred = model(x, t)
                elif 'FNO' in self.model_name:
                    y_pred = model(x)
                elif 'RNN' in self.model_name:
                    y_pred = model(x)
                    jump = y_pred.size(2) // y_true.size(2)
                    # y_pred = y_pred[:, :, ::jump, ...]
                    y_pred = y_pred[:, :, (jump - 1)::jump, ...]
                    if "RNN" in self.model_name:
                        y_true = y_true[:, 0, ...]
                        y_pred = y_pred[:, 0, ...]
                elif self.model_name in ['UNet', 'RegUNet']:
                    time_interp = max(32, int(16 * np.ceil(x.shape[2] / 16.)))
                    original_shape = x.shape[2:]
                    x = torch.nn.functional.interpolate(x, size=(time_interp,) + x.shape[3:])
                    if self.configs.scenario == 'burgers':
                        out = model(x, time=None, use_grid=False)
                        out = out[..., :original_shape[0], :original_shape[1]]
                        y_pred = out[:, 0, ...]
                    else:
                        out = model(x, time=None)
                        out = out[..., :original_shape[0], :original_shape[1], :original_shape[2]]
                        y_pred = out[:, 0, ...]

                if self.y_normalizer is not None:
                    y_pred = self.y_normalizer.decode(y_pred)
                    y_true = self.y_normalizer.decode(y_true)

                if self.configs.extrapolation:
                    y_true = y_true[:, 1:]
                    y_pred = y_pred[:, 1:]

                loss = l2_loss(y_pred, y_true)
                test_loss += loss.item()

                if y_pred_full is None:
                    y_pred_full = y_pred
                    y_test_full = y_true
                else:
                    y_pred_full = torch.concat([y_pred_full, y_pred], dim=0)
                    y_test_full = torch.concat([y_test_full, y_true], dim=0)

        self.logger.info(f'Finished evaluating {self.model_name} model')
        if 'ditto' in self.model_name and not self.configs.extrapolation:
            steps = self.configs.time_steps_inference
            y_pred_full = y_pred_full.reshape(y_pred_full.shape[0] // steps, steps, *self.configs.output_shape)
            y_test_full = y_test_full.reshape(y_test_full.shape[0] // steps, steps, *self.configs.output_shape)

        if self.configs.extrapolation:
            np.save(os.path.join(self.path_model, 'y_pred_full.npy'), y_pred_full.cpu().numpy())
            np.save(os.path.join(self.path_model, 'y_test_full.npy'), y_test_full.cpu().numpy())

        if (self.configs.plot and
                self.configs.plot_nt.get(str(self.configs.time_steps_inference)) is not None):
            self.plotter.plot(y_test_full, y_pred_full, t)

        num_samples = y_pred_full.shape[0]  # len(test_loader.dataset)
        for k, fn in losses_fn.items():
            if k in ["mse_loss", "mae_loss", "rmse_loss"]:
                losses_values[k] = fn(y_pred_full.reshape(num_samples, -1), y_test_full.reshape(num_samples, -1)).item() / num_samples
            elif k == "r2_score":
                losses_values[k] = fn(y_pred_full.reshape(num_samples, -1), y_test_full.reshape(num_samples, -1)).item()
            else:
                losses_values[k] = fn(y_pred_full, y_test_full).item() / num_samples
        metrics = self.report_metrics(losses_values)
        return metrics

    def report_metrics(self, losses_values):
        for k, v in losses_values.items():
            losses_values[k] = round(v, 16)

        model_name = self.path_model.split(os.path.sep)[-1][:-4]
        model_name_dict = {'ditto': 'DiTTO',
                           'ditto_point': 'DiTTO-point',
                           f'ditto_subsampled_X{self.configs.subsample}': 'DiTTO-s',
                           f'ditto_point_subsampled_X{self.configs.subsample}': 'DiTTO-point-s',
                           f'ditto_point_subsampled_X{self.configs.subsample}_gate': ' DiTTO-point-s-gate',
                           f'ditto_point_subsampled_X{self.configs.subsample}_deep_gate': ' DiTTO-point-s-deep-gate',
                           f'ditto_point_subsampled_X{self.configs.subsample}_deeper_gate': ' DiTTO-point-s-deeper-gate',
                           f'ditto_point_subsampled_X{self.configs.subsample}_block_gate': ' DiTTO-point-s-block-gate',
                           f'ditto_point_subsampled_X{self.configs.subsample}_att_gate': 'DiTTO-point-s-att-gate',
                           f'ditto_point_subsampled_X{self.configs.subsample}_att': 'DiTTO-point-s-att',
                           'ditto_no_attention': 'DiTTO-no-att',
                           'FNO': 'FNO',
                           'UNet': 'UNet',
                           'RNN': 'RNN',
                           'DeepONet': 'DeepONet',
                           }
        # dict_keys = list(model_name_dict.keys())
        # for k in dict_keys:
        # model_name_dict[k + '_extrapolation'] = model_name_dict[k] + '-extrapolation'
        model_key = self.model_name if 'seed' not in self.model_name else self.model_name.split('_seed')[0]
        model_key = model_key if 'noise' not in model_key else model_key.split('_noise')[0]
        if 'extrapolation' in self.model_name:
            model_key = self.model_name.split('_extrapolation')[0]
        model_name = model_name_dict[model_key]

        metrics = {k: {} for k in losses_values.keys()}
        for k, v in losses_values.items():
            metrics[k][model_name] = {r"$N_t^{test} = " + f"{self.configs.time_steps_inference}$": v}
        return metrics