import os
import wandb
import torch
import numpy as np
from accelerate import Accelerator
from torch.cuda.amp import autocast

from timeit import default_timer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.utils import (EarlyStopper, LpLoss)
from src.utils.losses import (fdm_wave_2d_loss, get_pi_wave_2d)


class Trainer:
    def __init__(self, model, configs, logger, model_name='ditto', y_normalizer=None, grid=None):
        self.model = model
        self.configs = configs
        self.logger = logger
        self.model_name = model_name
        self.model_key = self.model_name if 'noise' not in self.model_name else self.model_name.split('_noise')[0]
        if 'extrapolation' in self.model_name:
            self.model_key = self.model_name.split('_extrapolation')[0]

        self.y_normalizer = y_normalizer
        self.grid = grid['grid_space'].permute(1, 0)[None, ...].cuda()
        self.pi_variables = None

        self._set_paths()

        wandb.log({'Model name': model_name})

    def _set_paths(self):
        configs = self.configs
        self.path_logs = os.path.join(
            '..', 'outputs', 'logs', configs.scenario, self.model_name)
        if not os.path.exists(self.path_logs):
            os.makedirs(self.path_logs)
        self.path_model = os.path.join(
            '..', 'outputs', 'models', configs.scenario, self.model_name)
        if not os.path.exists(self.path_model):
            os.makedirs(self.path_model)

        prefix = self._set_prefix(configs)
        self.path_logs = os.path.join(self.path_logs, prefix + '.csv')
        self.path_model = os.path.join(self.path_model, prefix + '.pth')
    
    def _set_prefix(self, configs):
        if configs.scenario == 'navier_stokes':
            self.logger.info(f'Reynolds number: {configs.reynolds_number}')
            prefix = f'NavierStokes_v_{configs.viscosity}_N_{configs.N}_T_{configs.T}_nx_{configs.nx}_ny_{configs.ny}'
        elif configs.scenario == 'wave_equation_2d':
            prefix = f'WaveEq_N_{configs.N}_T_{configs.T}_velocity_{configs.velocity}_nx_{configs.nx}_ny_{configs.ny}'
        elif configs.scenario == 'shallow_water':
            prefix = f'Shallow_Water_N_{configs.N}_T_{configs.T}_nx_{configs.nx}_ny_{configs.ny}'
        elif configs.scenario == 'wave_equation_3d':
            prefix = (f'WaveEq_N_{configs.N}_T_{configs.T}_velocity_{configs.velocity}' +
                      f'_nx_{configs.nx}_ny_{configs.ny}_nz_{configs.nz}')
        elif configs.scenario == 'burgers':
            prefix = f'Burgers_v_{configs.viscosity}_N_{configs.N}_T_{configs.T}_size_{configs.nx}'
        elif configs.scenario == 'climate_2d':
            prefix = f"Climate_2d_N_{configs.N}_nx_{configs.nx}_ny_{configs.ny}_level_{configs.pressure_level}"
        elif configs.scenario == 'climate_3d':
            prefix = f"Climate_3d_N_{configs.N}_nx_{configs.nx}_ny_{configs.ny}_nz_{configs.nz}"
        elif configs.scenario == 'hypersonics':
            prefix = f'Hypersonics_N_{configs.N}_M_{configs.T}_nx_{configs.nx}_ny_{configs.ny}'
        elif configs.scenario == 'hypersonics_time':
            prefix = f'Hypersonics_time_N_{configs.N}_T_{configs.T}_nx_{configs.nx}_ny_{configs.ny}'
        else:
            raise NotImplementedError
        prefix += f'_time_steps_{configs.time_steps_train}'
        return prefix

    def train_step(self, model, train_loader, loss_fn, optimizer, scheduler, epoch, subsample):
        model.train()
        train_loss = 0

        for x, t, y in tqdm(train_loader):
            if subsample > 1:
                indices = torch.randint(0, x.shape[0], (x.shape[0] // subsample,))
                x, t, y = x[indices].cuda(), t[indices].cuda(), y[indices].cuda()
            else:
                x, t, y = x.cuda(), t.cuda(), y.cuda()

            optimizer.zero_grad()

            if 'point' in self.model_name:
                out = model(x, time=t, grid=self.grid)
            elif 'ditto' in self.model_name:
                out = model(x, t)
            elif 'FNO' in self.model_name:
                out = model(x)
            elif 'RNN' in self.model_name:
                out = model(x)[:, :, 3::4, ...]
                y = y[:, :, 3::4, ...]
            elif self.model_name in ['UNet', 'RegUNet']:
                time_interp = max(32, int(16 * np.ceil(x.shape[2] / 16.)))
                original_shape = x.shape[2:]
                x = torch.nn.functional.interpolate(x, size=(time_interp,) + x.shape[3:])
                if self.configs.scenario == 'burgers':
                    out = model(x, time=None, use_grid=False)
                    out = out[..., :original_shape[0], :original_shape[1]]
                else:
                    out = model(x, time=None)
                    out = out[..., :original_shape[0], :original_shape[1], :original_shape[2]]

            if self.y_normalizer is not None:
                out = self.y_normalizer.decode(out)
                y = self.y_normalizer.decode(y)
            loss = loss_fn(out.view(self.configs.batch_size_train[self.model_key] // subsample, -1),
                           y.contiguous().view(self.configs.batch_size_train[self.model_key] // subsample, -1))

            if self.configs.pi_loss:
                if epoch > self.configs.switch_epoch:
                    if self.pi_variables is None:
                        self.pi_variables = get_pi_wave_2d(x, self.configs.velocity)
                    pi_loss = fdm_wave_2d_loss(model, x, t, out, self.pi_variables, loss_fn)
                    loss = loss * 0.5 + pi_loss * 0.5

            self.accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        return train_loss

    def val_step(self, model, val_loader, loss_fn):
        model.eval()
        val_loss = 0.0
        full_val_l2_loss = 0.0

        with torch.no_grad():
            for x, t, y in val_loader:
                x, t, y = x.cuda(), t.cuda(), y.cuda()
                loss = 0

                if self.configs.extrapolation:
                    out = self.rollout(model, x, t, y.shape[1])
                elif 'point' in self.model_name:
                    out = model(x, time=t, grid=self.grid)
                elif 'ditto' in self.model_name:
                    out = model(x, t)
                elif 'FNO' in self.model_name:
                    out = model(x)
                elif 'RNN' in self.model_name:
                    out = model(x)[:, :, 3::4, ...]
                    y = y[:, :, 3::4, ...]
                elif self.model_name in ['UNet', 'RegUNet']:
                    time_interp = max(32, int(16 * np.ceil(x.shape[2] / 16.)))
                    original_shape = x.shape[2:]
                    x = torch.nn.functional.interpolate(x, size=(time_interp,) + x.shape[3:])
                    if self.configs.scenario == 'burgers':
                        out = model(x, time=None, use_grid=False)
                        out = out[..., :original_shape[0], :original_shape[1]]
                    else:
                        out = model(x, time=None)
                        out = out[..., :original_shape[0], :original_shape[1], :original_shape[2]]

                if self.y_normalizer is not None:
                    out = self.y_normalizer.decode(out)
                    y = self.y_normalizer.decode(y)

                loss += loss_fn(out, y)
                val_loss += loss.item()

        return val_loss, full_val_l2_loss
    
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

    def train(self, train_loader, val_loader):
        model = self.model
        configs = self.configs

        self.ntrain = len(train_loader.dataset)  # * configs.time_steps_train
        self.nval = len(val_loader.dataset)  # * configs.time_steps_train
        epochs = configs.epochs[self.model_key]

        subsample = 1
        if 'subsample' in self.model_name:
            subsample = configs.subsample
            self.ntrain = self.ntrain // subsample

        iterations = epochs * (len(train_loader.dataset) // configs.batch_size_train[self.model_key])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        self.accelerator = Accelerator(mixed_precision="fp16")
        model, optimizer, train_loader = self.accelerator.prepare(
            model, optimizer, train_loader)

        early_stopper = EarlyStopper(patience=500)

        lp_loss = LpLoss(size_average=False, p=2)

        val_losses = []
        logs = []

        for ep in range(epochs):
            start_time = default_timer()

            train_loss = self.train_step(model=model,
                                         train_loader=train_loader,
                                         loss_fn=lp_loss,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         epoch=ep,
                                         subsample=subsample)

            val_loss, full_val_l2_loss = self.val_step(model=model,
                                                       val_loader=val_loader,
                                                       loss_fn=lp_loss)

            train_loss /= self.ntrain
            val_loss /= self.nval

            end_time = default_timer()
            lr = '{:.3e}'.format(scheduler.optimizer.param_groups[0]['lr'])

            current_vals = {'Epoch': ep, 'Time': end_time - start_time,
                            'Train loss': train_loss,
                            'Val loss': val_loss, 'lr': lr}

            self.logger.info(f'Epoch: {ep:03d} ' +
                             f'Time: {end_time - start_time:3f}' + ' | ' +
                             f'Train loss: {train_loss:.3e}' + ' | ' +
                             f'Val loss: {val_loss:.3e}' + ' | ' +
                             f'lr: {lr}')

            logs.append(current_vals)
            wandb.log(logs[-1])

            if ep > 1:
                if val_loss < min(val_losses):
                    torch.save(model.state_dict(), self.path_model)
            val_losses.append(val_loss)
            pd.DataFrame(logs).to_csv(self.path_logs)

            if early_stopper.early_stop(val_loss):
                self.logger.info('Terminating due to early stopping')
                self.logger.info(
                    f'Best validation loss: {early_stopper.min_validation_loss}')
                break

        artifact = wandb.Artifact(
            self.path_model.split(os.path.sep)[-1], type='model')
        artifact.add_file(self.path_model)
        wandb.run.log_artifact(artifact)
        return model
