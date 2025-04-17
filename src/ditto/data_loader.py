import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view

from src.utils.utils import GaussianNormalizer
from src.utils.grids import (get_xt_grid, get_xyt_grid, get_xyzt_grid)
from src.data.navier_stokes_generator import generate_ns_data
from src.data.wave_generator import WaveDataGenerator2D, WaveDataGenerator3D
from src.data.shallow_water_preproccessor import preprocess_shallow_water
from src.data.hypersonics_preprocessor import preprocess_hypersonics


class DataHandler:
    def __init__(self, configs, model_name, logger) -> None:
        self.configs = configs
        self.model_name = model_name
        self.logger = logger

        nt_train = configs.time_steps_train if not configs.extrapolation else configs.look_forward
        nt_inference = configs.time_steps_inference if not configs.extrapolation else configs.look_forward

        if configs.scenario == 'navier_stokes':
            self.path = os.path.join(
                '..', 'data', 'navier_stokes',
                f'NavierStokes_v_{configs.viscosity}_N_{configs.N}_T_{configs.T}_nx_{configs.nx}_ny_{configs.ny}')
            self.grid_train = \
                get_xyt_grid(configs.nx, configs.ny, nt_train, dt=0,
                             bot=[0, 0, 0], top=[1, 1, 1], dtype=configs.dtype)
            self.grid_test = \
                get_xyt_grid(configs.nx, configs.ny, nt_inference, dt=0,
                             bot=[0, 0, 0], top=[1, 1, 1], dtype=configs.dtype)
        elif configs.scenario == 'wave_equation_2d':
            self.path = os.path.join(
                '..', 'data', 'wave_equation_2d',
                f'WaveEq_N_{configs.N}_T_{configs.T}_velocity_{configs.velocity}_nx_{configs.nx}_ny_{configs.ny}')
            self.grid_train = \
                get_xyt_grid(configs.nx, configs.ny, nt_train,
                             bot=[0, 0, 0], top=[np.pi, np.pi, 1], dtype=configs.dtype)
            self.grid_test = \
                get_xyt_grid(configs.nx, configs.ny, nt_inference,
                             bot=[0, 0, 0], top=[np.pi, np.pi, 1], dtype=configs.dtype)
        elif configs.scenario == 'wave_equation_3d':
            self.path = os.path.join(
                '..', 'data', 'wave_equation_3d',
                f'WaveEq_N_{configs.N}_T_{configs.T}_velocity_{configs.velocity}_nx_{configs.nx}_ny_{configs.ny}_nz_{configs.nz}')
            final_t = configs.T if not configs.extrapolation else 1.

            self.grid_train = \
                get_xyzt_grid(configs.nx, configs.ny, configs.nz, nt_train,
                              bot=[0, 0, 0, 0], top=[1, 1, 1, final_t], dtype=configs.dtype)
            self.grid_test = \
                get_xyzt_grid(configs.nx, configs.ny, configs.nz, nt_inference,
                              bot=[0, 0, 0, 0], top=[1, 1, 1, final_t], dtype=configs.dtype)
        elif configs.scenario == 'burgers':
            self.path = os.path.join(
                '..', 'data', 'burgers', f'Burgers_v_{configs.viscosity}_N_{configs.N}_T_{configs.T}_size_{configs.nx}')
            self.grid_train = \
                get_xt_grid(configs.nx, nt_train,
                            bot=[0, 0], top=[1, 1], dtype=configs.dtype)
            self.grid_test = \
                get_xt_grid(configs.nx, nt_inference, bot=[
                            0, 0], top=[1, 1], dtype=configs.dtype)
        elif configs.scenario == 'climate_2d':
            self.path = os.path.join('..', 'data', 'climate', 'climate_2d')
            final_t = 56. if not configs.extrapolation else 1.

            self.grid_train = \
                get_xyt_grid(configs.nx, configs.ny, nt_train,
                             bot=[0, -90., 0], top=[357.5, 90., final_t],
                             dtype=configs.dtype)
            self.grid_test = \
                get_xyt_grid(configs.nx, configs.ny, nt_inference,
                             bot=[0, -90., 0], top=[357.5, 90., final_t],
                             dtype=configs.dtype)
        elif configs.scenario == 'climate_3d':
            self.path = os.path.join('..', 'data', 'climate', 'climate_3d')
            final_t = 56. if not configs.extrapolation else 1.

            self.grid_train = \
                get_xyzt_grid(configs.nx, configs.ny, configs.nz, nt_train,
                              bot=[0, -90., None, 0], top=[357.5, 90., None, 16],
                              z_arr=np.array(configs.pressure_levels),
                              dtype=configs.dtype)
            self.grid_test = \
                get_xyzt_grid(configs.nx, configs.ny, configs.nz, nt_inference,
                              bot=[0, -90., None, 0], top=[357.5, 90., None, 16],
                              z_arr=np.array(configs.pressure_levels),
                              dtype=configs.dtype)
        elif configs.scenario == 'shallow_water':
            self.path = os.path.join(
                '..', 'data', 'shallow_water', f'Shallow_Water_N_{configs.N}_T_{configs.T}_size_nx_{configs.nx}_ny_{configs.ny}')
            self.grid_train = \
                get_xyt_grid(configs.nx, configs.ny, nt_train,
                             bot=[-np.pi, -np.pi, 0], top=[np.pi, np.pi, 1], dtype=configs.dtype)
            self.grid_test = \
                get_xyt_grid(configs.nx, configs.ny, nt_inference,
                             bot=[-np.pi, -np.pi, 0], top=[np.pi, np.pi, 1], dtype=configs.dtype)
        elif configs.scenario == 'hypersonics':
            # No extrapolation for hypersonics yet
            self.path = os.path.join(
                '..', 'data', 'hypersonics', f'Hypersonics_N_{configs.N}_T_{configs.T}_nx_{configs.nx}_ny_{configs.ny}')
            
            self.grid_train = \
                get_xyt_grid(configs.nx, configs.ny, nt_train,
                            bot=[-0.01, 0, 8], top=[0.185, 0.11886, 10], dtype=configs.dtype)
            self.grid_test = \
                get_xyt_grid(configs.nx, configs.ny, nt_inference,
                            bot=[-0.01, 0, 8.05], top=[0.185, 0.11886, 9.95], dtype=configs.dtype)
        # elif configs.scenario == 'hypersonics_time':
        #     # No extrapolation for hypersonics yet
        #     self.path = os.path.join(
        #         '..', 'data', 'hypersonics', f'Hypersonics_time_N_{configs.N}_T_{configs.T}_nx_{configs.nx}_ny_{configs.ny}')
            
        #     self.grid_train = \
        #         get_xyt_grid(configs.nx, configs.ny, nt_train,
        #                     bot=[-0.01, 0, 8], top=[0.185, 0.11886, 10], dtype=configs.dtype)
        #     self.grid_test = \
        #         get_xyt_grid(configs.nx, configs.ny, nt_inference,
        #                     bot=[-0.01, 0, 8.05], top=[0.185, 0.11886, 9.95], dtype=configs.dtype)
        elif configs.scenario == 'hypersonics_time':
            self.path = os.path.join(
                '..', 'data', 'hypersonics', f'Hypersonics_time_N_{configs.N}_T_{configs.T}_nx_{configs.nx}_ny_{configs.ny}')
            
            self.grid_train = \
                get_xyt_grid(configs.nx, configs.ny, configs.time_steps_train,
                            bot=[-0.01, 0, 0], top=[0.185, 0.11886, configs.T], dtype=configs.dtype)
            self.grid_test = \
                get_xyt_grid(configs.nx, configs.ny, configs.time_steps_inference,
                            bot=[-0.01, 0, 0], top=[0.185, 0.11886, configs.T], dtype=configs.dtype)
        else:
            raise ValueError(configs.scenario + ' is not a valid scenario')

        if not os.path.exists(self.path):
            self.logger.info(f'Generating dataset to {self.path}')
            os.makedirs(self.path)
            self._generate_data()

        self.y_normalizer = None

    def _generate_data(self):
        configs = self.configs
        if configs.scenario == 'navier_stokes':
            generate_ns_data(configs)
        elif configs.scenario == 'shallow_water':
            preprocess_shallow_water(configs)
        elif configs.scenario == 'hypersonics':
            preprocess_hypersonics(configs)
        elif configs.scenario == 'wave_equation_2d':
            generator = WaveDataGenerator2D(num_ic=configs.N, final_time=configs.T, num_sources=1,
                                            velocity=configs.velocity, nt=configs.time_steps_inference * 10,
                                            nx=configs.nx, ny=configs.ny,
                                            save_every=10)
            generator.generate(path=self.path)
        elif configs.scenario == 'wave_equation_3d':
            generator = WaveDataGenerator3D(num_ic=configs.N, final_time=configs.T, num_sources=1,
                                            velocity=configs.velocity, nt=configs.time_steps_inference * 10,
                                            nx=configs.nx, ny=configs.ny, nz=configs.nz,
                                            save_every=10, save_gif=False)
            generator.generate(path=self.path)
        else:
            raise NotImplementedError

    def split_data(self, x, y, t, save=True):
        configs = self.configs
        path = self.path

        if y.shape[-1] % self.configs.time_steps_train != 0:
            raise ValueError(
                f"Data contains {y.shape[-1]} time steps, which is not divisible by train steps {self.configs.time_steps_train}"
            )

        if y.shape[-1] % self.configs.time_steps_inference != 0:
            raise ValueError(
                f"Data contains {y.shape[-1]} time steps, which is not divisible by test steps {self.configs.time_steps_train}"
            )

        if None not in (configs.ntrain, configs.nval, configs.ntest):
            split_index1 = int(configs.ntrain)
            split_index2 = int((configs.ntrain + configs.nval))
            split_index3 = int((configs.ntrain + configs.nval + configs.ntest))
        else:
            split_index1 = int(x.shape[0] * 0.8)
            split_index2 = int(x.shape[0] * 0.9)
            split_index3 = x.shape[0]

        data_dict = {}
        data_dict["x_train"] = x[:split_index1]
        data_dict["y_train"] = y[:split_index1]
        if t is not None:
            data_dict["t_train"] = t[:split_index1]

        # This condition was added to allow a fair comparison with FNOs.
        # In the FNO paper the authors do not separate test and valdiation sets.
        if configs.use_same_validation_and_test:
            data_dict["x_val"] = data_dict["x_test"] = x[split_index1:]
            data_dict["y_val"] = data_dict["y_test"] = y[split_index1:]
            if t is not None:
                data_dict["t_val"] = data_dict["t_test"] = t[split_index1:]
        else:
            data_dict["x_val"] = x[split_index1:split_index2]
            data_dict["y_val"] = y[split_index1:split_index2]
            if t is not None:
                data_dict["t_val"] = t[split_index1:split_index2]

            data_dict["x_test"] = x[split_index2:split_index3]
            data_dict["y_test"] = y[split_index2:split_index3]
            if t is not None:
                data_dict["t_test"] = t[split_index2:split_index3]

        stride_train = data_dict["y_train"].shape[-1] // self.configs.time_steps_train
        if stride_train == data_dict["y_train"].shape[-1]:
            data_dict["y_train"] = data_dict["y_train"][..., -1:]
            data_dict["y_val"] = data_dict["y_val"][..., -1:]
            if t is not None:
                data_dict["t_train"] = data_dict["t_train"][..., -1:]
                data_dict["t_val"] = data_dict["t_val"][..., -1:]
        elif stride_train != self.configs.time_steps_train:
            data_dict["y_train"] = data_dict["y_train"][...,stride_train - 1::stride_train]
            data_dict["y_val"] = data_dict["y_val"][..., stride_train - 1::stride_train]
            if t is not None:
                data_dict["t_train"] = data_dict["t_train"][..., stride_train - 1::stride_train]
                data_dict["t_val"] = data_dict["t_val"][...,  stride_train - 1::stride_train]

        stride_test = data_dict["y_test"].shape[-1] // self.configs.time_steps_inference
        data_dict["y_test"] = data_dict["y_test"][..., stride_test - 1::stride_test]
        if t is not None:
            data_dict["t_test"] = data_dict["t_test"][..., stride_test - 1::stride_test]

        if save:
            for key, val in data_dict.items():
                np.save(os.path.join(path, f'{key}.npy'), val)
        return data_dict
    
    def _split_trajectories(self, traj, save=True):
        data_dict = {}

        traj = np.swapaxes(traj, 1, -1)
        train_split = (int(traj.shape[0] * 0.8), int(traj.shape[1] * 0.5))
        val_split = (int(traj.shape[0] * 0.1), int(traj.shape[1] * 0.25))

        traj_train = traj[:train_split[0], :train_split[1] + 1, ...]
        traj_val = traj[train_split[0]:train_split[0] + val_split[0], train_split[1]:train_split[1] + val_split[1] + 1, ...]
        traj_test = traj[train_split[0] + val_split[0]:, train_split[1]:, ...]

        data_dict['x_train'], data_dict['y_train'] = self._create_train_trajectories(traj_train)
        data_dict['x_val'], data_dict['y_val'] = self._create_val_test_trajectories(traj_val)
        data_dict['x_test'], data_dict['y_test'] = self._create_val_test_trajectories(traj_test)

        data_dict = self._convert_to_torch(data_dict)

        return data_dict

    def _create_train_trajectories(self, traj_train):
        configs = self.configs
        # Creating the sliding windows
        x_train = sliding_window_view(x=traj_train[:,:-(configs.look_forward - 1), :, :], 
                                      window_shape=configs.look_back, 
                                      axis=1)
        x_train = np.swapaxes(x_train, 2, -1)
        x_train = x_train.reshape(-1, configs.look_back, *configs.output_shape)
        
        y_train = sliding_window_view(x=traj_train[:, (configs.look_back -1):, :, :], 
                                      window_shape=configs.look_forward, 
                                      axis=1)
        y_train = np.swapaxes(y_train, 2, -1)
        y_train = y_train.reshape(-1, configs.look_forward, *configs.output_shape)

        return x_train, y_train
    
    def _create_val_test_trajectories(self, traj):
        x = traj[:, :1, ...]
        y = traj
        return x, y

    def _convert_to_torch(self, data_dict):
        for k in data_dict.keys():
            data_dict[k] = torch.from_numpy(
                data_dict[k].astype(self.configs.dtype))
        return data_dict

    def _permute_channels(self, data_dict):
        """Permute channels to match the input shape of the model"""
        for k in data_dict.keys():
            if k[0] == 'y':
                if len(data_dict[k].shape) == 5:  # 3D case
                    data_dict[k] = data_dict[k].permute(0, 4, 1, 2, 3)
                elif len(data_dict[k].shape) == 4:  # 2D case
                    data_dict[k] = data_dict[k].permute(0, 3, 1, 2)
                elif len(data_dict[k].shape) == 3:  # 1D case
                    data_dict[k] = data_dict[k].permute(0, 2, 1)
        return data_dict

    def _add_channel_dimension(self, data_dict, keys=None):
        if keys is None:
            keys = ['x']
            
        """Add channel dimension to x"""
        for k in data_dict.keys():
            if k[0] in keys:
                data_dict[k] = data_dict[k][:, None, ...]
        return data_dict

    def _create_t_component(self, data_dict):
        """
        Create t component for each model type.
        DeepONet: repeat the same spatio-temporal grid for each sample.
        FNO: set t to zero, since it accepts one input.
        ditto: no preprocessing needed.
        """
        if self.model_name == 'DeepONet':
            data_dict['t_train'] = self.grid_train['grid'].repeat(
                data_dict['t_train'].shape[0], 1, 1)
            data_dict['t_val'] = self.grid_train['grid'].repeat(
                data_dict['t_val'].shape[0], 1, 1)
            data_dict['t_test'] = self.grid_test['grid'].repeat(
                data_dict['t_test'].shape[0], 1, 1)
        elif self.model_name == 'FNO':
            data_dict['t_train'] = torch.zeros_like(data_dict['t_train'])
            data_dict['t_val'] = torch.zeros_like(data_dict['t_val'])
            data_dict['t_test'] = torch.zeros_like(data_dict['t_test'])
        return data_dict

    def _preprocess_fno(self, data_dict):
        data_dim = len(data_dict['x_train'].shape) - 2
        if data_dim != len(data_dict['x_test'].shape) - 2:
            raise ValueError('Data dimension mismatch between train and test sets.' +
                             f'Train: {data_dim}, Test: {len(data_dict["x_test"].shape)}')

        # Repeat shape is [1, time_steps_train, 1, 1, 1] for 3D data
        if not self.configs.extrapolation:
            time_steps_train = self.configs.time_steps_train
            time_steps_val = self.configs.time_steps_inference if self.configs.scenario == 'hypersonics' \
                else self.configs.time_steps_train
            time_steps_test = self.configs.time_steps_inference
        else:
            time_steps_train = self.configs.look_forward
            time_steps_val = self.configs.look_forward
            time_steps_test = self.configs.look_forward

        repeat_shape_train = [1, time_steps_train] + [1] * data_dim
        repeat_shape_val = [1, time_steps_val] + [1] * data_dim
        repeat_shape_test = [1, time_steps_test] + [1] * data_dim

        data_dict['x_train'] = data_dict['x_train'].repeat(repeat_shape_train)
        data_dict['x_val'] = data_dict['x_val'].repeat(repeat_shape_val)
        data_dict['x_test'] = data_dict['x_test'].repeat(repeat_shape_test)

        for dataset in ['x_train', 'x_val', 'x_test']:
            n_samples = data_dict[dataset].shape[0]
            if self.configs.scenario == 'hypersonics':
                grid = self.grid_test if dataset in ['x_test', 'x_val'] else self.grid_train
                time_steps = time_steps_test if dataset in ['x_test', 'x_val'] else time_steps_train
            else:
                grid = self.grid_test if dataset == 'x_test' else self.grid_train
                time_steps = time_steps_test if dataset == 'x_test' else time_steps_train

            grid_t = grid['grid_t'].reshape([1, time_steps] + [1] * data_dim)
            grid_t = grid_t.repeat(
                [n_samples, 1] + list(data_dict[dataset].shape[2:]))
            if data_dim == 1:
                x_shape = [1, 1] + list(data_dict[dataset].shape[2:3])
                grid_x = grid['grid_x'].reshape(
                    x_shape).repeat(n_samples, time_steps, 1)
                data_dict[dataset] = torch.stack(
                    [data_dict[dataset], grid_x, grid_t], dim=1)
            elif data_dim == 2:
                x_shape = [1, 1] + list(data_dict[dataset].shape[2:3]) + [1]
                y_shape = [1, 1, 1] + list(data_dict[dataset].shape[3:])

                grid_x = grid['grid_x'].reshape(x_shape).repeat(
                    n_samples, time_steps, 1, y_shape[-1])
                grid_y = grid['grid_y'].reshape(y_shape).repeat(
                    n_samples, time_steps, x_shape[-2], 1)

                data_dict[dataset] = torch.stack(
                    [data_dict[dataset], grid_x, grid_y, grid_t], dim=1)
            elif data_dim == 3:
                x_shape = [1, 1] + list(data_dict[dataset].shape[2:3]) + [1, 1]
                y_shape = [1, 1, 1] + list(data_dict[dataset].shape[3:4]) + [1]
                z_shape = [1, 1, 1, 1] + list(data_dict[dataset].shape[4:])
                grid_x = grid['grid_x'].reshape(x_shape)
                grid_y = grid['grid_y'].reshape(y_shape)
                grid_z = grid['grid_z'].reshape(z_shape)

                grid_x = grid_x.repeat(
                    n_samples, time_steps, 1, y_shape[-2], z_shape[-1])
                grid_y = grid_y.repeat(
                    n_samples, time_steps, x_shape[-3], 1, z_shape[-1])
                grid_z = grid_z.repeat(
                    n_samples, time_steps, x_shape[-3], y_shape[-2], 1)

                data_dict[dataset] = torch.stack(
                    [data_dict[dataset], grid_x, grid_y, grid_z, grid_t], dim=1)
        
        return data_dict

    def process_data(self, data_dict):
        data_dict = self._convert_to_torch(data_dict)
        data_dict = self._permute_channels(data_dict)
        data_dict = self._add_channel_dimension(data_dict)
        data_dict = self._create_t_component(data_dict)
        return data_dict

    def normalize(self, data_dict):
        scale = True if 'climate' in self.model_name.lower() else False
        self.x_normalizer = GaussianNormalizer(data_dict['x_train'], scale=scale)
        self.y_normalizer = GaussianNormalizer(data_dict['y_train'], scale=scale)
        for k in data_dict.keys():
            if k[0] == 'x':
                data_dict[k] = self.x_normalizer.encode(data_dict[k])
            elif k[0] == 'y':
                data_dict[k] = self.y_normalizer.encode(data_dict[k])
        return data_dict

    def _load_data_dict(self, path):
        x_raw = np.load(os.path.join(path, 'x.npy'), mmap_mode=None)
        y_raw = np.load(os.path.join(path, 'y.npy'), mmap_mode='r')
        t_raw = np.load(os.path.join(path, 't.npy'), mmap_mode=None)

        if 'point' in self.model_name:
            x_shape = x_raw.shape
            y_shape = y_raw.shape
            x_raw = x_raw.reshape(x_shape[0], math.prod(x_shape[1:]))
            y_raw = y_raw.reshape(y_shape[0], math.prod(y_shape[1:-1]), y_shape[-1])

        self.logger.info(f'Shape x: {x_raw.shape}')
        self.logger.info(f'Shape y: {y_raw.shape}')

        if self.configs.extrapolation:
            traj = np.append(np.expand_dims(x_raw, axis=-1), y_raw, axis=-1)
            data_dict = self._split_trajectories(traj, save=False)
        else:
            data_dict = self.split_data(x_raw, y_raw, t_raw, save=False)
        return data_dict
    
    def _load_data_dict_hypersonics(self, path):
        data_dict = np.load(os.path.join(path, 'raw_data.npy'), mmap_mode=None, allow_pickle=True).item()

        # Add noise based on standard deviation of the data
        np.random.seed(0)
        for key, val in data_dict.items():
            if 'x_' in key:
                data_dict[key] = val + np.random.normal(0, np.std(val) * self.configs.noise_level, val.shape)
        
        if 'point' in self.model_name:
            for mode in ['train', 'val', 'test']:
                x_shape = data_dict[f'x_{mode}'].shape
                y_shape = data_dict[f'y_{mode}'].shape
                data_dict[f'x_{mode}'] = data_dict[f'x_{mode}'].reshape(x_shape[0], math.prod(x_shape[1:]))
                data_dict[f'y_{mode}'] = data_dict[f'y_{mode}'].reshape(y_shape[0], math.prod(y_shape[1:-1]), y_shape[-1])
        return data_dict

    def _prepare_fno_extrapolation(self, data_dict):
        # data_dict['x_train'] = data_dict['x_train'][:, 0]
        # data_dict['y_train'] = torch.swapaxes(data_dict['y_train'], 1, -1)
        data_dict['t_train'] = self.grid_train['grid_t'][None, :].repeat(data_dict['x_train'].shape[0], 1)

        # data_dict['x_val'] = data_dict['x_val'][:, 0]
        # data_dict['y_val'] = torch.swapaxes(data_dict['y_val'], 1, -1)
        data_dict['t_val'] = self.grid_train['grid_t'][None, :].repeat(data_dict['x_val'].shape[0], 1)

        # data_dict['x_test'] = data_dict['x_test'][:, 0]
        # data_dict['y_test'] = torch.swapaxes(data_dict['y_test'], 1, -1)
        data_dict['t_test'] = self.grid_test['grid_t'][None, :].repeat(data_dict['x_test'].shape[0], 1)
        return data_dict

    def load_data(self):
        path = self.path
        self.logger.info(f'Loading data from {path}')
        self.logger.info(f'Current working dir: {os.getcwd()}')

        data_dict = self._load_data_dict_hypersonics(path) if self.configs.scenario == 'hypersonics' else self._load_data_dict(path)
        
        if not self.configs.extrapolation:
            data_dict = self.process_data(data_dict)
        else:
            data_dict = self._prepare_fno_extrapolation(data_dict)

        if self.model_name in ['FNO', 'UNet', 'RegUNet']:
            data_dict = self._preprocess_fno(data_dict)
        elif "RNN" in self.model_name:
            # For RNN we need to add the channel dimension to x and y
            data_dict = self._add_channel_dimension(data_dict, keys=['x', 'y'])
        
        self.logger.info("-"*50)
        self.logger.info("Shapes after preprocessing:")
        for k in data_dict.keys():
            self.logger.info(f'Shape {k}: {data_dict[k].shape}')

        if self.configs.normalize:
            data_dict = self.normalize(data_dict)

        if 'ditto' in self.model_name:
            concat_grid = False  # if 'point' in self.model_name else True
            if self.configs.scenario == 'hypersonics':
                grid_val = self.grid_test
            else:
                grid_val = self.grid_train
            train_ds = TimePointwiseDataset(data_dict['x_train'], data_dict['y_train'], self.grid_train, concat_grid)

            if not self.configs.extrapolation:
                val_ds = TimePointwiseDataset(data_dict['x_val'], data_dict['y_val'], grid_val, concat_grid)
                test_ds = TimePointwiseDataset(data_dict['x_test'], data_dict['y_test'], self.grid_test, concat_grid)
            else:
                grid_train = self.grid_train["grid_t"]
                grid_train = grid_train.repeat(data_dict['x_val'].shape[0], 1, 1)

                grid_test = self.grid_test["grid_t"]
                grid_test = grid_test.repeat(data_dict['x_test'].shape[0], 1, 1)
                val_ds = torch.utils.data.TensorDataset(data_dict["x_val"], grid_train, data_dict["y_val"])
                test_ds = torch.utils.data.TensorDataset(data_dict["x_test"], grid_test, data_dict["y_test"])
        elif self.configs.extrapolation:
            t_train = self.grid_train["grid_t"].repeat(data_dict['x_train'].shape[0], 1, 1)
            t_val = self.grid_train["grid_t"].repeat(data_dict['x_val'].shape[0], 1, 1)
            t_test = self.grid_test["grid_t"].repeat(data_dict['x_test'].shape[0], 1, 1)
            train_ds = torch.utils.data.TensorDataset(data_dict["x_train"], t_train, data_dict["y_train"])
            val_ds = torch.utils.data.TensorDataset(data_dict["x_val"], t_val, data_dict["y_val"])
            test_ds = torch.utils.data.TensorDataset(data_dict["x_test"], t_test, data_dict["y_test"])
        else:
            train_ds = torch.utils.data.TensorDataset(data_dict["x_train"], data_dict["t_train"], data_dict["y_train"])
            val_ds = torch.utils.data.TensorDataset(data_dict["x_val"], data_dict["t_val"], data_dict["y_val"])
            test_ds = torch.utils.data.TensorDataset(data_dict["x_test"], data_dict["t_test"], data_dict["y_test"])

        train_ds = torch.utils.data.DataLoader(train_ds,
                                               batch_size=self.configs.batch_size_train[self.model_name],
                                               shuffle=True, drop_last=True)

        val_ds = torch.utils.data.DataLoader(val_ds,
                                             batch_size=self.configs.batch_size_val[self.model_name],
                                             shuffle=False, drop_last=True)

        test_ds = torch.utils.data.DataLoader(test_ds,
                                              batch_size=self.configs.batch_size_test[self.model_name],
                                              shuffle=False, drop_last=True)
        self.logger.info('Done loading data')
        return train_ds, val_ds, test_ds


class TimePointwiseDataset(Dataset):
    def __init__(self, x_data, y_data, grid, concat_grid=False):
        self.intial_conditions = x_data
        self.full_solutions = y_data

        self.num_samples = x_data.shape[0]
        self.time_len = grid['grid_t'].shape[0]
        self.grid = grid['grid_t']  # (n_t, 1)
        self.grid_space = grid['grid_space'].permute(1, 0)  # (n_x X n_y, 2)

        self.concat_grid = concat_grid

    def __len__(self):
        return self.num_samples * self.time_len

    def __getitem__(self, idx):
        num_per_instance = self.time_len
        instance_id = idx // num_per_instance
        pos_id = idx % num_per_instance

        point = self.grid[pos_id]
        u0 = self.intial_conditions[instance_id]
        if self.concat_grid:
            u0 = torch.cat([u0, self.grid_space], axis=0)
        y = self.full_solutions[instance_id][pos_id]
        return u0, point, y
