import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from netCDF4 import Dataset as CDFDataset

from src.utils.grids import (get_xyt_grid, get_xyzt_grid)


def preprocess_climate(file_path, time_window=30, dim=2, pressure_level=None):
    file_id = CDFDataset(file_path)
    nx = file_id.variables['lon'][:].shape[0]
    ny = file_id.variables['lat'][:].shape[0]

    nx = nx - nx % 8
    ny = ny - ny % 8

    if dim == 3:
        nz = file_id.variables['level'][:].shape[0]
        nz = nz - nz % 8

    dtype = "float32"

    if dim == 2:
        grid = get_xyt_grid(nx, ny, time_window, bot=[0, 0, 0], top=[1, 1, 1], dtype=dtype, dt=0.)
    elif dim == 3:
        grid = get_xyzt_grid(nx, ny, nz, time_window, bot=[0, 0, 0, 0], top=[1, 1, 1, 1], dtype=dtype, dt=0.)

    full_solutions = np.array(file_id.variables['air'])
    intial_conditions = np.array(file_id.variables['air'])

    if dim == 2:
        full_solutions = full_solutions[:, pressure_level, :ny, :nx]
        intial_conditions = intial_conditions[:, pressure_level, :ny, :nx]
    elif dim == 3:
        full_solutions = full_solutions[:, :nz, :ny, :nx]
        intial_conditions = intial_conditions[:, :nz, :ny, :nx]

    full_shape = full_solutions.shape
    cutoff = (full_shape[0]//time_window)*time_window
    full_solutions = full_solutions[:cutoff]
    intial_conditions = intial_conditions[:cutoff][::time_window, ...]        

    full_solutions = full_solutions.reshape(cutoff//time_window, time_window, *full_shape[1:])
    full_solutions = torch.tensor(full_solutions)
    if dim == 2:
        full_solutions = full_solutions.permute(0, 1, 3, 2).numpy()
    elif dim == 3:
        full_solutions = full_solutions.permute(0, 4, 3, 2, 1).numpy()
    
    # For simplicity, we want the number of initial conditions to be divisible by 80
    cutoff_ic = full_solutions.shape[0] - full_solutions.shape[0] % 80

    full_solutions = full_solutions[:cutoff_ic, ...]
    intial_conditions = intial_conditions[:cutoff_ic, ...]

    t = grid['grid_t'][None, :].repeat(intial_conditions.shape[0], 1)
    scenario_name = f'climate_{dim}d'
    if pressure_level is not None:
        scenario_name += f"_level_{int(file_id.variables['level'][:][pressure_level])}"
    path_name = os.path.join('climate', scenario_name)
    
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    np.save(os.path.join(path_name, 't.npy'), t)
    np.save(os.path.join(path_name, 'x.npy'), intial_conditions)
    np.save(os.path.join(path_name, 'y.npy'), full_solutions)


def animate_climate(full_solutions, time_window):
    for j in range(time_window):
        plt.cla()
        plt.title(j)
        plt.contourf(full_solutions[0][j])
        plt.pause(0.1)
    plt.show()


if __name__ == '__main__':
    file_path = os.path.join('climate', 'climate_levels_2015_2020.nc')
    preprocess_climate(file_path, time_window=30, dim=2, pressure_level=0)    
    # preprocess_climate(file_path, time_window=30, dim=2, pressure_level=-1)    
    # preprocess_climate(file_path, time_window=30, dim=3)