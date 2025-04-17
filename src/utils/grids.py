import torch
import numpy as np


def get_xt_grid(nx, nt, bot=[0, 0], top=[1, 1], dtype='float32'):
    '''
    Args:
        S: number of points on each spatial domain
        T: number of points on temporal domain including endpoint
        bot: list or tuple, lower bound on each dimension
        top: list or tuple, upper bound on each dimension
        dtype: torch dtype

    Returns:
        (n_x, n_t, 2) array of grid points
    '''
    x_arr = np.linspace(bot[0], top[0], num=nx, endpoint=True)
    dt = (top[1] - bot[1]) / nt
    t_arr = np.linspace(bot[1] + dt, top[1], num=nt)
    x_grid, t_grid = np.meshgrid(x_arr, t_arr, indexing='ij')
    x_axis = np.ravel(x_grid)
    t_axis = np.ravel(t_grid)
    grid = np.stack([x_axis, t_axis], axis=0).T
    
    grid_space = x_arr.reshape(-1, 1)

    grids_dict = {'grid_x': x_arr, 
                  'grid_t': t_arr, 
                  'grid_space': grid_space, 
                  'grid': grid}
    for key in grids_dict.keys():
        grids_dict[key] = torch.tensor(grids_dict[key], dtype=eval('torch.' + dtype), device='cpu')
    return grids_dict


def get_xyt_grid(nx=None, ny=None, nt=None, bot=[0, 0, 0], top=[1, 1, 1], dtype='float32',
                 x_arr=None, y_arr=None, t_arr=None, dt=None):
    '''
    Args:
        S: number of points on each spatial domain
        T: number of points on temporal domain including endpoint
        bot: list or tuple, lower bound on each dimension
        top: list or tuple, upper bound on each dimension

    Returns:
        (n_x, n_y, n_t, 3) array of grid points
    '''
    if x_arr is None:
        x_arr = np.linspace(bot[0], top[0], num=nx, endpoint=True)

    if y_arr is None:
        y_arr = np.linspace(bot[1], top[1], num=ny, endpoint=True)

    if t_arr is None:
        if dt is None:
            dt = (top[2] - bot[2]) / nt
        t_arr = np.linspace(bot[2] + dt, top[2], num=nt)

    x_grid, y_grid, t_grid = np.meshgrid(x_arr, y_arr, t_arr, indexing='ij')
    x_axis = np.ravel(x_grid)
    y_axis = np.ravel(y_grid)
    t_axis = np.ravel(t_grid)
    grid = np.stack([x_axis, y_axis, t_axis], axis=0).T

    x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')
    x_axis = np.ravel(x_grid)
    y_axis = np.ravel(y_grid)
    grid_space = np.stack([x_axis, y_axis], axis=0).T

    grids_dict = {'grid_x': x_arr, 
                  'grid_y': y_arr,
                  'grid_t': t_arr, 
                  'grid_space': grid_space, 
                  'grid': grid}
    for key in grids_dict.keys():
        grids_dict[key] = torch.tensor(grids_dict[key], dtype=eval('torch.' + dtype), device='cpu')
    return grids_dict


def get_xyzt_grid(nx, ny, nz, nt, bot=[0, 0, 0, 0], top=[1, 1, 1, 1], dtype='float32', dt=None, z_arr=None):
    '''
    Args:
        S: number of points on each spatial domain
        T: number of points on temporal domain including endpoint
        bot: list or tuple, lower bound on each dimension
        top: list or tuple, upper bound on each dimension

    Returns:
        (n_x, n_y, n_z n_t, 4) array of grid points
    '''
    x_arr = np.linspace(bot[0], top[0], num=nx, endpoint=True)
    y_arr = np.linspace(bot[1], top[1], num=ny, endpoint=True)
    if z_arr is None:
        z_arr = np.linspace(bot[2], top[2], num=nz, endpoint=True)
    
    if dt is None:
        dt = (top[3] - bot[3]) / nt
    t_arr = np.linspace(bot[3] + dt, top[3], num=nt)
    x_grid, y_grid, z_grid, t_grid = np.meshgrid(x_arr, y_arr, z_arr, t_arr, indexing='ij')
    x_axis = np.ravel(x_grid)
    y_axis = np.ravel(y_grid)
    z_axis = np.ravel(z_grid)
    t_axis = np.ravel(t_grid)
    grid = np.stack([x_axis, y_axis, z_axis, t_axis], axis=0).T


    x_grid, y_grid, z_grid = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
    x_axis = np.ravel(x_grid)
    y_axis = np.ravel(y_grid)
    z_axis = np.ravel(z_grid)
    space_grid = np.stack([x_axis, y_axis, z_axis], axis=0).T

    grids_dict = {'grid_x': x_arr, 
                  'grid_y': y_arr,
                  'grid_z': z_arr,
                  'grid_t': t_arr, 
                   'grid_space': space_grid,
                  'grid': grid}
    for key in grids_dict.keys():
        grids_dict[key] = torch.tensor(grids_dict[key], dtype=eval('torch.' + dtype), device='cpu')
    return grids_dict
