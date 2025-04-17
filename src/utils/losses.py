import numpy as np
import torch


def get_pi_wave_2d(x, velocity):
    pi_variables = {}
    nx = x.shape[2]
    ny = x.shape[3]
    dx, dy = torch.pi / (nx - 1), torch.pi / (ny - 1)

    x_axis, y_axis = np.linspace(0, np.pi, nx), np.linspace(0, np.pi, ny)
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
    c0 = velocity * np.sin(1 * x_mesh) * np.sin(1 * y_mesh) + 1
    c0sq = np.power(c0, 2)
    c0sq = torch.tensor(c0sq).cuda()
    
    pi_variables['c0sq'] = c0sq
    pi_variables['nx'] = nx
    pi_variables['ny'] = ny 
    pi_variables['dx'] = dx
    pi_variables['dy'] = dy
    pi_variables['cfl'] = (1 / (2 * c0sq.max()) * (1 / (dx**-2 + dy**-2)) ** 0.5) * 0.9 # 0.9 is a safety factor
    return pi_variables


def fdm_wave_2d_loss(model, x, t, y_pred, pi_variables, loss_fn=None, eps=10**-6):        
    c0sq = pi_variables['c0sq']
    nx = pi_variables['nx']
    ny = pi_variables['ny']
    dx = pi_variables['dx']
    dy = pi_variables['dy']
    cfl = pi_variables['cfl']

    dt = (eps - cfl) * torch.rand(1, device='cuda') + cfl
    random_t = (0 - 2) * torch.rand(t.shape[0], device='cuda') + 2
    u_prev = model(x, random_t) 
    u_curr = model(x, random_t + dt)
    u_next = model(x, random_t + 2 * dt)
    
    u_next_pred = u_next[..., 1:nx - 1, 1:ny - 1]
    u_next_fd = 2 * u_curr[..., 1:nx - 1, 1:ny - 1] - u_prev[..., 1:nx - 1, 1:ny - 1] + \
                c0sq[1:nx - 1, 1:ny - 1] * dt ** 2 * (
                (u_curr[..., 2:nx, 1:ny - 1] - 2 * u_curr[..., 1:nx - 1, 1:ny -
                 1] + u_curr[..., 0:nx - 2, 1:ny - 1]) / (dx ** 2)
                + (u_curr[..., 1:nx - 1, 2:ny] - 2 * u_curr[..., 1:nx - 1, 1:ny - 1] + u_curr[..., 1:nx - 1, 0:ny - 2]) / (dy ** 2))
    
    if loss_fn is None:
        return torch.linalg.norm(u_next_pred - u_next_fd) / torch.linalg.norm(u_next_fd) 
    else:
        return loss_fn(u_next_pred, u_next_fd)  
