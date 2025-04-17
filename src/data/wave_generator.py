import copy
import os
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils.plot_utils_3d import plot_3d, save_gif


class WaveDataGenerator:
    def __init__(self, num_ic: int, final_time: float, num_sources: int,
                 velocity: float, nt: int, nx: int, ny: int, save_every: int) -> None:
        self.num_ic = num_ic
        self.final_time = final_time
        self.num_sources = num_sources
        self.velocity = velocity
        self.nt = nt
        self.nx = nx
        self.ny = ny
        self.save_every = save_every
        self.print_cfl = False

    def generate(self, path=None) -> None:
        data, labels, t = self.generate_samples()
        self.save(data, labels, t, path=path)

    def generate_samples(self):
        pass

    def generate_path(self):
        pass

    def save(self, x, y, t, path=None) -> None:
        if path is None:
            path = self.generate_path() 
        np.save(os.path.join(path, 'x.npy'), x)
        np.save(os.path.join(path, 'y.npy'), y)
        np.save(os.path.join(path, 't.npy'), t)

    def solve(self, ic):
        pass


class WaveDataGenerator2D(WaveDataGenerator):
    def __init__(self, num_ic: int, final_time: float, num_sources: int,
                 velocity: float, nt: int, nx: int, ny: int, save_every: int) -> None:
        super().__init__(num_ic, final_time, num_sources, velocity, nt, nx, ny, save_every)
        self.print_cfl = False

    def generate_samples(self):
        time_steps = self.nt // self.save_every

        x = np.zeros((self.num_ic, self.nx, self.ny))
        y = np.zeros((self.num_ic, self.nx, self.ny, time_steps))
        t = np.zeros((self.num_ic, time_steps))

        t_axis = np.linspace(0, self.final_time, self.nt + 1)[1::self.save_every]
        for n in tqdm(range(self.num_ic)):
            ic = np.zeros((self.nx, self.ny))
            for _ in range(self.num_sources):
                x_coordinate = np.random.randint(
                    int(20 / 2), self.nx - int(20 / 2) - 1)
                y_coordinate = np.random.randint(
                    int(20 / 2), self.ny - int(20 / 2) - 1)
                ic += self.generate_ic(x_coordinate, y_coordinate, 20, 1)

            solution = self.solve(ic)
            x[n, ...] = ic
            y[n, ...] = torch.tensor(solution).permute(1, 2, 0).numpy()
            t[n, ...] = t_axis
        return x, y, t

    def generate_path(self):
        filename = f'WaveEq_N_{self.num_ic}_T_{int(self.final_time)}_velocity_{self.velocity}_nx_{self.nx}_ny_{self.ny}'
        path = os.path.join('wave_equation_2d', filename)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def generate_ic(self, x_0, y_0, width, amp):
        g_erup = np.zeros((self.nx, self.ny))
        down_x, up_x, center_x = x_0 - \
            int(width / 2), x_0 + int(width / 2), x_0
        down_y, up_y, center_y = y_0 - \
            int(width / 2), y_0 + int(width / 2), y_0
        for k in range(down_x, up_x + 1):
            for j in range(down_y, up_y + 1):
                g_erup[k, j] = amp * \
                    np.exp(-(np.power((k - center_x), 2) +
                           np.power((j - center_y), 2)) / 10)
        return g_erup

    def solve(self, ic):
        amplification_factor = 1
        x_axis, y_axis = np.linspace(
            0, np.pi, self.nx), np.linspace(0, np.pi, self.ny)
        x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
        # c0 = self.velocity * (1 + amplification_factor) * (np.sin(1 * x_mesh) * np.sin(1 * y_mesh))
        c0 = self.velocity * amplification_factor * \
            (np.sin(1 * x_mesh) * np.sin(1 * y_mesh)) + 1
        c0sq = np.power(c0, 2)

        f_time = self.final_time
        u_next = np.zeros((self.nx, self.ny))
        solution = np.zeros(
            (int(self.nt // self.save_every), self.nx, self.ny))
        dx, dy = (np.pi / (self.nx - 1), np.pi / (self.ny - 1))
        nx = self.nx
        ny = self.ny
        dt = f_time / (self.nt - 1)

        if self.print_cfl:
            print('CFL is: ', c0sq * dt**2 / dx**2)
            self.print_cfl = False

        u_prev = ic
        u_curr = ic

        counter = 0
        if self.save_every == 1:
            solution[0, :, :] = ic
            counter = 1

        for n in range(2, self.nt + 1):
            u_next[1:nx - 1, 1:ny - 1] = 2 * u_curr[1:nx - 1, 1:ny - 1] - u_prev[1:nx - 1, 1:ny - 1] + \
                c0sq[1:nx - 1, 1:ny - 1] * np.power(dt, 2) * (
                (u_curr[2:nx, 1:ny - 1] - 2 * u_curr[1:nx - 1, 1:ny -
                 1] + u_curr[0:nx - 2, 1:ny - 1]) / (np.power(dx, 2))
                + (u_curr[1:nx - 1, 2:ny] - 2 * u_curr[1:nx - 1, 1:ny - 1] + u_curr[1:nx - 1, 0:ny - 2]) / (np.power(dy, 2)))
            u_prev = copy.copy(u_curr)
            u_curr = copy.copy(u_next)

            if (n + 1) % self.save_every == 0:
                solution[counter, :, :] = u_next
                counter += 1

        return solution


class WaveDataGenerator3D(WaveDataGenerator):
    def __init__(self, num_ic: int, final_time: float,
                 num_sources: int, velocity: float,
                 nt: int, nx: int, ny: int, nz: int,
                 save_every: int, save_gif: bool = False) -> None:
        super().__init__(num_ic, final_time, num_sources, velocity, nt, nx, ny, save_every)
        self.nz = nz
        self.save_gif = save_gif

    def generate_samples(self):
        time_steps = self.nt // self.save_every

        data = np.zeros((self.num_ic, self.nx, self.ny, self.nz))
        labels = np.zeros((self.num_ic, self.nx, self.ny, self.nz, time_steps))
        t = np.zeros((self.num_ic, time_steps))

        t_axis = np.linspace(0, self.final_time, self.nt)[1::self.save_every]
        for n in tqdm(range(self.num_ic)):
            ic = np.zeros((self.nx, self.ny, self.nz))
            for _ in range(self.num_sources):
                x_coordinate = np.random.randint(
                    int(20 / 2), self.nx - int(20 / 2) - 1)
                y_coordinate = np.random.randint(
                    int(20 / 2), self.ny - int(20 / 2) - 1)
                z_coordinate = np.random.randint(
                    int(20 / 2), self.nz - int(20 / 2) - 1)

                ic += self.generate_ic(x_coordinate,
                                       y_coordinate, z_coordinate, 20, 1)

            solution = self.solve(ic)
            data[n, ...] = ic
            labels[n, ...] = torch.tensor(solution).permute(1, 2, 3, 0).numpy()
            t[n, ...] = t_axis
        return data, labels, t

    def generate_path(self):
        path = os.path.join('wave_equation_3d',
                            f'WaveEq_N_{self.num_ic}_T_{int(self.final_time)}_velocity_{self.velocity}' + 
                            f'_nx_{self.nx}_ny_{self.ny}_nz_{self.nz}')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def generate_ic(self, x_0, y_0, z_0, width, amp, sigma=10):
        g_erup = np.zeros((self.nx, self.ny, self.nz))
        down_x, up_x, center_x = x_0 - \
            int(width / 2), x_0 + int(width / 2), x_0
        down_y, up_y, center_y = y_0 - \
            int(width / 2), y_0 + int(width / 2), y_0
        down_z, up_z, center_z = z_0 - \
            int(width / 2), z_0 + int(width / 2), z_0
        for k in range(down_x, up_x + 1):
            for j in range(down_y, up_y + 1):
                for l in range(down_z, up_z + 1):
                    g_erup[k, j, l] = amp * \
                        np.exp(-(np.power((k - center_x), 2) +
                                 np.power((j - center_y), 2) +
                                 np.power((l - center_z), 2)) / sigma)
        return g_erup

    def solve(self, ic):
        amplification_factor = 1
        x_axis = np.linspace(0, np.pi, self.nx)
        y_axis = np.linspace(0, np.pi, self.ny)
        z_axis = np.linspace(0, np.pi, self.nz)
        x_mesh, y_mesh, z_mesh = np.meshgrid(x_axis, y_axis, z_axis)
        # c0 = self.velocity * (1 + amplification_factor) * (np.sin(1 * x_mesh) * np.sin(1 * y_mesh))
        c0 = self.velocity * amplification_factor * \
            (np.sin(2 * x_mesh) * np.sin(1 * y_mesh) * np.sin(1 * z_mesh)) + 1
        # c0 = self.velocity * np.ones_like(x_mesh)
        c0sq = np.power(c0, 2)

        f_time = self.final_time
        u_next = np.zeros((self.nx, self.ny, self.nz))
        solution = np.zeros(
            (int(self.nt // self.save_every), self.nx, self.ny, self.nz))
        dx, dy, dz = (np.pi / (self.nx - 1), np.pi /
                      (self.ny - 1), np.pi / (self.nz - 1))
        nx = self.nx
        ny = self.ny
        nz = self.nz
        dt = f_time / (self.nt - 1)

        if self.print_cfl:
            print('CFL is: ', c0sq * dt**2 / dx**2)
            self.print_cfl = False
        # dt = 1 / (c0 * np.sqrt(2 * (1 / (np.power(dx, 2)) + 1 / (np.power(dy, 2)))))

        u_prev = ic
        u_curr = ic

        solution[0, :, :] = u_prev
        solution[1, :, :] = u_curr

        figs_path = os.path.join(
            '..', 'operator', 'figs', 'wave_equation', '3d', 'anim')
        if not os.path.exists(figs_path):
            os.makedirs(figs_path)

        ax = None
        counter = 0
        for n in range(2, self.nt):
            u_next[1:nx - 1, 1:ny - 1, 1:nz - 1] = \
                (2 * u_curr[1:nx - 1, 1:ny - 1, 1:nz - 1] -
                 u_prev[1:nx - 1, 1:ny - 1, 1:nz - 1] +
                 c0sq[1:nx - 1, 1:ny - 1, 1:nz - 1] * np.power(dt, 2) *
                 ((u_curr[2:nx, 1:ny - 1, 1:nz - 1] -
                   2 * u_curr[1:nx - 1, 1:ny - 1, 1:nz - 1] +
                   u_curr[0:nx - 2, 1:ny - 1, 1:nz - 1]) / (np.power(dx, 2)) +
                  (u_curr[1:nx - 1, 2:ny, 1:nz - 1] -
                   2 * u_curr[1:nx - 1, 1:ny - 1, 1:nz - 1] +
                   u_curr[1:nx - 1, 0:ny - 2, 1:nz - 1]) / (np.power(dy, 2)) +
                  (u_curr[1:nx - 1, 1:ny - 1, 2:nz] -
                   2 * u_curr[1:nx - 1, 1:ny - 1, 1:nz - 1] +
                   u_curr[1:nx - 1, 1:ny - 1, 0:nz - 2]) / (np.power(dz, 2))))
            u_prev = copy.copy(u_curr)
            u_curr = copy.copy(u_next)

            if (n + 1) % self.save_every == 0:
                solution[counter, :, :] = u_next
                counter += 1
                if self.save_gif:
                    ax = plot_3d(x_mesh,
                                 y_mesh,
                                 z_mesh,
                                 u_next,
                                 ax=ax,
                                 step=n, show_slice=False)
                    plt.savefig(os.path.join(figs_path, str(n) + '.png'))

        if self.save_gif:
            save_gif(
                figs_path, gif_name=f'wave_equation_3d_{nx}X{ny}X{nz}_velocity_{self.velocity}.gif')
            self.save_gif = False

        return solution


if __name__ == '__main__':
    np.random.seed(123)
    # generaor_2d = WaveDataGenerator2D(num_ic=1000, final_time=2., num_sources=1,
    #                                   velocity=1., nt=1000, nx=64, ny=64,
    #                                   save_every=1)
    # generaor_2d.generate()

    generaor_3d = WaveDataGenerator3D(num_ic=102, final_time=2., num_sources=1,
                                      velocity=1., nt=1000, nx=32, ny=32, nz=32,
                                      save_every=5, save_gif=False)
    generaor_3d.generate()
