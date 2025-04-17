import os
from copy import copy, deepcopy

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap

from src.utils.plot_utils_3d import save_gif, save_gif_climate


class Plotter:
    def __init__(self, configs, logger, path_figs, save_every, num_figures=1) -> None:
        self.configs = configs
        self.logger = logger
        if configs.scenario == 'burgers':
            self.dim = 1
        elif configs.scenario in ['navier_stokes', 'wave_equation_2d', 'shallow_water', 'hypersonics', 'hypersonics_time', 'climate_2d']:
            self.dim = 2
        elif configs.scenario == 'wave_equation_3d':
            self.dim = 3
        else:
            raise ValueError(configs.scenario + ' is not a valid scenario')
        self.path_figs = path_figs
        self.save_every = save_every

        self.num_figures = 1 if configs.scenario == 'hypersonics' else num_figures

    def plot(self, y_test, y_pred, t):
        for fig_num in range(self.num_figures):
            self.logger.info(f'Animating fig {fig_num + 1} out of {self.num_figures}')
            name = f"fig_{fig_num + 1}"
            if not os.path.exists(os.path.join(self.path_figs, name)):
                os.makedirs(os.path.join(self.path_figs, name))

            y_test_fig = y_test[fig_num].cpu()
            y_pred_fig = y_pred[fig_num].cpu()

            dt = self.configs.T / self.configs.time_steps_inference
            t_fig = torch.linspace(dt, self.configs.T, self.configs.time_steps_inference)
            if self.configs.scenario == 'hypersonics':
                t_fig = torch.linspace(8, self.configs.T, self.configs.time_steps_inference * 2 + 1)[1::2]
            
            if self.configs.scenario == 'climate_2d':
                self.plot_map_2d(y_test_fig, y_pred_fig, t_fig, name)
            elif self.dim == 1:
                self.plot_1d(y_test_fig, y_pred_fig, t_fig, name)
                # self.plot_1d_with_time(y_test_fig, y_pred_fig, name=name)
            elif self.dim == 2:
                self.plot_2d(y_test_fig, y_pred_fig, t_fig, name)
            elif self.dim == 3:
                self.plot_3d(y_test_fig, y_pred_fig, t_fig, name)

    def plot_1d(self, y_test, y_pred, t, name):
        x_axis = np.linspace(0, 1, y_pred.shape[1])

        # size = y_test[:, 0].shape[0] // 5
        size = y_test.shape[0] // 5
        if size >= 5:
            fig, ax = plt.subplots(1, 5, sharey=True, figsize=(20, 4))
            for i in range(5):
                index = (i + 1) * size - 1
                ax[i].plot(x_axis, y_test[index, :], color="r", label='Ground truth')
                ax[i].plot(x_axis, y_pred[index, :], '--', color="g", label='Prediction')
                relative_l2 = np.linalg.norm(y_test[index] - y_pred[index]) / np.linalg.norm(y_test[index])
                ax[i].set_title(f"Time: t = {t[index]:.3f}" + f"\n $L^2$: {relative_l2:.2e}")
            handles, labels = ax[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper left')
            plt.savefig(os.path.join(self.path_figs, name, 'full_snapshots.png'))

        fig, ax = plt.subplots()
        line1, = ax.plot(x_axis, y_test[0, :], color="r", label='Ground truth')
        line2, = ax.plot(x_axis, y_pred[0, :],
                         '--', color="g", label='Prediction')

        def update(n, line1, line2):
            relative_l2 = np.linalg.norm(
                y_test[n] - y_pred[n]) / np.linalg.norm(y_test[n])
            relative_l_inf = np.linalg.norm(y_test[n].cpu(
            ) - y_pred[n], ord=np.inf) / np.linalg.norm(y_test[n], ord=np.inf)
            ax.set_title(f"Time: t = {t[n]:.3f}" +
                         f"\n $L^2$: {relative_l2:.2e}")
            #  f"     $L^\infty$: {relative_l_inf:.2e}")

            line1.set_data(x_axis, y_test[n, :])
            line2.set_data(x_axis, y_pred[n, :])

            if (n % self.save_every == 0) or (n == self.configs.time_steps_inference - 1):
                plt.savefig(
                    (os.path.join(self.path_figs, name, f'snapshot_{n}')))

            return [line1, line2]

        fig.legend()
        anim = animation.FuncAnimation(fig, update, frames=self.configs.time_steps_inference, fargs=[line1, line2],
                                       interval=20, blit=False)
        anim.save(os.path.join(self.path_figs, name,
                  'animation.gif'), writer='pillow')

    def plot_1d_with_time(self, y_test, y_pred, name):
        relative_l2 = np.linalg.norm(y_test - y_pred) / np.linalg.norm(y_test)
        relative_l_inf = np.linalg.norm(y_test - y_pred, ord=np.inf) / np.linalg.norm(y_test, ord=np.inf)

        fig, ax = plt.subplots(1, 3, sharey=True)
        fig.suptitle(f"$L^2$: {relative_l2:.2e}",
                     #  f"\n$L^\infty$: {relative_l_inf:.2e}",
                     fontsize=16)
        ax[0].imshow(y_pred.cpu(), origin='lower')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('t')
        ax[0].set_title('Prediction')

        ax[1].imshow(y_test.cpu(), origin='lower')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('t')
        ax[1].set_title('Ground truth')

        im = ax[2].imshow(abs((y_test.cpu() - y_pred.cpu()))**2, origin='lower')
        ax[2].set_xlabel('x')
        ax[2].set_ylabel('t')
        ax[2].set_title('Error')
        fig.colorbar(im, ax=ax, format='%.0e')

        plt.savefig(os.path.join(self.path_figs, name, 'full_solution'))

    def plot_2d(self, y_test, y_pred, t, name):
        # size = y_test.shape[0] // 5
        size = y_test.shape[0] // 5
        if y_test.shape[0] >= 5:
            if self.configs.scenario == 'hypersonics':
                self.plot_2d_hypersonics(y_test, y_pred, t, name)
            else:
                fig, ax = plt.subplots(1, 5, sharey=True, figsize=(20, 4))

                for i in range(5):
                    index = (i + 1) * size - 1
                    relative_l2 = np.linalg.norm(y_test[index] - y_pred[index]) / np.linalg.norm(y_test[index])
                    ax[i].set_title(f"Time: t = {t[index]:.3f}" + f"\n $L^2$: {relative_l2:.2e}")
                    ax[i].imshow(y_pred[index])
                handles, labels = ax[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper left')
                plt.savefig(os.path.join(self.path_figs, name, 'full_snapshots.png'))

                fig, ax = plt.subplots(1, 5, sharey=True, figsize=(20, 4))
                for i in range(5):
                    index = (i + 1) * size - 1
                    if self.configs.scenario == 'hypersonics':
                        ax[i].contourf(y_pred[index], levels=100)
                        ax[i].set_title(f"Mach: M = {t[index]:.2f}")
                    else:
                        ax[i].imshow(y_test[index])
                        ax[i].set_title(f"Time: t = {t[index]:.3f}")
                plt.savefig(os.path.join(self.path_figs, name, 'full_snapshots_analytic.png'))

        fig, ax = plt.subplots(1, 3)

        def update(i):
            relative_l2 = np.linalg.norm(y_test[i].flatten() - y_pred[i].flatten()) \
                / np.linalg.norm(y_test[i].flatten())
            relative_l_inf = np.linalg.norm(y_test[i].flatten() - y_pred[i].flatten(), ord=np.inf) \
                / np.linalg.norm(y_test[i].flatten(), ord=np.inf)

            if self.configs.scenario == 'hypersonics':
                fig.suptitle(f"Mach: M = {t[i]:.2f}" + 
                            f"\n $L^2$: {relative_l2:.2e}",
                            # f"\n $L^\infty$: {relative_l_inf:.2e}",
                            fontsize=20)
            else:
                fig.suptitle(f"Time: t = {t[i].cpu():.3f}" + 
                                f"\n $L^2$: {relative_l2:.2e}",
                                # f"\n $L^\infty$: {relative_l_inf:.2e}",
                                fontsize=20)
                
            ax[0].contourf(y_pred[i].cpu(), levels=100) if self.configs.scenario == 'hypersonics'\
                  else ax[0].imshow(y_pred[i].cpu())
            ax[0].set_title("Prediction", fontsize=20)
            ax[0].set_axis_off()

            ax[1].contourf(y_test[i].cpu(), levels=100) if self.configs.scenario == 'hypersonics'\
                  else ax[1].imshow(y_test[i].cpu())
            ax[1].set_title("Ground truth", fontsize=20)
            ax[1].set_axis_off()

            ax[2].contourf(torch.abs(y_test[i].cpu() - y_pred[i].cpu()), levels=100) if self.configs.scenario == 'hypersonics'\
                  else ax[2].imshow(torch.abs(y_test[i].cpu() - y_pred[i].cpu()))
            ax[2].set_title("Absolute error", fontsize=20)
            ax[2].set_axis_off()

            if (i % self.save_every == 0) or (i == self.configs.time_steps_inference - 1):
                plt.savefig(
                    (os.path.join(self.path_figs, name, f'snapshot_{i}')))
        fig.subplots_adjust(left=0, bottom=0.05, right=1, top=0.8)

        anim = animation.FuncAnimation(
            fig, update, frames=self.configs.time_steps_inference, interval=50*4)
        anim.save(os.path.join(self.path_figs, name, 'animation.gif'), writer='pillow')

    def plot_2d_hypersonics(self, y_test, y_pred, t, name):
        size = y_test.shape[0] // 5
        if y_test.shape[0] < 5:
            return

        cmap = cm.get_cmap('twilight', 40)
        cmap = ListedColormap(cmap(range(36)))

        min_val = min(y_pred.min().item(), y_test.min().item())
        max_val = max(y_pred.max().item(), y_test.max().item())

        abs_error = torch.abs(y_test - y_pred) / torch.linalg.norm(y_test, dim=(1, 2))[:, None, None]
        min_abs_error = 0.
        max_abs_error = abs_error.max().item()

        f = 1.5
        fig, ax = plt.subplots(5, 3, sharex=True, sharey=True, figsize=(6*f, 8*f))

        for i in range(5):
            index = (i + 1) * size - 1
            relative_l2 = torch.linalg.norm(y_test[index] - y_pred[index]) / torch.linalg.norm(y_test[index])

            tensors_dict = self._hypersonic_mask_domain(y_test, y_pred, abs_error, index)
            # for key, tensor in tensors_dict.items():
            #     tensors_dict[key] = torch.where(tensor == -20., -15., tensor)
            # min_val = max(min_val, -15.)
            masked_pred = tensors_dict['masked_pred']
            masked_test = tensors_dict['masked_test']
            unmasked_pred = tensors_dict['unmasked_pred']
            unmasked_test = tensors_dict['unmasked_test']
            error_map = tensors_dict['masked_error']

            im_pred = ax[i, 0].contourf(masked_pred.T, levels=100, cmap=cmap, vmin=min_val, vmax=max_val, origin='upper')
            ax[i, 0].tick_params(axis=u'both', which=u'both', length=0)

            im_test = ax[i, 1].contourf(masked_test.T, levels=100, cmap=cmap, vmin=min_val, vmax=max_val, origin='upper')
            ax[i, 1].tick_params(axis=u'both', which=u'both', length=0)

            error_map = torch.abs(unmasked_test - unmasked_pred).T / torch.linalg.norm(y_test[index], ord=2)            
            im_error = ax[i, 2].contourf(error_map, levels=100, cmap=cmap, vmin=min_abs_error, vmax=max_abs_error, origin='upper')
            ax[i, 2].tick_params(axis=u'both', which=u'both', length=0)

            if i == 0:
                ax[i, 0].set_title('Prediction', fontsize=20, y=1.05)
                ax[i, 1].set_title('Ground truth', fontsize=20, y=1.05)
                ax[i, 2].set_title('Error map', fontsize=20, y=1.05)
            
            ax[i, 0].set_ylabel(fr"Mach: $M$ = {t[index]:.2f}" + f"\n $L^2$: {relative_l2:.2e}", 
                                fontsize=18, rotation=0, labelpad=80)

        fig.colorbar(im_test, ax=ax[:, :2].ravel().tolist(), shrink=0.67)
        cbar = fig.colorbar(im_error, ax=ax[:, 2].ravel().tolist())

        ticks = cbar.get_ticks()

        # Format tick labels in scientific notation as strings
        tick_labels = [f'{tick:.1e}' for tick in ticks]

        # Set the custom tick labels on the colorbar's axes
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        fig.suptitle('Hypersonics Predictions for Different Mach Numbers', fontsize=24, y=0.95, x=0.475)

        fig_name = 'full_snapshots' 
        if self.configs.noise_level > 0:
            fig_name += f'_noise_{int(self.configs.noise_level * 100)}'
        plt.savefig(os.path.join(self.path_figs, name, fig_name + '.png'), bbox_inches='tight', dpi=300)
    
    def _hypersonic_mask_domain(self, y_test, y_pred, abs_error, index):
        # y_test_to_plot = torch.where(y_test[index] == -20, 0, y_test[index])
        # y_pred_to_plot = torch.where(y_pred[index] == -20, 0, y_pred[index])
        y_test_to_plot = y_test[index]
        y_pred_to_plot = y_pred[index]
        abs_error_to_plot = abs_error[index]

        # Create mask for the two domains
        max_points = np.zeros(y_test_to_plot.shape[1])
        for j in range(y_test_to_plot.shape[1]):
            non_zero = np.where(y_test_to_plot[:, j] != -20)[0]
            if len(non_zero) > 0:
                max_points[j] = non_zero.max()
            else:
                max_points[j] = np.nan

        interior_mask = torch.zeros_like(y_test_to_plot)
        for j in range(y_test_to_plot.shape[1]):
            if np.isnan(max_points[j]):
                continue
            interior_mask[:int(max_points[j]), j] = 1

        # Extend the mask to the last pixel
        last_point = int(max_points[y_test_to_plot.shape[1] - 2])
        interior_mask[:last_point, 255] = 1

        tensors_dict = {}

        for tensor_name in ['pred', 'test', 'error']:
            if tensor_name == 'error':
                tensor = deepcopy(abs_error_to_plot)
            elif tensor_name == 'test':
                tensor = deepcopy(y_test_to_plot)
            else:
                tensor = deepcopy(y_pred_to_plot)

            # Mask the ground truth
            masked_tensor = torch.where(interior_mask != 1, np.nan, tensor)
            flip270 = torch.rot90(masked_tensor, k=3)
            flip270_lr = torch.fliplr(flip270)
            concat_tensor = torch.cat((flip270, flip270_lr), dim=1)
            tensors_dict['masked_' + tensor_name] = concat_tensor

            unmasked_tensor = deepcopy(tensor)
            flip270 = torch.rot90(unmasked_tensor, k=3)
            flip270_lr = torch.fliplr(flip270)
            concat_tensor = torch.cat((flip270, flip270_lr), dim=1)
            tensors_dict['unmasked_' + tensor_name] = concat_tensor
        return tensors_dict
  

    def plot_3d(self, y_test, y_pred, t, name):
        x_axis = np.linspace(0, np.pi, y_pred.shape[1])
        y_axis = np.linspace(0, np.pi, y_pred.shape[2])
        z_axis = np.linspace(0, np.pi, y_pred.shape[3])
        x_mesh, y_mesh, z_mesh = np.meshgrid(x_axis, y_axis, z_axis)
        ax = None

        # size = t.shape[0] // 5
        # indices = [(i + 1) * size - 1 for i in range(5)]
        # snapshots_pred = []
        # snapshots_test = []

        thresh = 0.01
        for n in range(t.shape[0]):
            to_plot_pred = copy(y_pred[n].flatten())
            to_plot_test = copy(y_test[n].flatten())

            relative_l_inf = np.linalg.norm(
                to_plot_test - to_plot_pred, ord=np.inf) / np.linalg.norm(to_plot_test, ord=np.inf)
            relative_l2 = np.linalg.norm(
                to_plot_test - to_plot_pred) / np.linalg.norm(to_plot_test)

            to_plot_pred[np.abs(to_plot_pred) <= thresh] = np.nan
            to_plot_test[np.abs(to_plot_test) <= thresh] = np.nan

            # if n in indices:
            #     snapshots_pred += [{'to_plot': to_plot_pred, 'time': t[n], 'l2': relative_l2}]
            #     snapshots_test += [{'to_plot': to_plot_test, 'time': t[n], 'l2': relative_l2}]

            fig = plt.figure()

            fig.suptitle(f"Time: t = {t[n]:.3f}" +
                         f"\n $L^2$: {relative_l2:.2e}",
                         #  f"     $L^\infty$: {relative_l_inf:.2e}",
                         fontsize=26, y=0.98)
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            scat = ax.scatter(x_mesh, y_mesh, z_mesh, c=to_plot_pred,
                              cmap='winter', vmin=-0.5, vmax=0.5, s=10 * to_plot_pred)
            ax.set_title('Prediction')
            ax.grid(False)
            ax.set_xbound(0, np.pi)
            ax.set_ybound(0, np.pi)
            ax.set_zbound(0, np.pi)

            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            scat = ax2.scatter(x_mesh, y_mesh, z_mesh, c=to_plot_test, cmap='winter',
                               vmin=-0.5, vmax=0.5, s=10 * to_plot_pred)
            ax2.set_title('Ground truth')

            ax2.grid(False)
            ax2.set_xbound(0, np.pi)
            ax2.set_ybound(0, np.pi)
            ax2.set_zbound(0, np.pi)

            fig.colorbar(scat, ax=[ax, ax2])
            plt.savefig(os.path.join(self.path_figs, str(n) + '_.png'))

            if (n % self.save_every == 0) or (n == self.configs.time_steps_inference - 1):
                plt.savefig(
                    (os.path.join(self.path_figs, name, f'snapshot_{n}')))

            plt.close()

        save_gif(in_dir=self.path_figs,
                 out_dir=os.path.join(self.path_figs, name),
                 gif_name='animation.gif')

    def plot_map_2d(self, y_test, y_pred, t, name):
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rc('font', family='serif')
        plt.rcParams.update({
                            "font.family": "serif",
                            'font.size': 11,
                            'lines.linewidth': 3,
                            'axes.labelsize': 11, 
                            'axes.titlesize': 11,
                            'xtick.labelsize': 11,
                            'ytick.labelsize': 11,
                            'legend.fontsize': 11,
                            'axes.linewidth': 2})

        lon = np.linspace(0, 355, num=144)
        lat = np.linspace(90, -87.5, num=72)

        lon[-1] = 360

        i = 0
        idxlist = [i]
        i = idxlist[0]

        colorbar_var = None
        min_val = min(y_pred.min().item(), y_test.min().item())
        max_val = max(y_pred.max().item(), y_test.max().item())
        
        for i in tqdm(range(t.shape[0])):
            # Create figure
            fig = plt.figure(figsize=(12 * 3, 6 * 2))
            # min_val = min(y_pred[i].min().item(), y_test[i].min().item())
            # max_val = max(y_pred[i].max().item(), y_test[i].max().item())

            relative_l2 = (torch.linalg.norm(y_test[i] - y_pred[i]) / torch.linalg.norm(y_test[i])).item()

            # Map 1
            ax1 = fig.add_subplot(131)

            map = Basemap(projection='cyl', llcrnrlon=0., llcrnrlat=-85., urcrnrlon=360., urcrnrlat=85., resolution='i')

            map.drawcoastlines()
            map.drawstates()
            map.drawcountries()
            map.drawlsmask(land_color='Linen', ocean_color='#CCFFFF')

            parallels = np.arange(-82.5, 82.5, 30)
            meridians = np.arange(0., 355, 30)
            map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
            map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)

            lons, lats = np.meshgrid(lon, lat)
            x, y = lons, lats

            temp = map.contourf(x, y, y_test[i, :, :], vmin=min_val, vmax=max_val, levels=50, cmap='coolwarm')
            plt.title('Measured surface air temperature in K')

            # Map 2
            ax2 = fig.add_subplot(132)

            map = Basemap(projection='cyl', llcrnrlon=0., llcrnrlat=-85., urcrnrlon=360., urcrnrlat=85., resolution='i')

            map.drawcoastlines()
            map.drawstates()
            map.drawcountries()
            map.drawlsmask(land_color='Linen', ocean_color='#CCFFFF')

            parallels = np.arange(-82.5, 82.5, 30)
            meridians = np.arange(0., 355, 30)
            map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
            map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)

            lons, lats = np.meshgrid(lon, lat)
            x, y = lons, lats

            temp = map.contourf(x, y, y_pred[i, :, :], vmin=min_val, vmax=max_val, levels=50, cmap='coolwarm')
            plt.title('Predicted surface air temperature in K')

            # if i == 0:
            colorbar_var = copy(temp)
            fig.colorbar(colorbar_var, ax=[ax1, ax2], shrink=0.5, pad=0.05)

            # Map 3
            ax3 = fig.add_subplot(133)

            map = Basemap(projection='cyl', llcrnrlon=0., llcrnrlat=-85., urcrnrlon=360., urcrnrlat=85., resolution='i')

            map.drawcoastlines()
            map.drawstates()
            map.drawcountries()
            map.drawlsmask(land_color='Linen', ocean_color='#CCFFFF')

            parallels = np.arange(-82.5, 82.5, 30)
            meridians = np.arange(0., 355, 30)
            map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
            map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)

            lons, lats = np.meshgrid(lon, lat)
            x, y = lons, lats

            abs_error = torch.abs(y_pred[i, :, :] - y_test[i, :, :])
            min_val = torch.min(abs_error)
            max_val = torch.max(abs_error)
            temp = map.contourf(x, y, abs_error, vmin=min_val, vmax=max_val, levels=50, cmap='coolwarm')
            plt.title('Absolute error')
            
            # if i == 0:
            colorbar_var = copy(temp)
            fig.colorbar(colorbar_var, ax=[ax3], shrink=0.5, pad=0.05)

            fig.suptitle(f"Time: t = {i}" +
                f"\n $L^2$: {relative_l2:.2e}", fontsize=26)
                # , y=0.75, x=0.5 * (ax1.get_position().x1 + ax2.get_position().x0))
            
            plt.savefig((os.path.join(self.path_figs, name, f'snapshot_{i}')))
        save_gif_climate(in_dir=os.path.join(self.path_figs, name),
                 out_dir=os.path.join(self.path_figs, name),
                 gif_name='animation.gif', remove_images=False)
        return


def plot_rnn_2d(y, out):
    fig, ax = plt.subplots(3, out.shape[2], figsize=(5 * out.shape[2], 10))
    for t in range(out.shape[2]):
        ax[0, t].imshow(y[0, 0, t, ...].cpu().detach(), cmap="twilight")
        ax[1, t].imshow(out[0, 0, t, ...].cpu().detach(), cmap="twilight")
        ax[2, t].imshow(torch.abs(y[0, 0, t, ...] - out[0, 0, t, ...]).cpu().detach(), cmap="twilight")
        l2_error = torch.norm(y[0, 0, t, ...] - out[0, 0, t, ...]) / torch.norm(y[0, 0, t, ...])
        ax[0, t].set_title(f"True: {t}")
        ax[1, t].set_title(f"Pred: {t}")     
        ax[2, t].set_title(f"L2: {l2_error:.2e}")
    fig.savefig("test.png")
    plt.close()
