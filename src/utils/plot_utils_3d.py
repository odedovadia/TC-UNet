import os
import imageio
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def plot_3d(x, y, z, c_data, ax=None, step=0, show_slice=False):
    if not show_slice:
        to_plot = c_data.flatten()
        to_plot[np.abs(to_plot) <= 0.0005] = np.nan
        if ax is None:
            ax = plt.axes(projection='3d')
            scat = ax.scatter(x, y, z, c=to_plot, cmap='winter',
                              vmin=-0.5, vmax=0.5)
            ax.grid(False)
            ax.set_xbound(0, np.pi)
            ax.set_ybound(0, np.pi)
            ax.set_zbound(0, np.pi)
            plt.colorbar(scat)
        else:
            ax.cla()
            ax.scatter(x, y, z, c=to_plot,
                       s=10 * to_plot, cmap='winter')
            ax.grid(False)
            ax.set_xbound(0, np.pi)
            ax.set_ybound(0, np.pi)
            ax.set_zbound(0, np.pi)
            ax.set_title('Step: ' + str(step))
            # plt.pause(0.01)
    return ax


def save_gif(in_dir, out_dir, gif_name, remove_images=True):
    # Build GIF
    filenames = glob.glob(os.path.join(in_dir, '*.png'))
    print('Creating gif\n')
    with imageio.get_writer(os.path.join(out_dir, gif_name + '.gif'), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    writer.close()
    print('gif complete\n')
    if remove_images:
        print('Removing Images\n')
        # Remove files
        for filename in set(filenames):
            os.remove(filename)


def save_gif_climate(in_dir, out_dir, gif_name, remove_images=True):
    # Build GIF
    filenames = glob.glob(os.path.join(in_dir, '*.png'))
    print('Creating gif\n')
    frames = []
    for filename in filenames:
        image = imageio.imread(filename)
        frames.append(image)
    imageio.mimsave(os.path.join(out_dir, gif_name + '.gif'), frames, duration=200.)
    print('gif complete\n')
    if remove_images:
        print('Removing Images\n')
        # Remove files
        for filename in set(filenames):
            os.remove(filename)


def plot_3d_during_training(y_pred, thresh=0.05):
    x_axis = np.linspace(0, np.pi, y_pred.shape[1])
    y_axis = np.linspace(0, np.pi, y_pred.shape[2])
    z_axis = np.linspace(0, np.pi, y_pred.shape[3])
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_axis, y_axis, z_axis)
    to_plot = y_pred.cpu().flatten()
    to_plot[np.abs(to_plot) <= thresh] = np.nan

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    scat = ax.scatter(x_mesh, y_mesh, z_mesh, c=to_plot, cmap='winter',
                            vmin=-0.5, vmax=0.5)
    ax.grid(False)
    ax.set_xbound(0, np.pi)
    ax.set_ybound(0, np.pi)
    ax.set_zbound(0, np.pi)
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    scat = ax2.scatter(x_mesh, y_mesh, z_mesh, c=to_plot, cmap='winter',
                            vmin=-0.5, vmax=0.5)
    ax2.grid(False)
    ax2.set_xbound(0, np.pi)
    ax2.set_ybound(0, np.pi)
    ax2.set_zbound(0, np.pi)
    
    fig.colorbar(scat, ax=[ax, ax2])
    plt.show()
