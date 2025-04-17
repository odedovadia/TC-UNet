import os
import scipy
import torch
import h5py
import numpy as np


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float
        

def ns_mat_to_np():
    mat_path = "navier_stokes\\fno_data\\NavierStokes_V1e-5_N1200_T20.mat"
    reader  = MatReader(mat_path)
    x = reader.read_field('a')
    y = reader.read_field('u')
    t = reader.read_field('t').repeat(x.shape[0], 1)
    T = 20
    
    np.save(os.path.join('navier_stokes', 'x_' + str(int(T)) + '_T.npy'), x)
    np.save(os.path.join('navier_stokes', 'y_' + str(int(T)) + '_T.npy'), y)
    np.save(os.path.join('navier_stokes', 't_' + str(int(T)) + '_T.npy'), t)


def burgers_mat_to_np():
    T = 2
    N = 1000
    v = 0.001
    size = 1024

    save_path = os.path.join('burgers', f'Burgers_v_{v}_N_{N}_T_{T}_size_{size}')
    mat_path = save_path + '.mat'
    reader  = MatReader(mat_path)
    solution = reader.read_field('sol')
    
    y = solution[:, 1:, :].permute(0, 2, 1)
    x = solution[:, 0, :]
    t = torch.linspace(0, T, solution.shape[1])[1:].repeat(y.shape[0], 1)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'x' + '.npy'), x)
    np.save(os.path.join(save_path, 'y' + '.npy'), y)
    np.save(os.path.join(save_path, 't' + '.npy'), t)
    
if __name__ == "__main__":
    burgers_mat_to_np()
