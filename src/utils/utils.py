import os
import logging
import torch
import numpy as np
import scipy.io
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

import operator
from functools import reduce


#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# reading data
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


class NumpyReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True, mmap_mode='r'):
        super(NumpyReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.mmap_mode = mmap_mode

        self.file_path = file_path

        self.data = None
        self._load_file()

    def _load_file(self):
        self.data = np.load(self.file_path, mmap_mode=self.mmap_mode)

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self):
        x = self.data

        # if not self.old_mat:
        #     x = x[()]
        #     x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

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


# normalization, Gaussian
class GaussianNormalizer:
    def __init__(self, x, eps=0.00001, scale=False):
        super(GaussianNormalizer, self).__init__()

        self.scale = scale
        if scale:
            self.min = torch.min(x)
            self.max = torch.max(x)
        else:
            self.mean = torch.mean(x)
            self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        if self.scale:
            x = (x - self.min) / (self.max - self.min + self.eps)
        else:
            x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if self.scale:
            x = x * (self.max - self.min + self.eps) + self.min
        else:
            x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# loss function with rel/abs Lp loss
class LpLoss:
    def __init__(self, d=2, p=2, size_average=True, reduction=True, use_relative=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.use_relative = use_relative

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        # h = 1.
        h = 1.0 / (x.size()[-1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        if self.use_relative:
            return self.rel(x, y)
        else:
            return self.abs(x, y)

# print the number of parameters


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    # Format c as comma separated value
    c = "{:,}".format(c)
    return c


def extract_number(column_name):
    return int(column_name.split('=')[1].strip().split('$')[0].strip())


def save_metrics_to_text(final_metrics, path_model, metric_name, noise=0):
    metrics_df = pd.DataFrame(final_metrics)
    metrics_df = metrics_df.T
    metrics_df = metrics_df[sorted(metrics_df.columns, key=extract_number)]
    # metrics_df[r"$\sigma$"] = metrics_df.std(axis=1, numeric_only=True)
    metrics_latex = metrics_df.to_latex(index=True,
                                        float_format="{:0.4f}".format,
                                        column_format="cccccc")

    metrics_path = os.path.join('..', 'outputs', 'metrics')
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    experiment_name = path_model.split(os.path.sep)[-1][:-4]
    if noise > 0:
        experiment_name += f'_noise_{int(noise * 100)}'
    filename = os.path.join(metrics_path, experiment_name)

    if not os.path.exists(filename):
        os.makedirs(filename)
    metrics_df.to_csv(os.path.join(filename, f'{metric_name}'  + '.csv'))
    filename = os.path.join(filename, f'{metric_name}' + '.txt')
    

    with open(filename, "w") as f:
        f.write(metrics_latex)
    return metrics_df


def save_sr_metrics_to_text(sr_metrics, path_model, model_name):
    metrics_df = pd.DataFrame(sr_metrics[model_name])
    metrics_latex = metrics_df.to_latex(index=True,
                                        float_format="{:0.4f}".format,
                                        column_format="cccccc")

    metrics_path = os.path.join('..', 'outputs', 'metrics')
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    filename = os.path.join(metrics_path, path_model.split(os.path.sep)[-1][:-4] + f'_{model_name}_sr' + '.txt')
    with open(filename, 'w') as f:
        f.write(metrics_latex)
    return metrics_df


def rmse_loss(y_pred, y_true):
    return torch.nn.MSELoss(reduction='none')(y_pred, y_true).mean(axis=1).sqrt().sum()


def get_logger(name='Logger'):
    logger = logging.Logger(name=name)
    c_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def correlation_over_time(full_solution):
    sns.set(font_scale=2)
    correlations_ic = []
    correlations_middle = []
    t = np.linspace(0, full_solution.shape[1], full_solution.shape[1])
    for i in range(full_solution.shape[1]):
        corr_ic = pearsonr(full_solution[:, 0], full_solution[:, i])[0]
        corr_mid = pearsonr(full_solution[:, 20], full_solution[:, i])[0]
        correlations_ic.append(corr_ic)
        correlations_middle.append(corr_mid)
    plt.plot(t, correlations_ic, label=r'Correlation between $u(x,0)$ and $u(x,t)$', linewidth=3, color='red')
    plt.plot(t, correlations_middle, label=r'Correlation between $u(x,t_{20})$ and $u(x,t)$', linewidth=3)
    plt.plot(t, [np.min(correlations_middle)] * len(correlations_middle), 'k--')
    plt.title('Correlation over time: Navier-Stokes equations', fontsize=40)
    plt.xlabel('Time-step')
    plt.ylabel('Correlation')
    plt.legend()
    plt.show()