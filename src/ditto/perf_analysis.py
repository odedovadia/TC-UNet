import time
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

from src.utils.utils import (count_params, get_logger)
from src.utils.architecture_builder import build_archtitecture


def format_memory(nbytes):
    """Returns a formatted memory size string"""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if (abs(nbytes) >= GB):
        return '{:.2f} Gb'.format(nbytes * 1.0 / GB)
    elif (abs(nbytes) >= MB):
        return '{:.2f} Mb'.format(nbytes * 1.0 / MB)
    elif (abs(nbytes) >= KB):
        return '{:.2f} Kb'.format(nbytes * 1.0 / KB)
    else:
        return str(nbytes) + ' b'


def profile(configs, model, model_name, logger, mode='train'):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    num_iter = 200

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    start_mem = torch.cuda.max_memory_allocated()
    logger.info('Start mem: ' + format_memory(start_mem))


    batch_size = 100
    if 'ditto' in model_name:
        # batch_size = batch_size * configs.time_steps_train
        x = torch.randn(batch_size, 1, configs.nx)
        t = torch.randn(batch_size,).to(device)
        target = torch.randn(batch_size, 1, configs.nx)
    else:
        x = torch.randn(batch_size, 3, configs.time_steps_train, configs.nx)
        t = None
        target = torch.randn(batch_size, 3, 1, configs.nx)

    start_time = time.time()
    for _ in tqdm(range(num_iter)):
        if mode == 'train':
            out_, target_ = step(x, target, device, model, model_name, configs, t=t)
            loss = ((out_ - target_)**2).mean()
            loss.backward()
        else:
            with torch.no_grad():
                out_, target_ = step(x, target, device, model, model_name, configs, t=t)

    end_time = time.time()
    ips = round(num_iter / (end_time - start_time), 2)

    max_mem = torch.cuda.max_memory_allocated()
    max_mem = format_memory(max_mem)
    params = count_params(model)

    results_dict = {'Model': model_name, 
                    '# Trainable parameters': f'{params:,}', 
                    'Memory': max_mem, 
                    'IPS': ips, 
                    'Batch size': f'{batch_size:,}'}

    logger.info(f'Model: {model_name}' + ' | ' + f' Max memory: {max_mem}' + ' | ' + f'IPS: {ips}' \
        + ' | ' + f'Params: {params:,}')
    return results_dict


def step(x, target, device, model, model_name, configs, t=None):
    if 'subsampled' in model_name:
        indices = torch.randint(0, x.shape[0], (x.shape[0] // configs.subsample,))
        x_, t_, target_ = x[indices].to(device), t[indices].to(device), target[indices].to(device)
    else:
        if model_name not in ['FNO', 'UNet', 'RegUNet']:
            t_ = t.to(device)
        x_, target_ = x.to(device), target.to(device)

    if model_name not in ['FNO', 'UNet', 'RegUNet']:
        out = model(x_, t_)
    elif model_name == 'FNO':
        out = model(x_)
    elif model_name in ['UNet', 'RegUNet']:
        time_interp = max(32, int(16 * np.ceil(x_.shape[2] / 16.)))
        original_shape = x_.shape[2:]
        x_ = torch.nn.functional.interpolate(x_, size=(time_interp,) + x_.shape[3:])
        if configs.scenario == 'burgers':
            out = model(x_, time=None, use_grid=False)
            out = out[..., :original_shape[0], :original_shape[1]]
        else:
            out = model(x_, time=None)
            out = out[..., :original_shape[0], :original_shape[1], :original_shape[2]]
    return out, target


if __name__ == '__main__':
    from src.configs import BURGERS_CONFIGS

    logger = get_logger(name='profile')

    configs = BURGERS_CONFIGS

    results = []
    for model_name in configs.models:
        model = build_archtitecture(model_type=model_name, configs=configs, logger=logger, mode='train')
        results += [profile(configs, model, model_name, logger, mode='train')]
    df = pd.DataFrame(results)
    print(df)
    print(df.to_latex(index=True, float_format="{:0.4f}".format,  column_format="ccccc"))
    a = 1
    