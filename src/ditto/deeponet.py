from typing import Iterator
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter


class DeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers, output_shape, configs):
        super(DeepONet, self).__init__()
        self.branch = BranchResNet(configs=configs) if branch_layers == 'resnet' else MLP(branch_layers)
        self.trunk = MLP(trunk_layers)
        self.output_shape = output_shape

    def forward(self, u0, grid):
        """
        Branch input size: (batch_size, 1, n_x)
        Trunk input size: (n_x * n_t, 2)
        
        Branch output size: (batch_size, latent_dim)
        Trunk output size: (n_x * n_t, latent_dim)
        """
        batch_size = u0.shape[0]
        branch_output = self.branch(u0.flatten(start_dim=1))
        trunk_output = self.trunk(grid)
        out = torch.einsum("bi,ni->bn", branch_output, trunk_output)
        return out.reshape((batch_size,) + self.output_shape)


class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            # nn.init.xavier_normal_(layer.weight.data)
            # layer.bias.data.zero_()
            self.layers.append(layer)   
        self.activation = nn.Tanh()
    
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x


class BranchResNet(nn.Module):
    def __init__(self, configs):
        super(BranchResNet, self).__init__()
        
        filters = 16
        self.conv1 = nn.Conv2d(1, filters, kernel_size=1, padding='valid')
        self.conv_block1 = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, padding='valid'), 
                                         nn.BatchNorm2d(filters),
                                         nn.ReLU(), 
                                         nn.Conv2d(filters, filters, kernel_size=3, padding='valid'), 
                                         nn.BatchNorm2d(filters),
                                         nn.ReLU(), 
                                         nn.Conv2d(filters, filters, kernel_size=3, padding='valid'), 
                                         nn.BatchNorm2d(filters),
                                         nn.ReLU(), 
                                        )
        
        self.conv2 = nn.Conv2d(filters, filters * 2, kernel_size=1, padding='valid')
        self.conv_block2 = nn.Sequential(nn.Conv2d(filters * 2, filters * 2, kernel_size=3, padding='valid'), 
                                         nn.BatchNorm2d(filters * 2),
                                         nn.ReLU(), 
                                         nn.Conv2d(filters * 2, filters * 2, kernel_size=3, padding='valid'), 
                                         nn.BatchNorm2d(filters * 2),
                                         nn.ReLU(), 
                                         nn.Conv2d(filters * 2, filters * 2, kernel_size=3, padding='valid'), 
                                         nn.BatchNorm2d(filters * 2),
                                         nn.ReLU(), 
                                        )
        
        self.conv3 = nn.Conv2d(filters * 2, filters * 4, kernel_size=1, padding='valid')
        self.conv_block3 = nn.Sequential(nn.Conv2d(filters * 4, filters * 4, kernel_size=3, padding='valid'), 
                                         nn.BatchNorm2d(filters * 4),
                                         nn.ReLU(), 
                                         nn.Conv2d(filters * 4, filters * 4, kernel_size=3, padding='valid'), 
                                         nn.BatchNorm2d(filters * 4),
                                         nn.ReLU(), 
                                         nn.Conv2d(filters * 4, filters * 4, kernel_size=3, padding='valid'), 
                                         nn.BatchNorm2d(filters * 4),
                                         nn.ReLU(),
                                        )

        self.final_conv = nn.Conv2d(filters * 4, 1, kernel_size=1, padding='valid')

        self.nx = configs.nx
        self.ny = configs.ny
        self.final_mlp = nn.Linear((self.nx - 18) * (self.ny - 18), configs.deeponet.latent_dim)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, self.nx, self.ny))
        
        x = self.conv1(x)

        # Block 1
        x1 = x
        x = self.conv_block1(x)
        # x = x + x1

        # Block 2
        x = self.conv2(x)
        x2 = x
        x = self.conv_block2(x)
        # x = x + x2
        
        # Block 3
        x = self.conv3(x)
        x3 = x
        x = self.conv_block3(x)
        # x = x + x3
        
        x = self.final_conv(x)

        # x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1)
        x = self.final_mlp(x)
        return x
