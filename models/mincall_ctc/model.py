"""Implementation of the MinCall model

Based on:
https://arxiv.org/abs/1904.10337
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseCTCModel
from layers import MinCallConvBlock

class MinCallCTC(BaseCTCModel):
    """Skeleton Model
    """
    def __init__(self, convolution = None, decoder = None, *args, **kwargs):
        super(MinCallCTC, self).__init__(*args, **kwargs)
        """
        Args:
           convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
           decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
        """        

        self.convolution = convolution
        self.decoder = decoder

        self.load_default_configuration()
        
    def forward(self, x):
        
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.decoder(x)
        return x
        
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network
        """

        num_layers = 72
        pool_every = 24
        kernel_size = 3
        padding = 'same'
        num_channels = 64
        max_pool_kernel = 2

        if self.convolution is None or default_all:
            layers = list()
            # add initial convolution to model to put to 64 channels
            # otherwise the residual connections in the block fails
            # since it tries to add tensors with 1 channel and 64 channels
            layers.append(nn.Conv1d(1, num_channels, kernel_size, 1, padding)) 
            for i in range(num_layers):
                if i % pool_every == 0 and i > 0:
                    layers.append(nn.MaxPool1d(max_pool_kernel))
                layers.append(MinCallConvBlock(kernel_size, num_channels, num_channels, padding))

            self.convolution = nn.Sequential(*layers)
        
        if self.decoder is None or default_all:
            self.decoder = nn.Sequential(nn.Linear(num_channels, 5), nn.LogSoftmax(-1))

