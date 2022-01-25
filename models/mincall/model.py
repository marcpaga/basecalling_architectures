"""Implementation of the MinCall model

Based on:
https://arxiv.org/abs/1904.10337
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelImpl
from layers.mincall import MinCallConvBlock

class MinCallModel(BaseModelImpl):
    """Skeleton Model
    """
    def __init__(self, convolution = None, decoder = None, load_default = False, *args, **kwargs):
        super(MinCallModel, self).__init__(*args, **kwargs)
        """
        Args:
           convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
           decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
        """        

        self.convolution = convolution
        self.decoder = decoder

        if load_default:
            self.load_default_configuration()
        
    def forward(self, x):
        
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.decoder(x)
        return x

    def build_cnn(self):

        num_layers = 72
        pool_every = 24
        kernel_size = 3
        padding = 'same'
        num_channels = 64
        max_pool_kernel = 2

        layers = list()
        # add initial convolution to model to put to 64 channels
        # otherwise the residual connections in the block fails
        # since it tries to add tensors with 1 channel and 64 channels
        layers.append(nn.Conv1d(1, num_channels, kernel_size, 1, padding)) 
        for i in range(num_layers):
            if i % pool_every == 0 and i > 0:
                layers.append(nn.MaxPool1d(max_pool_kernel))
            layers.append(MinCallConvBlock(kernel_size, num_channels, num_channels, padding))

        cnn = nn.Sequential(*layers)
        return cnn

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 64, 
            'cnn_output_activation': None,
            'encoder_input_size': None,
            'encoder_output_size': None,
            'cnn_stride': 4
        }
        return defaults
        
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network
        """

        if self.convolution is None or default_all:
            self.convolution = self.build_cnn()
        
        if self.decoder is None or default_all:
            self.decoder = self.build_decoder(encoder_output_size = 384, decoder_type = 'ctc')
            self.decoder_type = 'ctc'

