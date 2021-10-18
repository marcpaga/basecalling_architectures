"""Implementation of the Skeleton model
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseCTCModel
from torch import nn
from layers import CausalCallConvBlock

class CausalCallCTC(BaseCTCModel):
    """CasualCallCTC Model
    """
    def __init__(self, convolution = None, decoder = None, *args, **kwargs):
        super(CausalCallCTC, self).__init__(*args, **kwargs)
        """
        Args:
           convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
           decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
        """        

        self.convolution = convolution
        self.decoder = decoder

        self.load_default_configuration()  

    def forward(self, x):
        """Forward pass of a batch
        """
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.decoder(x)
        return x
    
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network
        """
    
        num_blocks = 5
        num_channels = 256
        kernel_size = 3
        dilation_multiplier = 2
        dilation = 1

        if self.convolution is None or default_all:
            layers = list()
            for i in range(num_blocks):
                if i == 0:
                    layers.append(CausalCallConvBlock(kernel_size, num_channels, 1, dilation))
                else:
                    layers.append(CausalCallConvBlock(kernel_size, num_channels, int(num_channels/2), dilation))
                dilation *= dilation_multiplier
            self.convolution = nn.Sequential(*layers)

        if self.decoder is None or default_all:
            self.decoder = nn.Sequential(nn.Linear(int(num_channels/2), num_channels), 
                                         nn.ReLU(),
                                         nn.Linear(num_channels, 5),
                                         nn.LogSoftmax(-1))