"""Implementation of the Skeleton model
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelImpl
from torch import nn
from layers.causalcall import CausalCallConvBlock

class CausalCallModel(BaseModelImpl):
    """CasualCall Model
    """
    def __init__(self, convolution = None, decoder = None, load_default = False, *args, **kwargs):
        super(CausalCallModel, self).__init__(*args, **kwargs)
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
        """Forward pass of a batch
        """
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.decoder(x)
        return x

    def build_cnn(self):

        num_blocks = 5
        num_channels = 256
        kernel_size = 3
        dilation_multiplier = 2
        dilation = 1
        
        layers = list()
        for i in range(num_blocks):
            if i == 0:
                layers.append(CausalCallConvBlock(kernel_size, num_channels, 1, dilation))
            else:
                layers.append(CausalCallConvBlock(kernel_size, num_channels, int(num_channels/2), dilation))
            dilation *= dilation_multiplier
        convolution = nn.Sequential(*layers)

        return convolution

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 128, 
            'cnn_output_activation': 'relu',
            'encoder_input_size': None,
            'encoder_output_size': None,
            'cnn_stride': 1,
        }
        return defaults
    
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network
        """

        if self.convolution is None or default_all:
            self.convolution = self.build_cnn()

        if self.decoder is None or default_all:
            self.decoder = self.build_decoder(encoder_output_size = 128, decoder_type = 'ctc')
            self.decoder_type = 'ctc'
