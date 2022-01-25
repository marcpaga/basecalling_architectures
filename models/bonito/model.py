"""Implementation of the Bonito-CTC model

Based on: 
https://github.com/nanoporetech/bonito
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelImpl
from layers.bonito import BonitoLSTM


class BonitoModel(BaseModelImpl):
    """Bonito Model
    """
    def __init__(self, convolution = None, encoder = None, decoder = None, reverse = True, load_default = False, *args, **kwargs):
        super(BonitoModel, self).__init__(*args, **kwargs)
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            encoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            reverse (bool): if the first rnn layer starts with reverse 
        """
    
        self.convolution = convolution
        self.encoder = encoder
        self.decoder = decoder
        self.reverse = reverse
        
        if load_default:
            self.load_default_configuration()

    def forward(self, x):
        """Forward pass of a batch
        
        Args:
            x (tensor) : [batch, channels (1), len]
        """
        
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def build_cnn(self):

        cnn = nn.Sequential(
            nn.Conv1d(
                in_channels = 1, 
                out_channels = 4, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(
                in_channels = 4, 
                out_channels = 16, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(
                in_channels = 16, 
                out_channels = 384, 
                kernel_size = 19, 
                stride= 5, 
                padding=19//2, 
                bias=True),
            nn.SiLU()
        )
        return cnn

    def build_encoder(self, input_size, reverse):

        if reverse:
            encoder = nn.Sequential(BonitoLSTM(input_size, 384, reverse = True),
                                    BonitoLSTM(384, 384, reverse = False),
                                    BonitoLSTM(384, 384, reverse = True),
                                    BonitoLSTM(384, 384, reverse = False),
                                    BonitoLSTM(384, 384, reverse = True))
        else:
            encoder = nn.Sequential(BonitoLSTM(input_size, 384, reverse = False),
                                    BonitoLSTM(384, 384, reverse = True),
                                    BonitoLSTM(384, 384, reverse = False),
                                    BonitoLSTM(384, 384, reverse = True),
                                    BonitoLSTM(384, 384, reverse = False))
        return encoder    

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 384, 
            'cnn_output_activation': 'silu',
            'encoder_input_size': 384,
            'encoder_output_size': 384,
            'cnn_stride': 5,
        }
        return defaults
        
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network
        """
        
        if self.convolution is None or default_all:
            self.convolution = self.build_cnn()
            self.cnn_stride = self.get_defaults()['cnn_stride']
        if self.encoder is None or default_all:
            self.encoder = self.build_encoder(input_size = 384, reverse = True)
        if self.decoder is None or default_all:
            self.decoder = self.build_decoder(encoder_output_size = 384, decoder_type = 'crf')
            self.decoder_type = 'crf'
