"""Implementation of the SkeletonModel

Based on: 
https://github.com/nanoporetech/bonito
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelImpl
from layers.layers import PositionalEncoding

class SACallModel(BaseModelImpl):
    """SACall model

    Based on: https://github.com/huangnengCSU/SACall-basecaller/ctc/ctc_encoder.py
    """
    def __init__(self, convolution = None, pe = None, encoder = None, decoder = None, load_default = False, *args, **kwargs):
        super(SACallModel, self).__init__(*args, **kwargs)
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            pe (nn.Module): positional encoding module with: in [len, batch, channel]; out [len, batch, channel]
            encoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
        """
    
        self.convolution = convolution
        self.pe = pe
        self.encoder = encoder
        self.decoder = decoder
        
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
        
        d_model = 256
        kernel = 3
        stride = 2
        padding = 1
        dilation = 1 

        cnn = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = d_model//2,
                kernel_size = kernel,
                stride = stride,
                padding = padding,
                dilation = dilation,
                bias = False),
            nn.BatchNorm1d(num_features = d_model//2),
            nn.ReLU(inplace = True),
            nn.Conv1d(
                in_channels = d_model//2,
                out_channels = d_model,
                kernel_size = kernel,
                stride = stride,
                padding = padding,
                dilation = dilation,
                bias = False),
            nn.BatchNorm1d(num_features=d_model),
            nn.ReLU(inplace=True)
        )

        return cnn

    def build_encoder(self):

        d_model = 256
        dropout = 0.1
        n_layers = 6
        n_head = 8
        d_ff = 1024

        pe = PositionalEncoding(d_model, dropout, max_len = 4000)
        transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model = d_model, dim_feedforward = d_ff, nhead = n_head, dropout = dropout) for _ in range(n_layers)])
        transformer_layers = nn.Sequential(*transformer_layers)
        encoder = nn.Sequential(pe, transformer_layers)

        return encoder

    def get_defaults(self):

        defaults = {
            'cnn_output_size': 256, 
            'cnn_output_activation': 'relu',
            'encoder_input_size': 256,
            'encoder_output_size': 256,
            'cnn_stride': 4,
        }
        return defaults
            
        
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network

        Based on: https://github.com/huangnengCSU/SACall-basecaller/blob/master/transformer/modules.py
        """
    
        if self.convolution is None or default_all:
            self.convolution = self.build_cnn()
        if self.encoder is None or default_all:
            self.encoder = self.build_encoder()
        if self.decoder is None or default_all:
            self.decoder = self.build_decoder(encoder_output_size = 256, model_type = self.model_type)
