"""Implementation of the SkeletonModel

Based on: 
https://github.com/relastle/halcyon/blob/master/halcyon/ml/net/cnn.py
and the config file downloaded from their github.
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelS2S
from layers.halcyon import HalcyonCNNBlock, HalcyonInceptionBlock
from layers.layers import DecoderS2S
from layers.attention import LuongAttention


class HalcyonModel(BaseModelS2S):
    """Halcyon Model
    """
    def __init__(self, 
        convolution = None, 
        encoder = None, 
        decoder = None, 
        load_default = False, 
        scheduled_sampling = 0.3,
        *args, **kwargs,
        ):
        super(HalcyonModel, self).__init__(scheduled_sampling = scheduled_sampling, *args, **kwargs)
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            encoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            load_default (bool): whether to load the modules as default
        """

        self.convolution = convolution
        self.encoder = encoder
        self.decoder = decoder
        
        if load_default:
            self.load_default_configuration(load_default)

    def build_cnn(self, mod):

        strides = [1] * 5
        num_kernels = [1, 3, 5, 7, 3]
        paddings = ['same'] * 5
        num_channels = [64, 96, 128, 160, 32]
        use_bn = [True] * 5
        if mod:
            padding = 'same'
        else:
            padding = 'valid'

        cnn = nn.Sequential(
            HalcyonCNNBlock( 1, 64, 3, 2, padding, True),
            HalcyonCNNBlock(64, 64, 3, 1, padding, True),
            HalcyonCNNBlock(64, 128, 3, 1, padding, True),
            nn.MaxPool1d(3, 2),
            HalcyonCNNBlock(128, 160, 3, 1, padding, True),
            HalcyonCNNBlock(160, 384, 3, 1, padding, True),
            nn.MaxPool1d(3, 2),
            HalcyonInceptionBlock(384, num_channels, num_kernels, strides, paddings, use_bn, scaler = 0.8**1),
            HalcyonInceptionBlock(382, num_channels, num_kernels, strides, paddings, use_bn, scaler = 0.8**2),
            HalcyonInceptionBlock(304, num_channels, num_kernels, strides, paddings, use_bn, scaler = 0.8**3),
        ) # output 243 channels
        return cnn    

    def build_encoder(self):
        encoder = nn.LSTM(
                input_size = 243, 
                hidden_size = 128, 
                num_layers = 5, 
                bidirectional = True
            )
        return encoder

    def build_decoder(self):
        embedding = nn.Embedding(7, 16)
        rnn = nn.LSTM(96, 96, num_layers = 5, bidirectional = False)
        attention = LuongAttention('dot', 96)
        out_linear = nn.Linear(96, 7)

        decoder = DecoderS2S(
            embedding = embedding, 
            rnn = rnn, 
            attention = attention, 
            out_linear = out_linear, 
            encoder_hidden = 256, 
            upstream_attention = True
        )
        return decoder

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 243, 
            'cnn_output_activation': 'relu',
            'encoder_input_size': 243,
            'encoder_output_size': 256,
            'decoder_input_size': 256,
        }
        return defaults

    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network

        Based on json file in the dropbox link in: https://github.com/relastle/halcyon/blob/master/halcyon/infer/download.py

        """

        if self.convolution is None or default_all:
            self.convolution = self.build_cnn(mod = False)

        if self.encoder is None or default_all:
            self.encoder = self.build_encoder()
            
        if self.decoder is None or default_all:
            self.decoder = self.build_decoder()
