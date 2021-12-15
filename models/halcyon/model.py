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


class Halcyon(BaseModelS2S):
    """Halcyon Model
    """
    def __init__(self, 
        convolution = None, 
        encoder = None, 
        decoder = None, 
        default_all = False, 
        *args, **kwargs,
        ):
        super(Halcyon, self).__init__(*args, **kwargs)
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            encoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            default_all (bool): whether to load all the modules as default
        """
        
        self.load_default_configuration(default_all)
            
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network

        Based on json file in the dropbox link in: https://github.com/relastle/halcyon/blob/master/halcyon/infer/download.py

        """

        scheduled_sampling = 0.3
        strides = [1] * 5
        num_kernels = [1, 3, 5, 7, 3]
        paddings = ['same'] * 5
        num_channels = [64, 96, 128, 160, 32]
        use_bn = [True] * 5

        if self.convolution is None or default_all:
            self.convolution = nn.Sequential(
                HalcyonCNNBlock( 1, 64, 3, 2, "valid", True),
                HalcyonCNNBlock(64, 64, 3, 1, "valid", True),
                HalcyonCNNBlock(64, 128, 3, 1, "valid", True),
                nn.MaxPool1d(3, 2, "valid"),
                HalcyonCNNBlock(128, 160, 3, 1, "valid", True),
                HalcyonCNNBlock(160, 384, 3, 1, "valid", True),
                nn.MaxPool1d(3, 2, "valid"),
                HalcyonInceptionBlock(384, num_channels, num_kernels, strides, paddings, use_bn, scaler = 0.8**1),
                HalcyonInceptionBlock(384, num_channels, num_kernels, strides, paddings, use_bn, scaler = 0.8**2),
                HalcyonInceptionBlock(307, num_channels, num_kernels, strides, paddings, use_bn, scaler = 0.8**3),
            ) # output 245 channels

        if self.encoder is None or default_all:
            self.encoder = nn.LSTM(
                input_size = 245, 
                hidden_size = 128, 
                num_layers = 5, 
                bidirectional = True
            )
            
        if self.decoder is None or default_all:
            embedding = nn.Embedding(7, 16)
            rnn = nn.LSTM(96, 96, num_layers = 5, bidirectional = False)
            attention = LuongAttention('dot', 96)
            out_linear = nn.Linear(96, 7)

            self.decoder = DecoderS2S(
                embedding = embedding, 
                rnn = rnn, 
                attention = attention, 
                out_linear = out_linear, 
                encoder_hidden = 256, 
                upstream_attention = True
            )
