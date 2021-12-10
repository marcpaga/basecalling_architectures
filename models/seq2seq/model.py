"""Implementation of a S2S Model
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelS2S
from layers.layers import DecoderS2S
from layers.attention import LuongAttention


class Seq2Seq(BaseModelS2S):
    """Halcyon Model
    """
    def __init__(self, 
        default_all = True,
        *args, **kwargs,
        ):
        super(Seq2Seq, self).__init__(*args, **kwargs)
        """
        See `BaseModelS2S` for args
        """
        self.load_default_configuration(default_all) 
        
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network

        """

        if self.convolution is None or default_all:
            self.convolution = nn.Sequential(
                nn.Conv1d(1, 4, kernel_size = 5, stride= 1, padding=5//2, bias=True),
                nn.SiLU(),
                nn.Conv1d(4, 16, kernel_size = 5, stride= 1, padding=5//2, bias=True),
                nn.SiLU(),
                nn.Conv1d(16, 384, kernel_size = 19, stride= 5, padding=19//2, bias=True),
                nn.SiLU()
            )

        if self.encoder is None or default_all:
            self.encoder = nn.LSTM(384, 96, num_layers = 2, bidirectional = True)
            
        if self.decoder is None or default_all:
            embedding = nn.Embedding(7, 16)
            rnn = nn.LSTM(192, 192, num_layers = 1, bidirectional = False)
            attention = LuongAttention('dot', 192)
            out_linear = nn.Linear(192, 7)

            self.decoder = DecoderS2S(
                embedding = embedding, 
                rnn = rnn, 
                attention = attention, 
                out_linear = out_linear, 
                encoder_hidden = 192, 
                upstream_attention = True
            )
