"""Implementation of a S2S Model
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelS2S
from layers.layers import RNNDecoderS2S
from layers.attention import LuongAttention
from constants import S2S_OUTPUT_CLASSES

class Seq2Seq(BaseModelS2S):
    """Halcyon Model
    """
    def __init__(self, 
        convolution = None,
        encoder = None,
        decoder = None,
        load_default = False,
        *args, **kwargs,
        ):
        super(Seq2Seq, self).__init__(*args, **kwargs)
        """
        See `BaseModelS2S` for args
        """
        self.convolution = convolution
        self.encoder = encoder
        self.decoder = decoder
        if load_default:
            self.load_default_configuration() 

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

    def build_encoder(self, input_size, output_size, num_layers, bidirectional):

        encoder = nn.LSTM(input_size, output_size, num_layers = num_layers, bidirectional = bidirectional)
        return encoder

    def build_decoder(self, embedding_dim, rnn_size, rnn_num_layers, attention, attention_type, attention_pos):

        embedding = nn.Embedding(S2S_OUTPUT_CLASSES, embedding_dim)
        
        if attention:
            attention = LuongAttention(attention_type, rnn_size)
        else:
            attention = None

        rnn = nn.LSTM(rnn_size, rnn_size, num_layers = rnn_num_layers, bidirectional = False)
        out_linear = nn.Linear(rnn_size, S2S_OUTPUT_CLASSES)

        decoder = RNNDecoderS2S(
            embedding = embedding, 
            rnn = rnn, 
            attention = attention, 
            out_linear = out_linear, 
            attention_pos = attention_pos
        )
        return decoder

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 384, 
            'cnn_output_activation': 'silu',
            'encoder_input_size': 384,
            'encoder_output_size': 192,
            'cnn_stride': 5,
        }
        return defaults

    def load_default_configuration(self):
        """Sets the default configuration for one or more
        modules of the network
        """

        self.convolution = self.build_cnn()
        self.cnn_stride = self.get_defaults()['cnn_stride']
        self.encoder = self.build_encoder(input_size = 384, output_size = 192, num_layers = 2, bidirectional = True)
        self.decoder = self.build_decoder(embedding_dim = 16, rnn_size = 384, rnn_num_layers = 1, attention = True, attention_type = 'dot', attention_pos = 'upstream')
            
