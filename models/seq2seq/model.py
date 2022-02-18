"""Implementation of a S2S Model
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))

from classes import BaseModelS2S
from layers.layers import RNNDecoderS2S
from layers.attention import LuongAttention
from constants import S2S_OUTPUT_CLASSES

from bonito.model import BonitoModel
from halcyon.model import HalcyonModelS2S

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

    def build_cnn(self, cnn_type):
        
        if cnn_type == 'bonito':
            cnn = BonitoModel.build_cnn(self)
            self.cnn_output_size = BonitoModel.get_defaults(self)['cnn_output_size']
        elif cnn_type == 'halcyon':
            cnn = HalcyonModelS2S.build_cnn(self, mod=False)
            self.cnn_output_size = HalcyonModelS2S.get_defaults(self)['cnn_output_size']
        else:
            raise ValueError('invalid cnn_type: '+ str(cnn_type))
        return cnn

    def build_encoder(self, input_size, output_size, num_layers, bidirectional):

        encoder = nn.LSTM(
            input_size, 
            output_size, 
            num_layers = num_layers, 
            bidirectional = bidirectional
        )
        return encoder

    def build_decoder(self, embedding_dim, rnn_size, rnn_num_layers, attention_type, attention_pos, monotonic):

        embedding = nn.Embedding(S2S_OUTPUT_CLASSES, embedding_dim)
        
        if attention_type is not None:
            attention = LuongAttention(attention_type, rnn_size, monotonic)
            if attention_pos == 'upstream':
                rnn = nn.LSTM(rnn_size, rnn_size, num_layers = rnn_num_layers, bidirectional = False)
            elif attention_pos == 'downstream':
                rnn = nn.LSTM(embedding_dim, rnn_size, num_layers = rnn_num_layers, bidirectional = False)
        else:
            attention = None
            rnn = nn.LSTM(embedding_dim, rnn_size, num_layers = rnn_num_layers, bidirectional = False)

        
        out_linear = nn.LazyLinear(rnn_size, S2S_OUTPUT_CLASSES)

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

        self.convolution = self.build_cnn('bonito')
        self.cnn_stride = self.get_defaults()['cnn_stride']
        self.encoder = self.build_encoder(input_size = 384, output_size = 192, num_layers = 2, bidirectional = True)
        self.decoder = self.build_decoder(embedding_dim = 16, rnn_size = 384, rnn_num_layers = 1, attention = True, attention_type = 'dot', attention_pos = 'upstream')
            
