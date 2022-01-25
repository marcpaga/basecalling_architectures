"""Implementation of the SkeletonModel

Based on: 
https://github.com/nanoporetech/bonito
"""

import os
import sys
import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelImpl
from layers.layers import PositionalEncoding
from layers.bonito import BonitoLinearCRFDecoder
from layers.catcaller import CATCallerEncoderLayer
from constants import CRF_STATE_LEN, CRF_BIAS, CRF_SCALE, CRF_BLANK_SCORE , CRF_N_BASE 


class CATCallerModel(BaseModelImpl):
    """URNano model

    Based on: https://github.com/yaozhong/URnano/blob/master/models/model_unet.py
    """
    def __init__(self, convolution = None, pe = None, encoder = None, decoder = None, load_default = False, *args, **kwargs):
        super(CATCallerModel, self).__init__(*args, **kwargs)
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

        d_model = 512
        padding = 1
        kernel = 3
        stride = 2
        dilation = 1

        self.padding = padding
        self.kernel_size = kernel
        self.stride = stride
        self.dilation = dilation

        cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=d_model//2,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False),
            nn.BatchNorm1d(num_features=d_model//2),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=d_model//2,
                out_channels=d_model,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False),
            nn.BatchNorm1d(num_features=d_model),
            nn.ReLU()
        )

        return cnn

    def build_encoder(self):

        d_model = 512
        d_ff = 512
        dropout = 0.1
        num_encoder_layers = 6
        embed_dims = 512
        heads = 4
        kernel_size = [3, 7, 15, 31, 31, 31]
        weight_softmax = True
        weight_dropout = 0.1
        with_linear = True
        glu = True

        pe = PositionalEncoding(
                d_model = d_model, 
                dropout = dropout, 
                max_len = 5000
            )
        encoder = nn.ModuleList([CATCallerEncoderLayer(
            d_model = d_model, 
            d_ff = d_ff, 
            kernel_size = kernel_size[i], 
            num_heads = heads, 
            channels = embed_dims, 
            dropout = dropout, 
            weight_softmax = weight_softmax, 
            weight_dropout = weight_dropout, 
            with_linear = with_linear,
            glu = glu,
        ) for i in range(num_encoder_layers)])

        encoder = nn.Sequential(pe, *encoder)
        return encoder

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 512, 
            'cnn_output_activation': 'relu',
            'encoder_input_size': 512,
            'encoder_output_size': 512,
            'cnn_stride': 4,
        }
        return defaults
        
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network

        Based on: https://github.com/lvxuan96/CATCaller/blob/master/train/apex/train_litetr_dist.py
        """

        if self.convolution is None or default_all:
            self.convolution = self.build_cnn()
            self.cnn_stride = self.get_defaults()['cnn_stride']
        if self.encoder is None or default_all:
            self.encoder = self.build_encoder()
        if self.decoder is None or default_all:
            self.decoder = self.build_decoder(encoder_output_size = 512, decoder_type = 'ctc')
            self.decoder_type = 'ctc'

