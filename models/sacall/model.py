"""Implementation of the SkeletonModel

Based on: 
https://github.com/nanoporetech/bonito
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelImpl
from layers import PositionalEncoding
from layers.bonito import BonitoLinearCRFDecoder
from constants import CRF_STATE_LEN, CRF_BIAS, CRF_SCALE, CRF_BLANK_SCORE , CRF_N_BASE 


class SACall(BaseModelImpl):
    """URNano model

    Based on: https://github.com/yaozhong/URnano/blob/master/models/model_unet.py
    """
    def __init__(self, convolution = None, pe = None, encoder = None, decoder = None, *args, **kwargs):
        super(SACall, self).__init__(*args, **kwargs)
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
        
        self.load_default_configuration()

    def forward(self, x):
        """Forward pass of a batch
        
        Args:
            x (tensor) : [batch, channels (1), len]
        """
        
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.pe(x)
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
            
        
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network

        Based on: https://github.com/huangnengCSU/SACall-basecaller/blob/master/transformer/modules.py
        """
        
        d_model = 256
        kernel = 3
        stride = 1
        padding = 1
        dilation = 1 
        dropout = 0.1
        n_layers = 6
        n_head = 8
        d_ff = 1024


        if self.convolution is None or default_all:
            
            self.convolution = nn.Sequential(
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

        if self.pe is None or default_all:
            self.pe = PositionalEncoding(d_model, dropout, max_len = 4000)
        if self.encoder is None or default_all:
            encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout)
            self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        if self.decoder is None or default_all:
            if self.model_type == 'ctc':
                self.decoder = nn.Sequential(nn.Linear(d_model, 5), nn.LogSoftmax(-1))
            elif self.model_type == 'crf':
                self.decoder = BonitoLinearCRFDecoder(
                    insize = d_model, 
                    n_base = CRF_N_BASE, 
                    state_len = CRF_STATE_LEN, 
                    bias=CRF_BIAS, 
                    scale= CRF_SCALE, 
                    blank_score= CRF_BLANK_SCORE
                )
