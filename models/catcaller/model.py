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


class CATCaller(BaseModelImpl):
    """URNano model

    Based on: https://github.com/yaozhong/URnano/blob/master/models/model_unet.py
    """
    def __init__(self, convolution = None, pe = None, encoder = None, decoder = None, *args, **kwargs):
        super(CATCaller, self).__init__(*args, **kwargs)
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

        signal_lengths = torch.full((x.shape[0],), x.shape[2], dtype = float, device = self.device)
        max_len = x.size(1)
        max_len = int(((max_len + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1))
        max_len = int(((max_len + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1))
        new_signal_lengths = ((signal_lengths + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1).int()
        new_signal_lengths = ((new_signal_lengths + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1).int()

        src_mask = torch.tensor([[0] * v.item() + [1] * (max_len - v.item()) for v in new_signal_lengths],
                                dtype=torch.uint8).unsqueeze(-2).to(self.device)
        src_mask = src_mask.squeeze(1)


        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.pe(x)
        x = x.permute(1, 0, 2) # [batch, len, channels]

        for encoder_layer in self.encoder:
            x = encoder_layer(x, src_mask)

        x = x.permute(1, 0, 2) # [len, batch, channels]
        x = self.decoder(x)
        
        return x
            
        
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network

        Based on: https://github.com/lvxuan96/CATCaller/blob/master/train/apex/train_litetr_dist.py
        """
        
        d_model = 512
        d_ff = 512
        dropout = 0.1
        padding = 1
        kernel = 3
        stride = 2
        dilation = 1

        num_encoder_layers = 6
        embed_dims = 512
        heads = 4
        kernel_size = [3, 7, 15, 31, 31, 31]
        weight_softmax = True
        weight_dropout = 0.1
        with_linear = True
        glu = True

        self.padding = padding
        self.kernel_size = kernel
        self.stride = stride
        self.dilation = dilation

        if self.convolution is None or default_all:
            self.convolution = nn.Sequential(
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

        if self.pe is None or default_all:
            self.pe = PositionalEncoding(
                d_model = d_model, 
                dropout = dropout, 
                max_len = 5000
            )

        if self.encoder is None or default_all:
            self.encoder = nn.ModuleList([CATCallerEncoderLayer(
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

        if self.decoder is None or default_all:
            if self.model_type == 'ctc':
                # they have 6 as vocab size, unclear why
                self.decoder = nn.Sequential(nn.Linear(d_model, 6), nn.LogSoftmax(-1))
            elif self.model_type == 'crf':
                self.decoder = BonitoLinearCRFDecoder(
                    insize = d_model, 
                    n_base = CRF_N_BASE, 
                    state_len = CRF_STATE_LEN, 
                    bias=CRF_BIAS, 
                    scale= CRF_SCALE, 
                    blank_score= CRF_BLANK_SCORE,
                )
