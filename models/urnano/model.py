"""Implementation of the SkeletonModel

Based on: 
https://github.com/nanoporetech/bonito
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelImpl
from layers.urnano import URNetDownBlock, URNetFlatBlock, URNetUpBlock, URNet
from layers.bonito import BonitoLinearCRFDecoder
from constants import CRF_STATE_LEN, CRF_BIAS, CRF_SCALE, CRF_BLANK_SCORE , CRF_N_BASE 


class URNanoModel(BaseModelImpl):
    """URNano model

    Based on: https://github.com/yaozhong/URnano/blob/master/models/model_unet.py
    """
    def __init__(self, convolution = None, rnn = None, decoder = None, *args, **kwargs):
        super(URNanoModel, self).__init__(*args, **kwargs)
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            rnn (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
        """
    
        self.convolution = convolution
        self.rnn = rnn
        self.decoder = decoder
        
        if self.convolution is None or self.self.rnn is None or self.decoder is None:
            self.load_default_configuration()

    def forward(self, x):
        """Forward pass of a batch
        
        Args:
            x (tensor) : [batch, channels (1), len]
        """
        
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x, _ = self.rnn(x)
        x = self.decoder(x)
        
        return x
            
        
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network

        Based on: https://github.com/yaozhong/URnano/blob/master/experiment/model/Unet.model.parameters.json
        """
        
        padding = 'same'
        stride = 1
        n_channels = [64, 128, 256, 512]
        kernel = 11
        maxpooling = [2, 2, 2] # in the github json it is [3, 2, 2], changed because we use even number segments


        if self.convolution is None or default_all:
            down = nn.ModuleList([URNetDownBlock(1, n_channels[0], kernel, maxpooling[0], stride, padding),
                                 URNetDownBlock(n_channels[0], n_channels[1], 3, maxpooling[1], stride, padding),
                                 URNetDownBlock(n_channels[1], n_channels[2], 3, maxpooling[2], stride, padding)])
            flat = nn.ModuleList([URNetFlatBlock(n_channels[2], n_channels[3], 3, stride, padding)])
            up = nn.ModuleList([URNetUpBlock(n_channels[3], n_channels[2], 3, maxpooling[2], maxpooling[2], stride, padding), 
                               URNetUpBlock(n_channels[2], n_channels[1], 3, maxpooling[1], maxpooling[1], stride, padding),
                               URNetUpBlock(n_channels[1], n_channels[0], 3, maxpooling[0], maxpooling[0], stride, padding)])
            
            self.convolution = nn.ModuleList([URNet(down, flat, up), 
                                              nn.Conv1d(n_channels[0], n_channels[0], 3, stride, padding), 
                                              nn.BatchNorm1d(n_channels[0]), 
                                              nn.ReLU()])

        if self.rnn is None or default_all:
            self.rnn = nn.GRU(n_channels[0], hidden_size = n_channels[0], num_layers = 3, bidirectional = True)
        if self.decoder is None or default_all:
            if self.model_type == 'ctc':
                self.decoder = nn.Sequential(nn.Linear(int(n_channels[0]*2), 5), nn.LogSoftmax(-1))
            elif self.model_type == 'crf':
                self.decoder = BonitoLinearCRFDecoder(
                    insize = n_channels[0], 
                    n_base = CRF_N_BASE, 
                    state_len = CRF_STATE_LEN, 
                    bias=CRF_BIAS, 
                    scale= CRF_SCALE, 
                    blank_score= CRF_BLANK_SCORE
                )
