"""Implementation of the Bonito-CTC model

Based on: 
https://github.com/nanoporetech/bonito
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseCTCModel
from layers import BonitoLSTM


class BonitoCTCModel(BaseCTCModel):
    """BonitoCTC Model
    """
    def __init__(self, convolution = None, rnn = None, decoder = None, *args, **kwargs):
        super(BonitoCTCModel, self).__init__(*args, **kwargs)
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
        x = self.rnn(x)
        x = self.decoder(x)
        return x
            
        
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network
        """
        
        if self.convolution is None or default_all:
            self.convolution = nn.Sequential(nn.Conv1d(1, 4, 
                                                       kernel_size = 5, stride= 1, padding=5//2, bias=True),
                                             nn.SiLU(),
                                             nn.Conv1d(4, 16, 
                                                       kernel_size = 5, stride= 1, padding=5//2, bias=True),
                                             nn.SiLU(),
                                             nn.Conv1d(16, 384, 
                                                       kernel_size = 19, stride= 5, padding=19//2, bias=True),
                                             nn.SiLU())
        if self.rnn is None or default_all:
            self.rnn = nn.Sequential(BonitoLSTM(384, 384, reverse = True),
                                     BonitoLSTM(384, 384, reverse = False),
                                     BonitoLSTM(384, 384, reverse = True),
                                     BonitoLSTM(384, 384, reverse = False),
                                     BonitoLSTM(384, 384, reverse = True))
        if self.decoder is None or default_all:
            self.decoder = nn.Sequential(nn.Linear(384, 5), nn.LogSoftmax(-1))