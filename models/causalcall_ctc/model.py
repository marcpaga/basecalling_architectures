"""Implementation of the Skeleton model
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModel
from evaluation import alignment_accuracy
from torch import nn
from layers import CausalCallConvBlock

class CausalCallCTC(BaseModel):
    """CasualCallCTC Model
    """
    def __init__(self, convolution = None, decoder = None, *args, **kwargs):
        super(CausalCallCTC, self).__init__(*args, **kwargs)
        """
        Args:
           convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
           decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
        """        

        self.convolution = convolution
        self.decoder = decoder
        self.criterion = nn.CTCLoss(zero_infinity = True).to(self.device)

        self.load_default_configuration()  

    def forward(self, x):
        """Forward pass of a batch
        """
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.decoder(x)
        return x

        
    def evaluate(self, batch, predictions):
        """Evaluate the predictions by calculating the accuracy
        
        Args:
            batch (dict): dict with tensor with [batch, len] in key 'y'
            predictions (list): list of predicted sequences as strings
        """
        y = batch['y'].cpu().numpy()
        y_list = self.dataloader_train.dataset.encoded_array_to_list_strings(y)
        accs = list()
        for i, sample in enumerate(y_list):
            accs.append(alignment_accuracy(sample, predictions[i]))
            
        return {'metric.accuracy': accs}

    def decode(self, p, greedy = True):
        """Decode the predictions
         
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        if greedy:
            return self.decode_ctc_greedy(p)
        else:
            return self.decode_ctc_beam_search(p)
            
    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        loss = self.calculate_ctc_loss(y, p)
        losses = {'loss.global': loss.item(), 'loss.ctc': loss.item()}

        return loss, losses
            
    
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network
        """
    
        num_blocks = 5
        num_channels = 256
        kernel_size = 3
        dilation_multiplier = 2
        dilation = 1

        if self.convolution is None or default_all:
            layers = list()
            for i in range(num_blocks):
                if i == 0:
                    layers.append(CausalCallConvBlock(kernel_size, num_channels, 1, dilation))
                else:
                    layers.append(CausalCallConvBlock(kernel_size, num_channels, int(num_channels/2), dilation))
                dilation *= dilation_multiplier
            self.convolution = nn.Sequential(*layers)

        if self.decoder is None or default_all:
            self.decoder = nn.Sequential(nn.Linear(int(num_channels/2), num_channels), 
                                         nn.ReLU(),
                                         nn.Linear(num_channels, 5),
                                         nn.LogSoftmax(-1))