"""Implementation of the MinCall model

Based on:
https://arxiv.org/abs/1904.10337
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModel
from evaluation import alignment_accuracy
from layers import MinCallConvBlock

class MinCallCTC(BaseModel):
    """Skeleton Model
    """
    def __init__(self, convolution = None, decoder = None, *args, **kwargs):
        super(MinCallCTC, self).__init__(*args, **kwargs)
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

        num_layers = 72
        pool_every = 24
        kernel_size = 3
        padding = 'same'
        num_channels = 64
        max_pool_kernel = 2

        if self.convolution is None or default_all:
            layers = list()
            # add initial convolution to model to put to 64 channels
            # otherwise the residual connections in the block fails
            # since it tries to add tensors with 1 channel and 64 channels
            layers.append(nn.Conv1d(1, num_channels, kernel_size, 1, padding)) 
            for i in range(num_layers):
                if i % pool_every == 0 and i > 0:
                    layers.append(nn.MaxPool1d(max_pool_kernel))
                layers.append(MinCallConvBlock(kernel_size, num_channels, num_channels, padding))

            self.convolution = nn.Sequential(*layers)
        
        if self.decoder is None or default_all:
            self.decoder = nn.Sequential(nn.Linear(num_channels, 5), nn.LogSoftmax(-1))

