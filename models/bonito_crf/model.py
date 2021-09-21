"""Implementation of the BonitoCRF model
"""

import sys
import os
sys.path.append('/hpc/compgen/users/mpages/babe/src')
from classes import BaseModel
from utils import alignment_accuracy
from constants import CTC_BLANK
import layers

import torch
from torch import nn

class BonitoCRFModel(BaseModel):
    """Bonito CRF model
    """
    def __init__(self, convolution = None, rnn = None, decoder = None, seqdist = None, *args, **kwargs):
        super(BonitoCRFModel, self).__init__(*args, **kwargs)
        """
        """        
        
        self.convolution = convolution
        self.rnn = rnn
        self.decoder = decoder
        self.seqdist = seqdist
        
        if self.convolution is None or self.self.rnn is None or self.decoder is None:
            self.load_default_configuration()
            
        self.criterion = self.seqdist.ctc_loss
        
    def train_step(self, batch):
        """Train a step with a batch of data
        
        Args:
            batch (dict): dict with keys 'x' (batch, len) 
                                         'y' (batch, len)
        """
        
        self.train()
        x = batch['x'].to(self.device)
        x = x.unsqueeze(1) # add channels dimension
        p = self.forward(x) # forward through the network
        y = batch['y'].to(self.device)
        
        loss, losses = self.calculate_loss(y, p)
        self.optimize(loss)
        
        return losses, p
        
    
    def validation_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            x = x.unsqueeze(1) # add channels dimension
            p = self.forward(x) # forward through the network
            y = batch['y'].to(self.device)
            
            loss, losses = self.calculate_loss(y, p)
            
        return losses, p
    
    def predict_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            x = x.unsqueeze(1)
            p = self.forward(x)
        return p
    
    
    def decode(self, p, greedy = True):
        """Decode the predictions
        
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        if greedy:
            return self.decode_greedy(p)
        else:
            raise NotImplementedError('Beam search not yet implemented')
    
    def decode_greedy(self, y):
        """Predict the sequences using a greedy approach
        
        Args:
            y (tensor): tensor with scores in shape [timesteps, batch, classes]
        Returns:
            A (list) with the decoded strings
        """
        scores = self.seqdist.posteriors(y.to(torch.float32)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        return [self.seqdist.path_to_str(y) for y in tracebacks.cpu().numpy()]
        
    def evaluate(self, batch, predictions):
        """Evaluate the predictions by calculating the accuracy
        
        Args:
            batch (dict): dict with tensor with [batch, len] in key 'y'
            predictions (list): list of predicted sequences as strings
        """
        y = batch['y'].cpu().numpy()
        accs = list()
        for i, sample in enumerate(y):
            y_str = ''
            for s in sample:
                if s == CTC_BLANK:
                    break
                y_str += self.dataloader_train.dataset.decoding_dict[s]
                
            accs.append(alignment_accuracy(y_str, predictions[i]))
            
        return {'metric.accuracy': accs}
            
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
        
        y_len = torch.sum(y != CTC_BLANK, axis = 1).to(self.device)
        loss = self.criterion(scores = p, targets = y, target_lengths = y_len, 
                              loss_clip = 10, reduction='mean', normalise_scores=True)
        losses = {'loss.global': loss.item(), 'loss.crf': loss.item()}
        
        return loss, losses
        
        
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
            self.rnn = nn.Sequential(layers.BonitoLSTM(384, 384, reverse = True),
                                     layers.BonitoLSTM(384, 384, reverse = False),
                                     layers.BonitoLSTM(384, 384, reverse = True),
                                     layers.BonitoLSTM(384, 384, reverse = False),
                                     layers.BonitoLSTM(384, 384, reverse = True))
        if self.decoder is None or default_all:
            self.decoder = layers.BonitoLinearCRFDecoder(insize = 384, n_base = 4, state_len = 4, bias=True, scale= 5.0, blank_score= 2.0)
        if self.seqdist is None or default_all:
            self.seqdist = layers.CTC_CRF(state_len = 4, alphabet = 'NACGT')
