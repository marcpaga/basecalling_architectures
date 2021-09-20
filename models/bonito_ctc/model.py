"""Implementation of the Bonito-CTC model

Most of the code has been copied with or without modifications from:
https://github.com/nanoporetech/bonito

"""

import sys
import os
sys.path.append('/hpc/compgen/users/mpages/babe/src')
from classes import BaseModel
from layers import BonitoLSTM
from constants import CTC_BLANK
from utils import decode_batch_greedy_ctc, alignment_accuracy

import torch
from torch import nn

class BonitoCTCModel(BaseModel):
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
        
        self.criterion = nn.CTCLoss(zero_infinity = True).to(self.device)
        
        if self.convolution is None or self.self.rnn is None or self.decoder is None:
            self.load_default_configuration()
        
        
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
    
    def predict(self, p):
        """Predict approach for evaluation metrics during training and evaluation. 
        """
        return self.predict_greedy(p)
        
    
    def predict_greedy(self, p):
        """Predict the bases in a greedy approach
        Args:
            p (tensor): with classes as last dimension
        """
        p = p.detach()
        p = p.argmax(-1).permute(1, 0)
        p = p.cpu().numpy()
        decoded_predictions =  decode_batch_greedy_ctc(y = p, 
                                                       decode_dict = self.dataloader_train.dataset.decoding_dict, 
                                                       blank_label = CTC_BLANK)
        return decoded_predictions
    
    def predict_beam_search(self):
        """Predict the bases using beam search
        """
        raise NotImplementedError()
        
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
        p_len = torch.full((p.shape[1], ), p.shape[0]).to(self.device)
        
        loss = self.criterion(p, y, p_len, y_len)
        losses = {'loss.global': loss.item(), 'loss.ctc': loss.item()}
        
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
            self.rnn = nn.Sequential(BonitoLSTM(384, 384, reverse = True),
                                     BonitoLSTM(384, 384, reverse = False),
                                     BonitoLSTM(384, 384, reverse = True),
                                     BonitoLSTM(384, 384, reverse = False),
                                     BonitoLSTM(384, 384, reverse = True))
        if self.decoder is None or default_all:
            self.decoder = nn.Sequential(nn.Linear(384, 5), nn.LogSoftmax(-1))