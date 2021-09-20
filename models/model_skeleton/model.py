"""Implementation of the Skeleton model
"""

import sys
import os
sys.path.append('/hpc/compgen/users/mpages/babe/src')
from classes import BaseModel
from utils import alignment_accuracy

import torch
from torch import nn

class Skeleton(BaseModel):
    """Skeleton Model
    """
    def __init__(self, *args, **kwargs):
        super(Skeleton, self).__init__(*args, **kwargs)
        """
        """        
        
    def train_step(self, batch):
        """Train a step with a batch of data
        """
        
        return losses, p
        
    
    def validation_step(self, batch):
        """Predicts a single batch of data for validation
        """
        
        self.eval()
        with torch.no_grad():
            pass
            
        return losses, p
    
    def predict_step(self, batch):
        """Predict a single batch of data
        """
        self.eval()
        with torch.no_grad():
            pass
            
        return losses
    
    
    def predict(self, p):
        """Predict approach for evaluation metrics during training and evaluation. 
        """
        return p
        
    def evaluate(self, batch, predictions):
        """Evaluate the predictions by calculating the accuracy
        """
            
        return #{'metric.accuracy': accs}
            
    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        """
        
        return loss, losses
        
        
    def forward(self, x):
        """Forward pass of a batch
        """
        return x
        
    def load_default_configuration(self, default_all = False):
        """Sets the default configuration for one or more
        modules of the network
        """
