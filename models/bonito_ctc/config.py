"""Configuration file of a Bonito-CRF model
"""
import sys
from model import BonitoCTCModel
sys.path.append('../../src')
from classes import BaseNanoporeDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## TRAIN CONFIGURATION #############################################
num_epochs = 3
train_batch_size = 128
validation_batch_size = 256
validate_every = 100
checkpoint_every = 500


##       DATA.         #############################################
data_dir = '/hpc/compgen/projects/nanoxog/babe/analysis/mpages/train_input/inter/2000.0'
encoding_dict = {'A': 1 , 'C':  2 , 'G':  3 , 'T':  4 , '':  0}
decoding_dict = { 1 :'A',  2 : 'C',  3 : 'G',  4 : 'T', 0 : ''}
dataset = BaseNanoporeDataset(data_dir = data_dir, 
                              decoding_dict = decoding_dict, 
                              encoding_dict = encoding_dict, 
                              split = 0.95, 
                              shuffle = True, 
                              seed = 1)

dataloader_train = DataLoader(dataset, batch_size = train_batch_size, 
                              sampler = dataset.train_sampler, num_workers = 1)
dataloader_validation = DataLoader(dataset, batch_size = validation_batch_size, 
                                   sampler = dataset.validation_sampler, num_workers = 1)


##   MODEL PART1        #############################################
model = BonitoCTCModel(device = device,
                       dataloader_train = dataloader_train, 
                       dataloader_validation = dataloader_validation)


##    OPTIMIZATION     #############################################
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
schedulers = dict()
clipping_value = 2
criterions = {'ctc': nn.CTCLoss(zero_infinity = True).to(device)}
use_sam = False


##   MODEL PART2        #############################################
model.optimizer = optimizer
model.schedulers = schedulers
model.clipping_value = clipping_value
model.criterions = criterions
model.use_sam = False