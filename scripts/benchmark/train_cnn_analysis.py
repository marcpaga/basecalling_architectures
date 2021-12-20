"""Script used for training for the CNN analysis experiment
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelImpl, BaseNanoporeDataset

from layers.causalcall import CausalCallConvBlock
from layers.halcyon import HalcyonCNNBlock, HalcyonInceptionBlock
from layers.mincall import MinCallConvBlock
from layers.urnano import URNetDownBlock, URNetFlatBlock, URNetUpBlock, URNet

import torch
from torch import nn
from torch.utils.data import DataLoader

from shutil import copyfile
import argparse
import numpy as np
import pandas as pd
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_log_df(losses_keys, metrics_keys):
    """Creates a data.frame to store the logging values
    """
    
    header = ['epoch', # epoch number
              'step',  # step number
              'time']  # time it took
    # add losses and metrics for train and validation
    for k in losses_keys:
        header.append(k + '.train')
        header.append(k + '.val')
    for k in metrics_keys:
        header.append(k + '.train')
        header.append(k + '.val')
    # whether a checkpoint was saved at this step
    header.append('lr')
    header.append('checkpoint')
    
    log_dict = dict()
    for k in header:
        log_dict[k] = [None]
    return pd.DataFrame(log_dict)

class CnnAnalysisModel(BaseModelImpl):

    def __init__(self, cnn_type, rnn_size = 0, rnn_type = 'lstm', num_layers = 2, bidirectional = True, *args, **kwargs):
        super(CnnAnalysisModel, self).__init__(*args, **kwargs)

        self.cnn_type = cnn_type
        self.cnn_out_size = None
        self.convolution = self._build_cnn()
        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn = self._build_rnn()
        self.decoder = self._build_decoder()

    def forward(self, x):
        """Forward pass of a batch
        
        Args:
            x (tensor) : [batch, channels (1), len]
        """
        
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        if self.rnn_size > 0:
            x = self.rnn(x)
            if self.rnn_type == 'lstm':
                x = x[0]
        x = self.decoder(x)
        return x
        
    def _build_cnn(self):
        
        if self.cnn_type == 'bonito':
            self.cnn_out_size = 384

            cnn = nn.Sequential(
                nn.Conv1d(
                        in_channels = 1, 
                        out_channels = 4, 
                        kernel_size = 5, 
                        stride= 1, 
                        padding=5//2, 
                        bias=True
                ),
                nn.SiLU(),
                nn.Conv1d(
                        in_channels = 4, 
                        out_channels = 16, 
                        kernel_size = 5, 
                        stride= 1, 
                        padding=5//2, 
                        bias=True
                ),
                nn.SiLU(),
                nn.Conv1d(
                        in_channels = 16, 
                        out_channels = 384, 
                        kernel_size = 19, 
                        stride= 5, 
                        padding=19//2, 
                        bias=True
                ),
                nn.SiLU()
            )

        elif self.cnn_type == 'catcaller':
            self.cnn_out_size = 512

            d_model = 512
            padding = 1
            kernel = 3
            stride = 2
            dilation = 1

            cnn = nn.Sequential(
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

        elif self.cnn_type == 'causalcall':
            self.cnn_out_size = 128

            num_blocks = 5
            num_channels = 256
            kernel_size = 3
            dilation_multiplier = 2
            dilation = 1

            layers = list()
            for i in range(num_blocks):
                if i == 0:
                    layers.append(CausalCallConvBlock(kernel_size, num_channels, 1, dilation))
                else:
                    layers.append(CausalCallConvBlock(kernel_size, num_channels, int(num_channels/2), dilation))
                dilation *= dilation_multiplier

            cnn = nn.Sequential(*layers)

        elif self.cnn_type == 'halcyon':
            self.cnn_out_size = 243

            strides = [1] * 5
            num_kernels = [1, 3, 5, 7, 3]
            paddings = ['same'] * 5
            num_channels = [64, 96, 128, 160, 32]
            use_bn = [True] * 5

            cnn = nn.Sequential(
                HalcyonCNNBlock( 1, 64, 3, 2, "valid", True),
                HalcyonCNNBlock(64, 64, 3, 1, "valid", True),
                HalcyonCNNBlock(64, 128, 3, 1, "valid", True),
                nn.MaxPool1d(3, 2),
                HalcyonCNNBlock(128, 160, 3, 1, "valid", True),
                HalcyonCNNBlock(160, 384, 3, 1, "valid", True),
                nn.MaxPool1d(3, 2),
                HalcyonInceptionBlock(384, num_channels, num_kernels, strides, paddings, use_bn, scaler = 0.8**1),
                HalcyonInceptionBlock(382, num_channels, num_kernels, strides, paddings, use_bn, scaler = 0.8**2),
                HalcyonInceptionBlock(304, num_channels, num_kernels, strides, paddings, use_bn, scaler = 0.8**3),
            )

        elif self.cnn_type == 'mincall':
            self.cnn_out_size = 64

            num_layers = 72
            pool_every = 24
            kernel_size = 3
            padding = 'same'
            num_channels = 64
            max_pool_kernel = 2

            layers = list()
            layers.append(nn.Conv1d(1, num_channels, kernel_size, 1, padding)) 
            for i in range(num_layers):
                if i % pool_every == 0 and i > 0:
                    layers.append(nn.MaxPool1d(max_pool_kernel))
                layers.append(MinCallConvBlock(kernel_size, num_channels, num_channels, padding))

            cnn = nn.Sequential(*layers)

        elif self.cnn_type == 'sacall':
            self.cnn_out_size = 256

            d_model = 256
            kernel = 3
            maxpooling_stride = 2 

            cnn = nn.Sequential(
                nn.Conv1d(1, d_model//2, kernel, 1, 1, bias=False),
                nn.BatchNorm1d(d_model//2),
                nn.ReLU(),
                nn.MaxPool1d(kernel, maxpooling_stride, 1),
                nn.Conv1d(d_model//2, d_model, kernel, 1, 1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.MaxPool1d(kernel, maxpooling_stride, 1),
                nn.Conv1d(d_model, d_model, kernel, 1, 1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.MaxPool1d(kernel, maxpooling_stride, 1)
            )

        elif self.cnn_type == 'urnano_cnn':
            self.cnn_out_size = 64
            raise NotImplementedError()

        elif self.cnn_type == 'urnano':
            self.cnn_out_size = 64

            padding = 'same'
            stride = 1
            n_channels = [64, 128, 256, 512]
            kernel = 11
            maxpooling = [2, 2, 2] # in the github json it is [3, 2, 2], changed because we use even number segments

            down = nn.ModuleList([URNetDownBlock(1, n_channels[0], kernel, maxpooling[0], stride, padding),
                                    URNetDownBlock(n_channels[0], n_channels[1], 3, maxpooling[1], stride, padding),
                                    URNetDownBlock(n_channels[1], n_channels[2], 3, maxpooling[2], stride, padding)])
            flat = nn.ModuleList([URNetFlatBlock(n_channels[2], n_channels[3], 3, stride, padding)])
            up = nn.ModuleList([URNetUpBlock(n_channels[3], n_channels[2], 3, maxpooling[2], maxpooling[2], stride, padding), 
                                URNetUpBlock(n_channels[2], n_channels[1], 3, maxpooling[1], maxpooling[1], stride, padding),
                                URNetUpBlock(n_channels[1], n_channels[0], 3, maxpooling[0], maxpooling[0], stride, padding)])

            cnn = nn.Sequential(URNet(down, flat, up), 
                                nn.Conv1d(n_channels[0], n_channels[0], 3, stride, padding), 
                                nn.BatchNorm1d(n_channels[0]), 
                                nn.ReLU())

        else:
            raise ValueError('cnn_type not recognized: ' + str(self.cnn_type))
        return cnn

    def _build_rnn(self):
        
        if self.rnn_size == 0:
            return None

        if self.rnn_type == 'lstm':
            rnn =  nn.LSTM(
                input_size = self.cnn_out_size, 
                hidden_size = self.rnn_size, 
                num_layers = self.num_layers, 
                bidirectional = self.bidirectional
            )
        
        if self.rnn_type == 'gru':
            rnn =  nn.GRU(
                input_size = self.cnn_out_size, 
                hidden_size = self.rnn_size, 
                num_layers = self.num_layers, 
                bidirectional = self.bidirectional
            )
            
        return rnn

    def _build_decoder(self):

        if self.rnn_size > 0:
            if self.bidirectional:
                rnn_out = int(self.rnn_size * 2)
            else:
                rnn_out = self.rnn_size
        else:
            rnn_out = self.cnn_out_size

        return nn.Sequential(nn.Linear(rnn_out, 5), nn.LogSoftmax(-1))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-type", type=str, choices=[
        'bonito',
        'catcaller',
        'causalcall',
        'halcyon',
        'mincall',
        'sacall',
        'urnano_rnn'
    ], help='Type of CNN')
    parser.add_argument("--rnn-size", type=int, help='Hidden size of RNN, set to 0 for no RNN', default = 0)
    parser.add_argument("--rnn-type", type=str, choices=['lstm', 'gru'], help='Type of RNN', default = 'lstm')
    parser.add_argument("--num-layers", type=int, help='NUmber of stacked RNN layers', default = 2)
    parser.add_argument("--bidirectional", action='store_true', help="Whether to have a bidirectional RNN")
    parser.add_argument("--window-size", type=int, choices=[400, 2000, 4000], help='Window size for the data')
    parser.add_argument("--task", type=str, choices=['human', 'global', 'inter'])
    args = parser.parse_args()
    
    num_epochs = 3
    validate_every = 100
    checkpoint_every = 5000

    data_dir = '/hpc/compgen/projects/nanoxog/babe/analysis/mpages/train_input/' + args.task + '/' + str(args.window_size) + '.0'
    encoding_dict = {'A': 1 , 'C':  2 , 'G':  3 , 'T':  4 , '':  0}
    decoding_dict = { 1 :'A',  2 : 'C',  3 : 'G',  4 : 'T', 0 : ''}
    dataset = BaseNanoporeDataset(data_dir = data_dir, 
                                decoding_dict = decoding_dict, 
                                encoding_dict = encoding_dict, 
                                split = 0.95, 
                                shuffle = True, 
                                seed = 1)

    dataloader_train = DataLoader(dataset, batch_size = 64, 
                                sampler = dataset.train_sampler, num_workers = 1)
    dataloader_validation = DataLoader(dataset, batch_size = 64, 
                                    sampler = dataset.validation_sampler, num_workers = 1)
    
    model = CnnAnalysisModel(
        cnn_type = args.cnn_type, 
        rnn_size = args.rnn_size, 
        rnn_type = args.rnn_type, 
        num_layers = args.num_layers, 
        bidirectional = args.bidirectional, 
        device = device,
        dataloader_train = dataloader_train, 
        dataloader_validation = dataloader_validation, 
        model_type = 'ctc'
    )
    model = model.to(device)

    ##    OPTIMIZATION     #############################################
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    schedulers = {'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (len(dataset.train_idxs)*num_epochs)/64, 
                                                                            eta_min=0.00001, last_epoch=-1, verbose=False),
                }
    clipping_value = 2
    use_sam = False


    ##   MODEL PART2        #############################################
    model.optimizer = optimizer
    model.schedulers = schedulers
    model.clipping_value = clipping_value
    model.use_sam = False

    # output stuff
    output_dir = os.path.join('/hpc/compgen/projects/nanoxog/babe/analysis/mpages/models/cnn_analysis', args.task)
    output_dir += '/'
    config = [args.cnn_type, args.rnn_size, args.rnn_type, args.num_layers, args.bidirectional, args.window_size]
    for i, c in enumerate(config):
        if i > 0:
            output_dir += '_'
        output_dir += str(c)

    checkpoints_dir = os.path.join(output_dir, 'checkpoints')

    # check output dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        os.mkdir(checkpoints_dir)
    else:
        if len(os.listdir(output_dir)) > 0:
            raise FileExistsError('Output dir contains files')
        else:
            os.mkdir(checkpoints_dir)
    
    # keep track of losses and metrics to take the average
    train_results = dict()
    
    total_num_steps = 1
    for epoch_num in range(num_epochs):
        
        loader_train = model.dataloader_train
        loader_validation = model.dataloader_validation
        # use this to restart the in case we finish all the validation data
        validation_iterator = iter(loader_validation) 
        
        st_time = time.time()
        # iterate over the train data
        for train_batch_num, train_batch in enumerate(loader_train):
            
            losses, predictions = model.train_step(train_batch)
            total_num_steps += 1
            
            for k, v in losses.items():
                if k not in train_results.keys():
                    train_results[k] = list()
                train_results[k].append(v)
            
            if total_num_steps % validate_every == 0:
                
                # calculate accuracy for the training only here since doing for every batch
                # is expensive and slow...
                predictions_decoded = model.decode(predictions, greedy = True)
                metrics = model.evaluate(train_batch, predictions_decoded)
                
                # log the train results
                log_df = generate_log_df(list(losses.keys()), list(metrics.keys()))
                for k, v in train_results.items():
                    log_df[k + '.train'] = np.mean(v)
                for k, v in metrics.items():
                    log_df[k + '.train'] = np.mean(v)
                train_results = dict() # reset the dict
                
                try:
                    validation_batch = next(validation_iterator)
                except StopIteration:
                    validation_iterator = iter(loader_validation)
                    validation_batch = next(validation_iterator)
                                
                # calculate and log the validation results
                losses, predictions = model.validation_step(validation_batch)
                predictions_decoded = model.decode(predictions, greedy = True)
                metrics = model.evaluate(validation_batch, predictions_decoded)
                
                for k, v in losses.items():
                    log_df[k + '.val'] = v # do not need the mean as we only did it once
                for k, v in metrics.items():
                    log_df[k + '.val'] = np.mean(v)
                    
                # calculate time it took since last validation step
                log_df['epoch'] = str(epoch_num)
                log_df['step'] = str(total_num_steps)
                log_df['time'] = int(time.time() - st_time)
                for param_group in model.optimizer.param_groups:
                    log_df['lr'] = param_group['lr']
                st_time = time.time()
                    
                # save the model if we are at a saving step
                if total_num_steps % checkpoint_every == 0:
                    log_df['checkpoint'] = 'yes'
                    model.save(os.path.join(checkpoints_dir, 'checkpoint_' + str(total_num_steps) + '.pt'))
                else:
                    log_df['checkpoint'] = 'no'
                
                # write to log
                if not os.path.isfile(os.path.join(output_dir, 'train.log')):
                    log_df.to_csv(os.path.join(output_dir, 'train.log'), 
                                  header=True, index=False)
                else: # else it exists so append without writing the header
                    log_df.to_csv(os.path.join(output_dir, 'train.log'), 
                                  mode='a', header=False, index=False)
                    
                # write results to console
                print(log_df)
                
    
    model.save(os.path.join(checkpoints_dir, 'checkpoint_' + str(total_num_steps) + '.pt'))