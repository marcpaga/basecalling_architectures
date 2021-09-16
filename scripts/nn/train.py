"""General train script

Imports the config file, from which it should have all the 
objects necessary for training.
Necessary objects:
- model
- num_epochs
- validate_every
- checkpoint_every
"""


import os
import sys
from importlib.machinery import SourceFileLoader
import argparse
import torch
import numpy as np
import pandas as pd
import time

def generate_log_df(losses_keys, metrics_keys):
    """Creates a data.frame to store the logging values
    """
    
    header = ['epoch', # epoch number
              'step',  # step number
              'time']  # time it took
    # add losses and metrics for train and validation
    for k in losses_keys:
        header.append(k + '_train')
        header.append(k + '_val')
    for k in metrics_keys:
        header.append(k + '_train')
        header.append(k + '_val')
    # whether a checkpoint was saved at this step
    header.append('checkpoint')
    
    log_dict = dict()
    for k in header:
        log_dict[k] = [None]
    return pd.DataFrame(log_dict)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, help='Path to config model file')
    parser.add_argument("--output-dir", type=str, help='Path where to save the results')
    args = parser.parse_args()
    
    config_file_dir = '/'.join(args.config_file.split('/')[:-1])
    sys.path.append(config_file_dir)
    from config import model, num_epochs, validate_every, checkpoint_every
    model = model.to(model.device)
    
    # check output dir
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        if len(os.listdir(args.output_dir)) > 0:
            raise FileExistsError('Output dir contains files')
    
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
                predictions_decoded = model.predict(predictions)

                metrics = model.evaluate(train_batch, predictions_decoded)
                
                # log the train results
                log_df = generate_log_df(list(losses.keys()), list(metrics.keys()))
                for k, v in train_results.items():
                    log_df[k + '_train'] = np.mean(v)
                for k, v in metrics.items():
                    log_df[k + '_train'] = np.mean(v)
                train_results = dict() # reset the dict
                
                try:
                    validation_batch = next(validation_iterator)
                except StopIteration:
                    validation_iterator = iter(validation_loader)
                    validation_batch = next(validation_iterator)
                                
                # calculate and log the validation results
                losses, predictions = model.validation_step(validation_batch)
                predictions_decoded = model.predict(predictions)
                metrics = model.evaluate(validation_batch, predictions_decoded)
                for k, v in losses.items():
                    log_df[k + '_val'] = v # do not need the mean as we only did it once
                for k, v in metrics.items():
                    log_df[k + '_val'] = np.mean(v)
                    
                # calculate time it took since last validation step
                log_df['epoch'] = str(epoch_num)
                log_df['step'] = str(total_num_steps)
                log_df['time'] = int(time.time() - st_time)
                st_time = time.time()
                    
                # save the model if we are at a saving step
                if total_num_steps % checkpoint_every == 0:
                    log_df['checkpoint'] = 'yes'
                    model.save(os.path.join(args.output_dir, 'checkpoint_' + str(total_num_steps) + '.pt'))
                else:
                    log_df['checkpoint'] = 'no'
                
                # write to log
                if not os.path.isfile(os.path.join(args.output_dir, 'model.log')):
                    log_df.to_csv(os.path.join(args.output_dir, 'model.log'), 
                                  header=True, index=False)
                else: # else it exists so append without writing the header
                    log_df.to_csv(os.path.join(args.output_dir, 'model.log'), 
                                  mode='a', header=False, index=False)
                    
                # write results to console
                print(log_df)
