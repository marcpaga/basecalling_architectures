

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
import argparse

from classes import BaseBasecaller, BaseFast5Dataset
from gridmodel import GridAnalysisModel # pyright: reportMissingImports=false

import pandas as pd
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--window-overlap", type=int)
    parser.add_argument("--file-list", type=str)
    parser.add_argument("--batch-size", type=int, default = 64)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--beam-threshold", type=float)
    parser.add_argument("--model-stride", type=int)
    parser.add_argument("--chunk-size", type=int)
    args = parser.parse_args()

    fast5_dataset = BaseFast5Dataset(fast5_list= args.file_list, buffer_size = 10)

    model_dir = '/hpc/compgen/projects/nanoxog/babe/analysis/mpages/models/grid_analysis'
    config = args.model_name.split('_')
    output_file = os.path.join(model_dir, args.task, args.model_name, 'basecalls_' + str(args.beam_size) + '_' + str(args.beam_threshold) + '.fastq')

    cnn_type = config[0]
    encoder_type = config[1]
    decoder_type = config[2]
    use_connector = config[3]
    window_size = config[4]

    # load model
    log = df = pd.read_csv(os.path.join(model_dir, args.task, args.model_name, 'train.log'))
    log = log[log['checkpoint'] == 'yes']
    best_step = log['step'].iloc[np.argmax(log['metric.accuracy.val'])]
    checkpoint_file = os.path.join(model_dir, args.task, args.model_name, 'checkpoints', 'checkpoint_' + str(best_step) + '.pt')

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model = GridAnalysisModel(
        cnn_type = cnn_type, 
        encoder_type = encoder_type, 
        decoder_type = decoder_type,
        use_connector = use_connector,
        device = device,
        dataloader_train = None, 
        dataloader_validation = None, 
        scaler = scaler,
        use_amp = use_amp
    )

    state_dict = torch.load(checkpoint_file)
    model.load_state_dict(state_dict['model_state'])
    model = model.to(device)

    basecaller = BaseBasecaller(
        dataset = fast5_dataset, 
        model = model, 
        batch_size = args.batch_size, 
        output_file = output_file, 
        n_cores = 4, 
        chunksize = args.chunk_size, 
        overlap = args.window_overlap, 
        stride = args.model_stride,
        beam_size = args.beam_size,
        beam_threshold = args.beam_threshold,
    )

    basecaller.basecall()

