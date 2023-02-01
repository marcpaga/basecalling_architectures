

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
import argparse

from classes import BasecallerImpl, BaseFast5Dataset

import pandas as pd
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=[
        'bonito',
        'catcaller',
        'causalcall',
        'mincall',
        'sacall',
        'urnano',
        'halcyon',
    ], required = True)
    parser.add_argument("--fast5-dir", type=str, required = True)
    parser.add_argument("--checkpoint", type=str, help='checkpoint file to load model weights', required = True)
    parser.add_argument("--output-file", type=str, help='output fastq file', required = True)
    parser.add_argument("--chunk-size", type=int, default = 2000)
    parser.add_argument("--window-overlap", type=int, default = 200)
    parser.add_argument("--batch-size", type=int, default = 64)
    parser.add_argument("--beam-size", type=int, default = 1)
    parser.add_argument("--beam-threshold", type=float, default = 0.1)
    parser.add_argument("--model-stride", type=int, default = None)
    
    args = parser.parse_args()


    file_list = list()
    for f in os.listdir(args.fast5_dir):
        if f.endswith('.fast5'):
            file_list.append(os.path.join(args.fast5_dir, f))

    fast5_dataset = BaseFast5Dataset(fast5_list= file_list, buffer_size = 1)

    output_file = args.output_file

    # load model
    checkpoint_file = args.checkpoint
    
    use_amp = False
    scaler = None

    if args.model == 'halcyon':
        from halcyon.model import HalcyonModelS2S as Model# pyright: reportMissingImports=false
        args.model_stride = 1
    elif args.model == 'bonito':
        from bonito.model import BonitoModel as Model# pyright: reportMissingImports=false
    elif args.model == 'catcaller':
        from catcaller.model import CATCallerModel as Model# pyright: reportMissingImports=false
    elif args.model == 'causalcall':
        from causalcall.model import CausalCallModel as Model # pyright: reportMissingImports=false
    elif args.model == 'mincall':
        from mincall.model import MinCallModel as Model # pyright: reportMissingImports=false
    elif args.model == 'sacall':
        from sacall.model import SACallModel as Model # pyright: reportMissingImports=false
    elif args.model == 'urnano':
        from urnano.model import URNanoModel as Model # pyright: reportMissingImports=false

    model = Model(
        load_default = True,
        device = device,
        dataloader_train = None, 
        dataloader_validation = None, 
        scaler = scaler,
        use_amp = use_amp,
    )
    model = model.to(device)

    model = model.to(device)
    model.load(checkpoint_file, initialize_lazy = True)
    model = model.to(device)

    basecaller = BasecallerImpl(
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

    basecaller.basecall(verbose = True)

