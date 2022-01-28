

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
import argparse

from classes import BasecallerImpl, BaseFast5Dataset

import pandas as pd
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--model-dir", type=str, help='where all the models are saved')
    parser.add_argument("--output-file", type=str, help='output fastq file', default = None)
    parser.add_argument("--task", type=str)
    parser.add_argument("--window-overlap", type=int)
    parser.add_argument("--file-list", type=str)
    parser.add_argument("--batch-size", type=int, default = 64)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--beam-threshold", type=float)
    parser.add_argument("--model-stride", type=int, default = None)
    parser.add_argument("--chunk-size", type=int)
    args = parser.parse_args()

    fast5_dataset = BaseFast5Dataset(fast5_list= args.file_list, buffer_size = 1)

    model_folder_name = args.model+'_'+str(args.chunk_size)

    if args.output_file is None:
        output_file = os.path.join(args.model_dir, args.task, model_folder_name, 'basecalls_' + str(args.beam_size) + '_' + str(args.beam_threshold) + '.fastq')
    else:
        output_file = args.output_file

    # load model
    log = pd.read_csv(os.path.join(args.model_dir, args.task, model_folder_name, 'train.log'))
    log = log[log['checkpoint'] == 'yes']
    best_step = log['step'].iloc[np.argmax(log['metric.accuracy.val'])]
    checkpoint_file = os.path.join(args.model_dir, args.task, model_folder_name, 'checkpoints', 'checkpoint_' + str(best_step) + '.pt')

    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.model == 'bonito':
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

    state_dict = torch.load(checkpoint_file)
    model.load_state_dict(state_dict['model_state'])
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

