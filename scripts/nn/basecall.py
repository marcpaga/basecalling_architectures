import argparse
import os
import sys
import pandas as pd
import torch
from pathlib import Path

def get_best_checkpoint(log_df, metric, mode):
    """Finds the best checkpoint
    """
    log_df = log_df[log_df['checkpoint'] == 'yes']
    if mode == 'max':
        step = list(log_df['step'])[np.argmax(log_df[metric])]
    elif mode == 'min':
        step = list(log_df['step'])[np.argmin(log_df[metric])]
    else:
        raise ValueError('mode must be either "max" or "min", input was: ' + str(mode))
    
    return 'checkpoint_' + str(step) + '.pt'
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast5-dir", type=str, help='Dir with fast5 files, files are searched recursively', default = None)
    parser.add_argument("--file-list", type=str, help='File with a list of fast5 files to basecall', default = None)
    parser.add_argument("--model-dir", type=str, help='Directory where the model was saved', required = True)
    parser.add_argument("--output-dir", type=str, help='Path to where to write the basecalls', required = True)
    parser.add_argument("--reads-file", type=int, help='How many reads will be written per file, if 0 then a single file is used', default = 0)
    parser.add_argument("--metric", type=str, help='Metric name to be used to find the best checkpoint, if not given', default = None)
    parser.add_argument("--mode", type=str, help="""For the metric, how it should be used 'max' or 'min'; 
                                                    for example, for accuracy one wants to use 'max' and 
                                                    for loss one wants to use 'min'""", default = None)
    parser.add_argument("--chunk-size", type=int, help='', default = 2000)
    parser.add_argument("--overlap", type=int, help='Overlap between chunks of a read', default = 500)
    parser.add_argument("--batch-size", type=int, help='Batch size for the neural network', default = 64)
    parser.add_argument("--greedy", action='store_true', help='Whether to use a greedy approach for decoding')
    parser.add_argument("--gpu", action='store_true', help='A GPU will be used for basecalling')
    parser.add_argument("--silent", action='store_true', help='Do not output verbosity')
    args = parser.parse_args()
    
    # check argument compatibility
    if args.fast5_dir and args.file_list:
        raise ValueError('Only --data-dir or --file-list must be given as inputs')
    if not args.fast5_dir and not args.file_list:
        raise ValueError('Either --data-dir or --file-list must be provided as input')
    
    if args.overlap >= args.chunk_size:
        raise ValueError('The overlap between chunks ({0}) cannot be larger or equal to the chunk size ({1})'.format(args.overlap, args.chunk_size))
    
    
    # load the model
    sys.path.append(args.model_dir)
    from config import model
    from basecaller import Basecaller
    
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    
    model = model.to(device)
    model.device = device
    
    # get the best checkpoint according to metric
    if args.metric:
        log_df = pd.read_csv(os.path.join(args.model_dir, 'model.log'))
        best_checkpoint_file = get_best_checkpoint(log_df, args.metric, args.mode)
    else: # or the last one saved
        max_val = 0
        best_checkpoint_file = ''
        for f in  os.listdir(os.path.join(args.model_dir, 'checkpoints')):
            val = int(f.split('_')[-1].split('.')[0])
            if val > max_val:
                max_val = val
                best_checkpoint_file = f
                
    # load the weights into the model
    checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoints', best_checkpoint_file), map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state'])
    
    # get the list of files to basecall
    files_list = list()
    if args.fast5_dir is not None:
        for path in Path(args.fast5_dir).rglob('*.fast5'):
            files_list.append(str(path))
    elif fast5_files_list is not None:
        with open(fast5_files_list, 'r') as f:
            for line in f:
                files_list.append(line.strip('\n'))

    basecaller = Basecaller(model = model, 
                            chunk_size = args.chunk_size, 
                            overlap = args.overlap, 
                            batch_size = args.batch_size)
    
    basecaller.basecall(files_list, 
                        output_dir = args.output_dir, 
                        reads_per_file = args.reads_file, 
                        greedy = args.greedy, 
                        silent = args.silent)