import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
import argparse

from classes import BasecallerImpl, BaseFast5Dataset
from gridmodel import GridAnalysisModel # pyright: reportMissingImports=false

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Basecaller')
    parser.add_argument("--model-name", type=str, help='Name of the model')
    parser.add_argument("--model-dir", type=str, help='where all the models are saved')
    parser.add_argument("--cnn-type", type=str, help='cnn type', 
        choices= [
            'bonito',
            'catcaller',
            'causalcall',
            'halcyon',
            'halcyonmod',
            'mincall',
            'sacall',
            'urnano',
        ]
    )
    parser.add_argument("--encoder-type", type=str, help='encoder type',
        choices = [
            'bonitofwd', 
            'bonitorev', 
            'catcaller', 
            'sacall', 
            'urnano', 
            'lstm1', 
            'lstm3', 
            'lstm5',
        ]
    )
    parser.add_argument("--decoder-type", type=str, help='decoder type', 
        choices = ['ctc', 'crf']
    )
    parser.add_argument("--use-connector", action='store_true', help='Use a connection layer between layers')
    parser.add_argument("--checkpoint-file", type=str, help='Checkpoint to load')
    parser.add_argument("--output-file", type=str, help='output fastq file')
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--window-overlap", type=int, default=200)
    parser.add_argument("--file-list", type=str)
    parser.add_argument("--batch-size", type=int, default = 64)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--beam-threshold", type=float, default=0.1)
    parser.add_argument("--model-stride", type=int, default = None)
    parser.add_argument("--buffer-size", type=int, default = 10)
    parser.add_argument("--num-cores", type=int, default=1, help='Number of multiprocessing cores to use')
    args = parser.parse_args()

    fast5_dataset = BaseFast5Dataset(
        fast5_list= args.file_list, 
        buffer_size = args.buffer_size,
        window_size = args.chunk_size,
        window_overlap = args.window_overlap,
    )
    
    model = GridAnalysisModel(
        cnn_type = args.cnn_type, 
        encoder_type = args.encoder_type, 
        decoder_type = args.decoder_type,
        use_connector = args.use_connector,
        device = device,
        dataloader_train = None, 
        dataloader_validation = None, 
        scaler = None,
        use_amp = False
    )

    model = model.to(device)
    model.load(args.checkpoint_file)

    basecaller = BasecallerImpl(
        dataset = fast5_dataset, 
        model = model, 
        batch_size = args.batch_size, 
        output_file = args.output_file, 
        n_cores = args.num_cores, 
        chunksize = args.chunk_size, 
        overlap = args.window_overlap, 
        stride = args.model_stride,
        beam_size = args.beam_size,
        beam_threshold = args.beam_threshold,
    )

    basecaller.basecall(verbose = True)

