"""This script will evaluate the basecalling performance, it produces two files:
    - A per read table file
    - A global report file
"""

import argparse
import os
import sys
import multiprocessing as mp

import pandas as pd
import mappy

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from evaluation import eval_pair
from read import read_fast, read_fasta

def results_queue_writer(output_file, q):
    
    while True:
        df = q.get()
        if df == 'kill':
            break
        else:
            df = pd.DataFrame(df, index=[0])
        if not os.path.isfile(output_file):
            df.to_csv(output_file, header=True, index=False)
        else: # else it exists so append without writing the header
            df.to_csv(output_file, mode='a', header=False, index=False)
            

def eval_pair_wrapper(reads_queue, writer_queue, tmp_dir):
    """Wrapper evaluate a prediction in the queue
    Args:
        references (dict): dictionary with reference sequences
        read_queue (multiprocessing.Queue): queue from where to get the predictions
        writer_queue (multiprocessing.Queue): queue where to send the results
    """
    
    while not reads_queue.empty():

        data = reads_queue.get()
        read_id, reference, prediction = data

        if isinstance(prediction, tuple):
            pred, _, phredq = prediction
        else:
            pred = prediction
            phredq = None
            
        tmp_fasta = os.path.join(tmp_dir, read_id + '.fasta')
        with open(tmp_fasta, 'w') as f:
            f.write('>'+read_id+'\n')
            f.write(reference+'\n')

        ref = mappy.Aligner(tmp_fasta) 

        result = eval_pair(
            ref = ref, 
            que = pred, 
            read_id = read_id, 
            phredq = phredq, 
            align_method = 'minimap2'
        )
        
        writer_queue.put(result)

        os.remove(tmp_fasta)

    return None

def find_fast_files(top, maxdepth = 1):
    dirs, nondirs = [], []
    for name in os.listdir(top):
        (dirs if os.path.isdir(os.path.join(top, name)) else nondirs).append(name)
        for nondir in nondirs:
            if nondir.endswith('.fasta') or nondir.endswith('.fastq'):
                yield os.path.join(top, nondir)
    if maxdepth > 1:
        for name in dirs:
            for x in find_fast_files(os.path.join(top, name), maxdepth-1):
                yield x


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--basecalls-path", type=str, help='Path to a fasta or fastq file or dir to be searched')
    parser.add_argument("--references-path", type=str, help='Path to a fasta reference file')
    parser.add_argument("--output-file", type=str, help='csv output file', default = None)
    parser.add_argument("--depth", type=int, help='How deep to look for fastq or fasta files', default = 1)
    parser.add_argument("--processes", type=int, help='Number of parallel processes', default = 2)
    args = parser.parse_args()
    
    fast_files = list()
    if os.path.isfile(args.basecalls_path):
        fast_files.append(args.basecalls_path)
    else:
        for fast_file in find_fast_files(args.basecalls_path, args.depth):
            fast_files.append(fast_file)

    fast_files = set(fast_files)

    # read all the reference sequences into memory
    print('reading references: ' + args.references_path)
    references = read_fasta(args.references_path)

    for fast_file in fast_files:
        if args.output_file is None:
            output_file = ".".join(fast_file.split('.')[:-1]) + '_evaluation.csv'
        else:
            output_file = args.output_file
        tmp_dir = "/".join(fast_file.split('/')[:-1])
        finished_file = fast_file.split('.')[0] + '_done.txt'

        if os.path.isfile(finished_file):
            print('skipping: ' + str(fast_file))
            continue

        # read all the basecalls into memory
        print('reading basecalls: ' + fast_file)
        basecalls = read_fast(fast_file)

        manager = mp.Manager() 
        writer_queue = manager.Queue()
        reads_queue = manager.Queue()

        processed_ids = set()
        if os.path.isfile(output_file):
            df = pd.read_csv(output_file, header = 0, index_col = False)
            processed_ids = set(df['read_id'])

        print('sending jobs')
        jobs = list()
        j = 0
        j_done = 0
        for read_id, predictions in tqdm(basecalls.items()):
            if read_id in processed_ids:
                j_done += 1
                continue
            reads_queue.put((read_id, references[read_id], predictions))
            j += 1

        print('reads already evaluated in output file: ' + str(j_done))
        print('jobs to be processed: ' + str(j))

        watcher_writer = mp.Process(target = results_queue_writer, args = (output_file, writer_queue, ))
        watcher_writer.start()

        print('running jobs')
        with mp.Pool(processes=args.processes) as pool:
            
            multiple_results = [pool.apply_async(eval_pair_wrapper, (reads_queue, writer_queue, tmp_dir)) for _ in range(args.processes-1)]
            results = [res.get() for res in multiple_results]
                
        writer_queue.put('kill')
        watcher_writer.join()