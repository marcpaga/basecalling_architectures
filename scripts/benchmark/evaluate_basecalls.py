"""This script will evaluate the basecalling performance, it produces two files:
    - A per read table file
    - A global report file
"""

import argparse
import os
import sys
import re
import multiprocessing as mp

import pandas as pd

from tqdm import tqdm

sys.path.append('../../src')
from evaluation import eval_pair, REPORT_COLUMNS
from read import iter_fastq, iter_fasta

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
            

def eval_pair_wrapper(references, prediction, write_queue):
    """Wrapper evaluate a prediction in the queue
    Args:
        references (dict): dictionary with reference sequences
        read_queue (multiprocessing.Queue): queue from where to get the predictions
        writer_queue (multiprocessing.Queue): queue where to send the results
    """
    
    read_id, pred = prediction
    try:
        ref = references[read_id]
        result = eval_pair(ref, pred)
        result['comment'] = 'pass'
    except KeyError:
        result = dict()
        for k in REPORT_COLUMNS:
            result[k] = None
        result['comment'] = 'read id not found'

    result['read_id'] = read_id
    writer_queue.put(result)
            
    return None

    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--basecalls-path", type=str, help='Path to fasta or fastq files')
    parser.add_argument("--references-path", type=str, help='Path to fasta reference files')
    parser.add_argument("--output-file", type=str, help='Output csv file')
    parser.add_argument("--processes", type=int, help='Number of parallel processes')
    args = parser.parse_args()
    
        # read all the reference sequences into memory
    references = dict()
    for fasta_file in os.listdir(args.references_path):
        if not fasta_file.endswith('.fasta'):
            continue
        for k, v in iter_fasta(os.path.join(args.references_path, fasta_file)):
            references[k] = v

    manager = mp.Manager() 
    writer_queue = manager.Queue()
    pool = mp.Pool(processes = args.processes + 1)
    watcher_writer = pool.apply_async(results_queue_writer, (args.output_file, writer_queue))

    for bf in os.listdir(args.basecalls_path):
        if not bf.endswith('.fastq'):
            continue
        bf = os.path.join(args.basecalls_path, bf)

        basecalls = iter_fastq(bf)
        jobs = list()
        for pred in basecalls:
            jobs.append(pool.apply_async(eval_pair_wrapper, (references, pred, writer_queue)))

        for job in tqdm(jobs):
            job.get()

    writer_queue.put('kill')
    pool.close()
    pool.join()