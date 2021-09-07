"""
Script to delete segmentation tables in fast5 files or
write a report on the segmentation status of fast5 files.
"""


import os
import sys
import glob
import multiprocessing as mp
import h5py

sys.path.append('../../src')

import read
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse



def delete_table(file_path):
    """Delete the segmentation table from a fast5 file
    
    Args:
        file_path (str): path to the fast5 file
    """
    
    corr_grp_slot = 'RawGenomeCorrected_000/BaseCalled_template/Events'
    with h5py.File(file_path, 'r+') as fh:
        try:
            analysis_grp = fh['/Analyses']
            del analysis_grp[corr_grp_slot]
        except KeyError:
            pass
        
def report_read(file_path, q, p):
    """Extract relevant info from the segmented read
    """
    
    read_data = read.read_fast5(file_path)
    read_id = list(read_data.keys())[0]
    read_data = read_data[read_id]
    
    if read_data.segmentation is None:
        s = file_path + '\t' + read_id + '\t' + 'Failed_to_segment' + '\t' + 'NaN' + '\t' + 'NaN' + '\t' + 'NaN' + '\t' + 'NaN'
        q.put(s)
        return file_path, read_id, 'Failed to segment'
    
    st = read_data.alignment_info['mapped_start']
    nd = read_data.alignment_info['mapped_end']
    strand = read_data.alignment_info['mapped_strand']
    chrom = read_data.alignment_info['mapped_chrom']
    
    s = file_path + '\t' + read_id + '\t' + 'Success' + '\t' + str(st) + '\t' + str(nd) + '\t' + str(strand) + '\t' + str(chrom)
    q.put(s)
    
    file_name = file_path.split('/')[-1].split('.')[0]
    s = '>' + str(read_id) + '\n' + str("".join(read_data.segmentation['base'].tolist()))
    p.put(s)
    return file_path, read_id, 'Success'
    
def listener_writer(queue, output_file):
    """Listens to outputs on the queue and writes to a file
    """
    
    with open(output_file, 'a') as f:
        while True:
            m = queue.get()
            if m == 'kill':
                break
            f.write(str(m) + '\n')
            f.flush()
            
            
def main(fast5_path, output_file, n_cores, mode, verbose = True):
    """Process all the reads with multiprocessing
    Args:
        fast5_path (str): path to fast5 files, searched recursively
        reference_file (str): fasta file with references
        output_file (str): output txt file to write outcome of resquiggle
        n_cores (int): number of parallel processes
        mode (str): 'delete' for delete segmentation data and 'report' for report table
        verbose (bool): output a progress bar
    """
    
    if mode == 'report':
        # queue for writing to a report
        manager = mp.Manager() 
        q = manager.Queue()  # report queue
        p = manager.Queue()  # fasta queue
        pool = mp.Pool(n_cores) # pool for multiprocessing
        watcher_report = pool.apply_async(listener_writer, (q, output_file))
        fasta_file = os.path.join("/".join(output_file.split('/')[:-1]), 'read_references_benchmark.fasta')
        watcher_fasta = pool.apply_async(listener_writer, (p, fasta_file))

        processed_reads = list()
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    processed_reads.append(line.split('\t')[0])
        
        else:
            with open(output_file, 'w') as f:
                s = 'file'+'\t'+'read_id'+'\t'+'result'+'\t'+'align_st'+'\t'+'align_nd'+'\t'+'align_strand'+'\t'+'align_chrom'+'\n'
                f.write(s)
        processed_reads = set(processed_reads)
    
        if len(processed_reads) > 0:
            print('Reads already processed: ' + str(len(processed_reads)))

        jobs = list()
        file_list = glob.glob(fast5_path + '/**/*.fast5', recursive=True)
        for file_path in file_list:
            if file_path in processed_reads:
                continue

            job = pool.apply_async(report_read, (file_path, q, p))
            jobs.append(job)

        print('Reads to be processed: ' + str(len(jobs)))

        for job in tqdm(jobs, disable = not verbose):
            job.get()

        q.put('kill')
        p.put('kill')
        pool.close()
        pool.join()

        df = pd.read_csv(output_file, sep = '\t', header = None)
        successes = np.sum(df[2] == 'Success')
        total = len(df)
        perc = round((successes/total)*100, 2)
    
        print(str(successes)+'/'+str(total)+' ('+str(perc)+'%) successfully segmented reads')
        
    elif mode == 'delete':
        
        pool = mp.Pool(n_cores)
        jobs = list()
        file_list = glob.glob(fast5_path + '/**/*.fast5', recursive=True)
        for file_path in file_list:
            job = pool.apply_async(delete_table, (file_path,))
            jobs.append(job)
            
        print('Reads to be processed: ' + str(len(jobs)))
        for job in tqdm(jobs, disable = not verbose):
            job.get()
            
        pool.close()
        pool.join()
        
    else:
        raise ValueError('mode must be "delete" or "report"')
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast5-path", type=str, help='Path to fast5 files, it is searched recursively')
    parser.add_argument("--output-file", type=str, help='Text file that contains information on the result of the segmentation')
    parser.add_argument("--n-cores", type=int, help='Number of processes')
    parser.add_argument("--delete", action='store_true', help='Run script to delete segmentation tables')
    parser.add_argument("--report", action='store_true', help='Run script to summarize read info')
    parser.add_argument("--verbose", action='store_true', help='Output a progress bar')
    
    args = parser.parse_args()
    
    if args.delete and args.report:
        raise ValueError('--delete and --report flags cannot both be given')
        
    if args.delete:
    
        main(fast5_path = args.fast5_path, 
             output_file = None, 
             n_cores = args.n_cores, 
             verbose = args.verbose, 
             mode = 'delete')
        
    elif args.report:
        
        main(fast5_path = args.fast5_path, 
             output_file = args.output_file, 
             n_cores = args.n_cores, 
             verbose = args.verbose, 
             mode = 'report')
    
    
    
    
    