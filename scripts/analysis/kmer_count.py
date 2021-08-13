"""
Count the k-mer frequency of genomes.
"""

import os
import sys
sys.path.append('../../src')
from read import read_fna

import glob
import numpy as np
import itertools
from tqdm import tqdm
import multiprocessing as mp
import re

import argparse

def count_kmers_genome(genome_file, output_dir, kmer_size):
    """Count the kmers in a genome
    
    Counts are written to txt file with format KMER\tCOUNTS\n
    Lower case or N bases in genome are not counted
    
    Args:
        genome_file (str): file that contains the genome, in .fna format
        output_dir (str): path to where to write the counts
        kmer_size (int): kmer size
    """
    
    genome = read_fna(genome_file)
    
    ## this is the same for all, but takes so little to calculate
    mers = ["".join(k) for k in itertools.product('ACGT', repeat= kmer_size)]
    mer_to_pos = dict()
    for i, k in enumerate(mers):
        mer_to_pos[k] = i
        
    kmer_counts = np.zeros((len(mers), ), dtype = np.int64)
    for k, v in tqdm(genome.items()): # go over chromosomes and plasmids
        
        # just go for the main genomic reference
        if re.search('unplaced', k):
            continue
        if re.search('unlocalized', k):
            continue
            
        for i in range(len(v) - kmer_size + 1):
            kmer = v[i:i+kmer_size]
            try:
                kmer_counts[mer_to_pos[kmer]] += 1
            except KeyError: # ignore N and acgt
                continue
    
    genome_file_name = genome_file.split('/')[-1].split('.')[0] + '_k' + str(kmer_size) + '.txt'
    output_file = os.path.join(output_dir, genome_file_name)
    with open(output_file, 'w') as f:
        for k, c in zip(mers, kmer_counts):
            f.write(str(k) + '\t' + str(c) + '\n')
            
    return kmer_counts

def main(path, output_dir, kmer_size, n_cores = 1, verbose = True):
    """
    Args:
        path (str): path where to look for fna files, recursively
        output_dir (str): path to store the output files
        kmer_size (list): list of integers that denote the kmers sizes to evaluate
        n_cores (int): number of processes, defaults to 1
        verbose (bool): output progress bar, defaults to True
    """
        
    ## multiprocessing
    pool = mp.Pool(n_cores)
    ## get all the fna files
    file_list = glob.glob(path + '/**/*.fna', recursive=True)
    
    ## send jobs for all genome and kmer size combinations
    jobs = list()
    for file_path in file_list:
        for kmer in kmer_size:
            job = pool.apply_async(count_kmers_genome, (file_path, output_dir, kmer))
            jobs.append(job)
    
    print('Genomes*kmers to be processed: ' + str(len(jobs)))
    for job in tqdm(jobs, disable = not verbose):
        job.get()
        
    pool.close()
    pool.join()

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help='Path to where to search for genome files (in fna format)')
    parser.add_argument("--output-dir", type=str, help='Text file that contains information on the result of the segmentation')
    parser.add_argument("--kmer-size", type=int, nargs='+', help='Kmer size')
    parser.add_argument("--n-cores", type=int, help='Number of processes')
    parser.add_argument("--verbose", action='store_true', help='Output a progress bar')
    
    args = parser.parse_args()
    
    main(path = args.path,
         output_dir = args.output_dir,
         kmer_size = args.kmer_size,
         n_cores = args.n_cores, 
         verbose = args.verbose)
    

