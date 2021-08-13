"""
This script splits the data for the 3 tasks:
- Global
- Inter-species
- Human

Reads are chosen randomly with equal amounts per species based on the max
amount of allowed reads (train+test). If a certain species does not reach 
the threshold then all available reads for that species are picked. The 
lack of reads in a species is NOT compensated in the other species.

For human, the same principle is applied but within chromosomes.

Splits are random based on pre-defined seeds for reproducibility
"""

import os
import sys
sys.path.append('../../src')
import seeds
from read import read_fna
from constants import ALL_SPECIES
import numpy as np
import pandas as pd
from copy import deepcopy
import glob
import re
import random

import argparse

def split_spe(spe, dirs, num_train_reads, num_test_reads, genome_split, spe_seed):
    """Split train and test reads for a given species
    
    Args:
        spe (str): species name
        dirs (list): list of dirs that containt that species data
        num_train_reads (int): number of reads for training
        num_test_reads (int): number of reads for testing
        genome_split (float): proportion of the genome used for training
        spe_seed (int): seed for random read choosing
    """
    
    genome_file = os.path.join('/'.join(dirs[0].split('/')[:-2]), 'genomes', spe + '.fna')
    genome = read_fna(genome_file)
    df_list = list()
    for d in dirs:
        df = pd.read_csv(os.path.join(d, 'segmentation_report.txt'), sep = '\t', header = None)
        df = df[df[2] == 'Success']
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.sort_values(by = 6)
    
    train_split = list()
    test_split = list()
    
    for k, v in genome.items():
        mp = int(len(v) * genome_split)
        
        k = k.split(' ')[0]
        
        st = df[6].searchsorted(k, side = 'left')
        nd = df[6].searchsorted(k, side = 'right')
        
        tmp_df = df[st:nd]
        ends = tmp_df[4].astype(int)
        
        train_split += tmp_df[ends < mp][0].tolist()
        test_split += tmp_df[ends > mp][0].tolist()
        
    train_split = sorted(train_split)
    random.seed(spe_seed)
    random.shuffle(train_split)
    
    test_split = sorted(test_split)
    random.seed(spe_seed+1)
    random.shuffle(test_split)
    
    
    if len(train_split) > num_train_reads:
        train_split = train_split[:num_train_reads]
        
    if len(test_split) > num_test_reads:
        test_split = test_split[:num_test_reads]
        
    return train_split, test_split

def global_split(main_dir, num_train_reads, num_test_reads, genome_split, output_dir):
    """Split data for all the species without genomic overlap
    
    For all species this is done by dividing all chromosomes/contigs by genome_split
        and taking the first fraction for training and the rest for testing. For human
        is based on odd/even chromosomes
        
    Args:
        main_dir (str): main dir where all the species are
        num_train_reads (int): total number of max reads for training
        num_test_reads (int): total number of max reads for testing
        genome_split (float): fraction of the genome used for training
        output_dir (str): dir where to save the list of reads
        
    Returns:
        a list with all the training files
        a list with all the testing files
    """
    
    dirs = glob.glob(os.path.join(main_dir, '*/*/'), recursive = True)
    
    multi_dirs = dict()
    for spe in ALL_SPECIES:
        multi_dirs[spe] = list()
        for dd in dirs:
            if re.search(spe, dd):
                multi_dirs[spe].append(dd)
    
    train_reads = list()
    test_reads = list()
    train_reads_per_spe = int(num_train_reads / len(ALL_SPECIES))
    test_reads_per_spe = int(num_test_reads / len(ALL_SPECIES))
    
    spe_seed = seeds.GLOBAL_SEED
    for spe, dirs in multi_dirs.items():
        if spe == 'Homo_sapiens':
            continue
            
        train, test = split_spe(spe, dirs, train_reads_per_spe, test_reads_per_spe, genome_split, spe_seed)
        train_reads += train
        test_reads += test
        
        spe_seed += 1
        
    with open(os.path.join(output_dir, 'global_task_train_reads.txt'), 'w') as f:
        for r in train_reads:
            f.write(r + '\n')
            
    with open(os.path.join(output_dir, 'global_task_test_reads.txt'), 'w') as f:
        for r in test_reads:
            f.write(r + '\n')
        
    return train_reads, test_reads

def calculate_dist(dist_matrix, chosen_pos):
    """Calculates how many of each bin 
    
    Args:
        dist_matrix (np.array): 2D array with binarized distance matrix
        chosen_pos (list): list of integers with the indices that belong to the 
            species for the training test
            
    Returns:
        A np.array of length number of bins with the counts of each bin
    """
    
    chosen_pos = np.array(chosen_pos)
    # given the train set, calculate at which is the minimum distance between
    # the set and the rest of species
    dist_dist = dist_matrix[chosen_pos].min(0)
    # count how many species are in each bin: very-close, close, medium, far
    c = list()
    for i in range(int(np.max(dist_matrix) + 1)):
        c.append(np.sum(dist_dist == i))
    
    # return the counts
    c = np.array(c)
    return c

def inter_species_split(bindist_matrix_file, cluster_order_file, max_species, min_species):
    """
    Find a set of species so that the rest of species fall within all the bins in respect to the 
        chosen ones.
        
    Args:
        bindist_matrix_file (str): txt file with the binarized distance matrix
        cluster_order_file (str): txt file with the names of the species in order of the matrix
        max_species (int): max number of species in the train set
        min_species (int): min number of species in the test set
    
    Returns:
        a list with the names of the species
        a list with the chosen indices for the training set
    """
    
    # load binarizred distance matrix and species order
    bin_matrix = np.loadtxt(bindist_matrix_file)
    
    names_list = list()
    with open(cluster_order_file, 'r') as f:
        for line in f:
            names_list.append(line.strip('\n'))
    
    # reproducibility seeds
    starting_seed = seeds.INTER_START_SEED
    inner_seed = seeds.INTER_INNER_SEED
    # try at least 100 times before starting again from a different starting point
    max_num_iterations_small = 100
    
    assert max_species >= min_species
    
    while True:

        iters = 0
        
        # available species
        available_pos = np.arange(0, len(names_list), 1)
        # pick one and remove from pool
        np.random.seed(starting_seed)
        curr_chosen = np.random.choice(available_pos, size = 1)
        available_pos = np.delete(available_pos, np.where(available_pos == curr_chosen)[0])

        final_chosen = [curr_chosen[0]]
        while True:

            # calculate how much does the distribution of species in bins for each of the rest
            # of species in the pool
            c_list = list()
            for p in available_pos:
                tmp_chosen = deepcopy(final_chosen)
                tmp_chosen.append(p)
                c_list.append(calculate_dist(bin_matrix, tmp_chosen))

            # choose one at random that offers the least change to the distribution and that does
            # not set any bin to zero
            c_arr = np.vstack(c_list)
            dists = np.sum(np.abs(c_arr - calculate_dist(bin_matrix, curr_chosen)), 1)
            assert len(available_pos) == len(dists)
            min_pos = np.where(dists == np.min(dists))[0] 
            min_pos = min_pos[~np.isin(min_pos, np.where(c_arr == 0)[0])]

            # if no positions left to choose from stop loop
            if len(min_pos) == 0:
                break

            pos_to_choose_from = available_pos[min_pos]
            np.random.seed(inner_seed)
            curr_chosen = np.random.choice(pos_to_choose_from, size = 1)
            available_pos = np.delete(available_pos, np.where(available_pos == curr_chosen)[0])
            final_chosen.append(curr_chosen[0])

            # if we have already the max_species limit break loop
            if len(final_chosen) >= max_species:
                break

            # if we do another loop change the seed so that we get different results
            inner_seed += 1
            
            # if too many iters, break loop and start from different start pos
            iters += 1
            if iters >= max_num_iterations_small:
                break

        # if we break from innter loop and we have enough species then we can end
        if len(final_chosen) >= min_species:
            break

        # otherwise set another seed
        starting_seed += 1
        
    print('Species in train:')
    for i in final_chosen:
        print('    ' + names_list[i])
    
    
    print('Species in test:')
    for i, n in enumerate(names_list):
        m = np.min(bin_matrix[sorted(final_chosen), i])
        if i in final_chosen:
            continue
        print('    ' + names_list[i] + ': ' + str(m))
        
    return names_list, final_chosen 

def inter_reads_split(main_dir, training_species, testing_species, num_training_reads, num_testing_train_reads, num_testing_test_reads, output_dir):
    """Split the reads between train and test for training species and choose
    testing reads for test set.
    
    Args: 
        main_dir (str): path to where all the data is
        training_species (list): list with species used for training
        testing_species (list): list with species used for testing
        num_training_reads (int): total number of training reads
        num_testing_train_reads (int): total number of testing reads for the train species
        num_testing_test_reads (int): total number of testing reads for the test species
        output_dir (str): dir where to save the list of reads for each set
        
    Returns:
        a list with all the training files
        a list with all the testing files
    """
    
    # get all the paths with the species names
    dirs = glob.glob(os.path.join(main_dir, '*/*/'), recursive = True)
    
    # split paths between train and test species
    train_dirs = dict()
    for train_s in training_species:
        train_s = train_s.replace(' ', '_')
        train_dirs[train_s] = list()
        for dd in dirs:
            if re.search(train_s, dd):
                train_dirs[train_s].append(dd)
                
    test_dirs = dict()
    for test_s in testing_species:
        test_s = test_s.replace(' ', '_')
        test_dirs[test_s] = list()
        for dd in dirs:
            if re.search(test_s, dd):
                test_dirs[test_s].append(dd)
                
    # to calculate number of training and testing reads in case that lower amount of 
    # reads are available
    reads_per_spe = (num_training_reads + num_testing_train_reads)/len(train_dirs)
    train_test_ratio = num_training_reads/num_testing_train_reads
    
    reads_seed = seeds.INTER_READ_SEED
    
    all_train_files = list()
    all_test_files = list()
    
    for spe, dirs in train_dirs.items():
        df_list = list()
        for d in dirs:
            df_list.append(pd.read_csv(os.path.join(d, 'segmentation_report.txt'), sep = '\t', header = None))

        df = pd.concat(df_list)
        df = df[df[2] == 'Success']

        # maintain train/test ratio in case of less reads
        available_reads = len(df)
        if available_reads > reads_per_spe:
            num_train_reads = int((reads_per_spe/(train_test_ratio + 1)) * train_test_ratio)
            num_test_reads = int(reads_per_spe/(train_test_ratio + 1))
        else:
            num_train_reads = int((available_reads/(train_test_ratio + 1)) * train_test_ratio)
            num_test_reads = int(available_reads/(train_test_ratio + 1))

        # sort and randomize order with reproducibility
        read_files = sorted(df[0].tolist())
        random.seed(reads_seed)
        random.shuffle(read_files)

        train_files = read_files[:num_train_reads]
        test_files = read_files[num_train_reads:num_test_reads + num_train_reads]

        reads_seed += 1
        
        all_train_files += train_files
        all_test_files += test_files
        
    # do the same for the test species
    reads_per_spe = int(num_testing_test_reads/len(test_dirs))
    for spe, dirs in test_dirs.items():
        df_list = list()
        for d in dirs:
            df_list.append(pd.read_csv(os.path.join(d, 'segmentation_report.txt'), sep = '\t', header = None))

        df = pd.concat(df_list)
        df = df[df[2] == 'Success']
        
        available_reads = len(df)
        
        if available_reads == 0:
            continue
        elif available_reads > reads_per_spe:
            num_test_reads = reads_per_spe
        else:
            num_test_reads = available_reads
        
        read_files = sorted(df[0].tolist())
        random.seed(reads_seed)
        random.shuffle(read_files)
        
        test_files = read_files[:num_test_reads]
        all_test_files += test_files
        
    # write the results
    with open(os.path.join(output_dir, 'inter_task_train_reads.txt'), 'w') as f:
        for r in all_train_files:
            f.write(r + '\n')
            
    with open(os.path.join(output_dir, 'inter_task_test_reads.txt'), 'w') as f:
        for r in all_test_files:
            f.write(r + '\n')
    
    return all_train_files, all_test_files
    



def main(main_dir, output_dir, bindist_matrix, cluster_order, max_species, min_species, 
         inter_traintrain, inter_traintest, inter_test, global_train, global_test, genome_split, human_train, human_test):
    
    # inter species 
    names_list, final_chosen = inter_species_split(bindist_matrix, cluster_order, max_species, min_species)
    
    training_species = list()
    testing_species = list()
    for i, n in enumerate(names_list):
        if i in final_chosen:
            training_species.append(n)
        else:
            testing_species.append(n)
    
    inter_reads_split(main_dir, training_species, testing_species, inter_traintrain, inter_traintest, inter_test, output_dir)
    
    # human species
    
    
    # global
    global_split(main_dir, global_train, global_test, genome_split, output_dir)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help='Path to the main data dir where the segmentation_report.txt files are')
    parser.add_argument("--bindist-matrix", type=str, help='Path to the binarized distance matrix between species')
    parser.add_argument("--cluster-order", type=str, help='Path to the txt file with the order of the clustering')
    parser.add_argument("--output-dir", type=str, help='Path to save the output files')
    parser.add_argument("--max-species", type=int, help='Number of max species in the train set for the inter-species task', default = 12)
    parser.add_argument("--min-species", type=int, help='Number of min species in the train set for the inter-species task', default = 10)
    parser.add_argument("--inter-traintrain", type=int, help='Number of reads for training in the inter-species task', default = 50000)
    parser.add_argument("--inter-traintest", type=int, help='Number of reads for testing in the inter-species task from the train species set', default = 10000)
    parser.add_argument("--inter-test", type=int, help='Number of reads for testing in the inter-species task from the test species set', default = 20000)
    parser.add_argument("--global-train", type=int, help='Number of reads for training for the global task', default = 100000)
    parser.add_argument("--global-test", type=int, help='Number of reads for testing for the global task', default = 50000)
    parser.add_argument("--human-train", type=int, help='Number of reads for training for the human task', default = 100000)
    parser.add_argument("--human-test", type=int, help='Number of reads for testing for the human task', default = 50000)
    parser.add_argument("--genome-split", type=float, help='Fraction of the genome that should be used for training', default = 0.5)
    args = parser.parse_args()
    
    main(main_dir = args.path, 
         output_dir = args.output_dir, 
         bindist_matrix = args.bindist_matrix, 
         cluster_order = args.cluster_order, 
         max_species = args.max_species, 
         min_species = args.min_species, 
         inter_traintrain = args.inter_traintrain, 
         inter_traintest = args.inter_traintest, 
         inter_test = args.inter_test, 
         global_train = args.global_train, 
         global_test = args.global_test, 
         genome_split = args.genome_split, 
         human_train = args.human_train, 
         human_test = args.human_test)