"""Cluster the different species based on k-mer content

It's done based on Jensen-Shannon distance and single linkeage clustering.
"""

import os
import numpy as np
import re

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import single, leaves_list, dendrogram
from matplotlib import pyplot as plt

import argparse

def main(kmer_dir, kmer_size, divisions, output_dir, plot_dir = None):
    """Cluster the different species based on k-mer content
    
    Args:
        kmer_dir (str): dir where the k-mer counts files are 
        kmer_size (int): k-mer size to be analized
        divisions (int): number of bins for the binning of the distance matrix
        output_dir (str): path to save the distance matrices and species order
        plot_dir (str): path to save the clustering plots
    """
    
    # find all files that have counts for that k-mer size
    pattern = 'k' + str(kmer_size)
    matching_files = list()
    for f in os.listdir(kmer_dir):
        if re.search(pattern, f):
            matching_files.append(f)
            
    # read all the data and make a frecuencies matrix
    counts_list = list()
    names_list = list()
    for f in matching_files:
        counts = np.zeros((4**kmer_size, ), dtype = np.float64)
        with open(os.path.join(kmer_dir, f), 'r') as of:
            for i, line in enumerate(of):
                counts[i] = int(line.split('\t')[1].strip('\n'))

        # convert k-mer counts to k-mer frequencies
        counts /= np.sum(counts)
        counts_list.append(counts)
        names_list.append(" ".join(f.split('_')[:-1]))
    X = np.vstack(counts_list)
    
    # calculate the distance matrix
    dist = pdist(X, 'jensenshannon')
    
    # cluster
    Z = single(dist)
    order = leaves_list(Z)
    
    # write the order of the clusters
    with open(os.path.join(output_dir, 'clustering_species_order_k'+str(kmer_size)+'.txt'), 'w') as f:
        for s in np.array(names_list)[order].tolist():
            f.write(s + '\n')
    
    if plot_dir:
        fig = plt.figure(figsize=(10, 5))
        dendrogram(Z, labels = names_list, leaf_rotation = 90, color_threshold = 0.3)
        plt.show()
        fig.savefig(os.path.join(plot_dir, 'dendrogram_k'+str(kmer_size)+'.png'), dpi=fig.dpi, bbox_inches='tight')
                    
    # convert dense distance matrix to complete
    dist_matrix = squareform(dist)
    dist_matrix = dist_matrix[order, ]
    dist_matrix = dist_matrix[:, order]
                    
    np.savetxt(os.path.join(output_dir, 'distance_matrix_k'+str(kmer_size)+'.txt'), dist_matrix)
                    
    if plot_dir:
        fig = plt.figure(figsize=(15, 15))
        plt.imshow(dist_matrix, cmap = plt.get_cmap('magma'))
        plt.xticks(np.arange(0, len(names_list), 1), labels = np.array(names_list)[order], rotation = 90)
        plt.yticks(np.arange(0, len(names_list), 1), labels = np.array(names_list)[order])
        plt.colorbar()
        plt.show()
        fig.savefig(os.path.join(plot_dir, 'distance_matrix_k'+str(kmer_size)+'.png'), dpi=fig.dpi, bbox_inches='tight')
                    
    # binarize the distance matrix
    s = 0
    d = np.max(dist_matrix) / divisions
    n = np.max(dist_matrix) / divisions
    bin_matrix = np.zeros(dist_matrix.shape)
    for i in range(divisions+1):
        bin_matrix[ (dist_matrix > s) & (dist_matrix <= n) ] = i
        s = n
        n += d
                    
    np.savetxt(os.path.join(output_dir, 'distance_matrix_binarized_'+str(divisions)+'_k'+str(kmer_size)+'.txt'), bin_matrix)
    
    if plot_dir:
        fig = plt.figure(figsize=(15, 15))
        plt.imshow(bin_matrix, cmap = plt.get_cmap('magma'))
        plt.xticks(np.arange(0, len(names_list), 1), labels = np.array(names_list)[order], rotation = 90)
        plt.yticks(np.arange(0, len(names_list), 1), labels = np.array(names_list)[order])
        plt.colorbar()
        plt.show()
        fig.savefig(os.path.join(plot_dir, 'distance_matrix_binarized_'+str(divisions)+'_k'+str(kmer_size)+'.png'), dpi=fig.dpi, bbox_inches='tight')
    
    return np.array(names_list)[order], dist_matrix, bin_matrix

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help='Path to the dir where the k-mer counts are')
    parser.add_argument("--kmer-size", type=int, help='k-mer size to be analyzed')
    parser.add_argument("--divisions", type=int, help='Number of bins to binarized the distance matrix')
    parser.add_argument("--output-dir", type=str, help='Path to save the distance matrices and order of species in clustering')
    parser.add_argument("--plot-dir", type=str, help='Path to save the cluster plots', default = None)
    args = parser.parse_args()
    
    main(kmer_dir = args.path, 
         kmer_size = args.kmer_size, 
         divisions = args.divisions, 
         output_dir = args.output_dir, 
         plot_dir = args.plot_dir)