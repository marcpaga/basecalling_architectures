"""This script prepares two numpy arrays with the data
ready to be used for dataloading for training a model.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from read import read_fast5
from normalization import normalize_signal_wrapper

import numpy as np
from math import inf
from pathlib import Path
import multiprocessing as mp
import h5py
from tqdm import tqdm

import argparse

def regular_break_points(n, chunk_len, overlap=0, align='mid'):
    """Define the start and end points of the raw data based on the 
    window length and overlap
    
    Copied from: https://github.com/nanoporetech/bonito/blob/master/bonito/cli/convert.py
    
    Args:
        n (int): length of the raw data
        chunk_len (int): window size
        overlap (int): overlap between windows
        align (str): relative to the whole length, how should the windows align
    """
    num_chunks, remainder = divmod(n - overlap, chunk_len - overlap)
    start = {'left': 0, 'mid': remainder // 2, 'right': remainder}[align]
    starts = np.arange(start, start + num_chunks*(chunk_len - overlap), (chunk_len - overlap))
    return np.vstack([starts, starts + chunk_len]).T
    
def listener_writer(queue, output_file_h5, output_file_txt, chunk_size):
    """Listens to outputs on the queue and writes to a hdf5 file
    
    Arrays are stored in the 'x' and 'y' datasets
    
    Args:
        queue (): from where to get the arrays to write
        output_file_f5 (str): output file to write to, must be hdf5
    """
    
    with open(output_file_txt, "a") as f_txt:
        with h5py.File(output_file_h5, "a") as f_h5:
            while True:
                m = queue.get()

                if isinstance(m, str):
                    if m == 'kill':
                        break
                        
                if m[0] is not None:
                    n_samples = m[0].shape[0]

                    if 'x' not in f_h5:
                        window_length = m[0].shape[1]
                        max_bases = m[1].shape[1]

                        dset_x = f_h5.create_dataset('x', (n_samples, window_length), 
                                                     maxshape=(None, window_length),
                                                     chunks = (chunk_size, window_length),
                                                     data = m[0])
                        dset_y = f_h5.create_dataset('y', (n_samples, max_bases), 
                                                     maxshape=(None, max_bases), 
                                                     chunks = (chunk_size, window_length),
                                                     data = m[1])

                    else:
                        dset_x = f_h5['x']
                        dset_y = f_h5['y']
                        dset_x.resize(dset_x.shape[0]+n_samples, axis=0)   
                        dset_x[-n_samples:] = m[0]
                        dset_y.resize(dset_y.shape[0]+n_samples, axis=0)   
                        dset_y[-n_samples:] = m[1]
                
                f_txt.write(str(m[2]) + '\n')
                f_txt.flush()
            
    return None
                
def segment_read(read_file, window_length, overlap, min_bases, max_bases, queue = None):
    """Processes a single/multi fast5 read into chunks for deep learning training
    
    Args:
        read_file (str): fast5 file
        window_Length (int): size of the chunks in raw datapoints
        overlap (int): overlap between windows
        min_bases (int): minimum number of bases for a chunk to be considered
        max_bases (int): max number of bases for a chunk to be considered
        queue_h5 (manager.queue): queue to handle the writing to hdf5 and txt files
    """
    multiread_obj = read_fast5(read_file)
    
    # lists to store all the segments
    x_list = list() 
    y_list = list()

    for k in multiread_obj.keys():
        read_obj = multiread_obj[k]

        if read_obj.segmentation is None:
            continue

        read_data = dict()
        read_data['Dacs'] = read_obj.raw
        read_data['Ref_to_signal'] = read_obj.segmentation['start'] + read_obj.start_rel_to_raw
        read_data['Reference'] = ''.join(read_obj.segmentation['base'])
        read_data['range'] = read_obj.range
        read_data['digitisation'] = read_obj.digitisation
        read_data['offset'] = read_obj.offset
        read_id = read_obj.read_id

        # segment start and end points
        break_points = regular_break_points(len(read_data['Dacs']), window_length, overlap = overlap, align = 'left')

        # normalize the data
        sample = normalize_signal_wrapper(read_data['Dacs'], read_data['offset'], read_data['range'], read_data['digitisation'], 
                                          method = 'noisiest', samples=100, threshold=6.0, factor=1.4826)

        # get the positions of the sequence and the sequence as a whole
        pointers = read_data['Ref_to_signal']
        reference_str = read_data['Reference']

        for l, ((i, j), (ti, tj)) in enumerate(zip(break_points, np.searchsorted(pointers, break_points))):

            # if they do not fit the min and max filters skip
            str_len = tj-ti
            if str_len > max_bases:
                continue
            if str_len < min_bases:
                continue

            # extract the data that we need
            x_list.append(sample[i:j])
            y_list.append(reference_str[ti:tj])

    if len(x_list) < 1:
        if queue:
            queue.put((None, None, read_file))
        return None, None
    # stack all the arrays
    x = np.vstack(x_list)
    # for the y array we have to pre allocate the array as each
    # segment can be of a different length
    if max_bases == inf:
        max_len = window_length
    else:
        max_len = max_bases
    
    # use S1, do not use U1 as it is not supported by hdf5
    y = np.full((len(y_list), max_len), '', dtype = 'S1')
    for i, seg_str in enumerate(y_list):
        seg_str = list(seg_str)
        y[i, 0:len(seg_str)] = seg_str   
        
    if queue:
        queue.put((x, y, read_file))
    
    return x, y

def main(fast5_dir, fast5_files_list, output_dir, window_length, window_overlap, min_bases, max_bases, chunk_size, 
         n_cores = 1, verbose = True):
    """Processes a set of fast5 reads into chunks for deep learning training
    
    Args:
        fast5_dir (str): directory with fast5 reads
        fast5_files_list (str): file with list of fast5 files to process
        output_dir (str): dir where to store the processed data
        window_Length (int): size of the chunks in raw datapoints
        window_overlap (int): overlap between windows
        min_bases (int): minimum number of bases for a chunk to be considered
        max_bases (int): max number of bases for a chunk to be considered
        n_cores (int): number of processes
        verbose (bool): output a progress bar
    """
    
    output_file = os.path.join(output_dir, "data_" +str(window_length) + "-" + str(window_overlap) + ".hdf5")
    processed_files_file = os.path.join(output_dir, "processed_" +str(window_length) + "-" + str(window_overlap) + ".txt")
    
    already_processed_fast5_files = list()
    if os.path.isfile(processed_files_file):
        with open(processed_files_file, 'r') as f:
            for line in f:
                already_processed_fast5_files.append(line.strip('\n'))
    already_processed_fast5_files = set(already_processed_fast5_files)
    print('Found ' + str(len(already_processed_fast5_files)) + ' files already processed')
    
    print('Finding files to process')
    # find all the files that we have to process
    files_list = list()
    if fast5_dir is not None:
        for path in Path(fast5_dir).rglob('*.fast5'):
            filename = str(path)
            if filename not in already_processed_fast5_files:
                files_list.append(filename)
    elif fast5_files_list is not None:
        with open(fast5_files_list, 'r') as f:
            for line in f:
                filename = line.strip('\n')
                if filename not in already_processed_fast5_files:
                    files_list.append(filename)
    else:
        raise ValueError('Either --fast5-dir or --fast5-list must be given')
        
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(processes = n_cores + 2) # add two to ensure that there is a process for the queue writer
    writer = pool.apply_async(listener_writer, (q, output_file, processed_files_file, chunk_size))
    
    jobs = list()
    for file in files_list[:1000]:
        
        if not os.path.isfile(file):
            print('File not found, skipping: ' + file)
            continue
        
        jobs.append(pool.apply_async(segment_read, (file, window_length, window_overlap, min_bases, max_bases, q)))
    
    print('Processing ' + str(len(jobs)) +  ' files')
    for job in tqdm(jobs, disable = not verbose):
        job.get()
    
    q.put('kill')
    pool.close()
    pool.join()
    
    return None

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast5-dir", type=str, help='Path to fast5 files', default = None)
    parser.add_argument("--fast5-list", type=str, help='Path to file with list of files to be processed', default = None)
    parser.add_argument("--output-dir", type=str, help='Path to save the numpy arrays')
    parser.add_argument("--window-size", type=int, help='Size of the window of a raw data segment', default = 2000)
    parser.add_argument("--window-slide", type=int, help='Number of datapoints of overlap between sequential segments', default = 0)
    parser.add_argument("--min-bases", type=int, help='Minimum number of bases for a segment to be kept', default = 10)
    parser.add_argument("--max-bases", type=int, help='Maximum number of bases for a segment to be kept', default = inf)
    parser.add_argument("--chunk-size", type=int, help='Sample chunk size for writing the hdf5', default = 64)
    parser.add_argument("--n-cores", type=int, help='Number of parallel processes', default = 1)
    parser.add_argument("--verbose", action='store_true', help='Print a progress bar')
    args = parser.parse_args()
    
    main(fast5_dir = args.fast5_dir, 
         fast5_files_list = args.fast5_list, 
         output_dir = args.output_dir,
         window_length = args.window_size, 
         window_overlap = args.window_slide, 
         min_bases = args.min_bases, 
         max_bases = args.max_bases, 
         chunk_size = args.chunk_size, 
         n_cores = args.n_cores, 
         verbose = args.verbose)


