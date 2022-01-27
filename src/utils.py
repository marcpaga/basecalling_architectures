"""Contains general utilities
"""
import zipfile
import numpy as np
import torch

def stich_segments():
    """Stiches different predicted segments together
    """
    raise NotImplementedError('TODO')
    
def decode_batch_greedy_ctc(y, decode_dict, blank_label = 0):
    """Decodes a batch of CTC predictions in a greedy manner
    by merging consequent labels together and removing blanks.
    
    Args:
        y (tensor): tensor with shape [batch, len]
        decode_dict (dict): dict with integer to string mapping
        blank_label (int): integer that denotes blank label
    """
    
    decoded_predictions = list()
    for i in range(y.shape[0]):
        seq = ''
        prev_s = blank_label
        for s in y[i]:
            s = int(s)
            if s == blank_label:
                prev_s = blank_label
            elif s != prev_s:
                prev_s = s
                seq += decode_dict[s]
            elif s == prev_s:
                continue
        decoded_predictions.append(seq)
    return decoded_predictions

def read_metadata(file_name):
    """Read the metadata of a npz file
    
    Args:
        filename (str): .npz file that we want to read the metadata from
        
    Returns:
        (list): with as many items as arrays in the file, each item in the list
        is filename (within the zip), shape, fortran order, dtype
    """
    zip_file=zipfile.ZipFile(file_name, mode='r')
    arr_names=zip_file.namelist()

    metadata=[]
    for arr_name in arr_names:
        fp=zip_file.open(arr_name,"r")
        version=np.lib.format.read_magic(fp)

        if version[0]==1:
            shape,fortran_order,dtype=np.lib.format.read_array_header_1_0(fp)
        elif version[0]==2:
            shape,fortran_order,dtype=np.lib.format.read_array_header_2_0(fp)
        else:
            print("File format not detected!")
        metadata.append((arr_name,shape,fortran_order,dtype))
        fp.close()
    zip_file.close()
    return metadata

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths

def stitch_by_stride(chunks, chunksize, overlap, length, stride, reverse=False):
        """
        Stitch chunks together with a given overlap
        
        This works by calculating what the overlap should be between two outputed
        chunks from the network based on the stride and overlap of the inital chunks.
        The overlap section is divided in half and the outer parts of the overlap
        are discarded and the chunks are concatenated. There is no alignment.
        
        Chunk1: AAAAAAAAAAAAAABBBBBCCCCC
        Chunk2:               DDDDDEEEEEFFFFFFFFFFFFFF
        Result: AAAAAAAAAAAAAABBBBBEEEEEFFFFFFFFFFFFFF
        
        Args:
            chunks (tensor): predictions with shape [samples, length, classes]
            chunk_size (int): initial size of the chunks
            overlap (int): initial overlap of the chunks
            length (int): original length of the signal
            stride (int): stride of the model
            reverse (bool): if the chunks are in reverse order
            
        Copied from https://github.com/nanoporetech/bonito
        """
        if chunks.shape[0] == 1: return chunks.squeeze(0)

        semi_overlap = overlap // 2
        start, end = semi_overlap // stride, (chunksize - semi_overlap) // stride
        stub = (length - overlap) % (chunksize - overlap)
        first_chunk_end = (stub + semi_overlap) // stride if (stub > 0) else end

        if reverse:
            chunks = list(chunks)
            return torch.cat([
                chunks[-1][:-start], *(x[-end:-start] for x in reversed(chunks[1:-1])), chunks[0][-first_chunk_end:]
            ])
        else:
            return torch.cat([
                chunks[0, :first_chunk_end], *chunks[1:-1, start:end], chunks[-1, start:]
            ])