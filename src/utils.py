"""Contains general utilities
"""
import re
from constants import ALIGN_FUNCTION, MATRIX, ALIGNMENT_GAP_OPEN_PENALTY, ALIGNMENT_GAP_EXTEND_PENALTY

def stich_segments():
    """Stiches different predicted segments together
    """
    raise NotImplemetedError('TODO')
    
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
            if s != blank_label and s != prev_s:
                seq += decode_dict[int(s)]
                prev_s = s
        decoded_predictions.append(seq)
    return decoded_predictions

def elongate_cigar(cigar):
    cigar_counts = re.split('H|X|=|I|D|N|S|P|M', cigar)
    cigar_strs = re.split('[0-9]', cigar)
    
    cigar_counts = [c for c in cigar_counts if c != '']
    cigar_strs = [c for c in cigar_strs if c != '']
    
    assert len(cigar_strs) == len(cigar_counts)
    
    longcigar = ''
    for c, s in zip(cigar_counts, cigar_strs):
        longcigar += s*int(c)
    return longcigar, cigar_counts, cigar_strs

def alignment_accuracy(y, p, alignment_function = ALIGN_FUNCTION, matrix = MATRIX, 
                       open_penalty = ALIGNMENT_GAP_OPEN_PENALTY, extend_penalty = ALIGNMENT_GAP_EXTEND_PENALTY):
    """Calculates the accuracy between two sequences
    Accuracy is calculated by dividing the number of matches 
    over the length of the true sequence.
    
    Args:
        y (str): true sequence
        p (str): predicted sequence
        alignment_function (object): alignment function from parasail
        matrix (object): matrix object from `parasail.matrix_create`
        open_penalty (int): penalty for opening a gap
        extend_penalty (int): penalty for extending a gap
        
    Returns:
        (float): with the calculated accuracy
    """
    
    if len(p) == 0:
        if len(y) == 0:
            return 1
        else:
            return 0
    
    alignment = alignment_function(p, y, open_penalty, extend_penalty, matrix)
    decoded_cigar = alignment.cigar.decode.decode()
    long_cigar, cigar_counts, cigar_strs = elongate_cigar(decoded_cigar)
    if len(long_cigar) == 0:
        return 0
    
    matches = 0
    for s, i in zip(cigar_strs, cigar_counts):
        if s == '=':
            matches += int(i)
    
    return matches/len(y)

def read_metadata(file_name):
    """Read the metadata of a npz file
    
    Args:
        filename (str): .npz file that we want to read the metadata from
        
    Returns:
        (list): with as many items as arrays in the file, each item in the list
        is filename (within the zip), shape, pickled (I think?), dtype
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