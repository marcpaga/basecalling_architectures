"""
Wrapper code to simulate the mean signal of each base using tombo and its
signal model.
"""

from tombo import tombo_helper as th
from tombo import tombo_stats as ts
from tombo._default_parameters import DNA_SAMP_TYPE, RNA_SAMP_TYPE
import numpy as np

def seq_to_signal(seq):
    """Compute expected signal levels for a sequence from a reference model
    Args:
        seq (str): genomic seqeunce to be converted to expected signal levels
        std_ref (:class:`tombo.tombo_stats.TomboModel`): expected signal level 
            model
        rev_strand (bool): flip sequence (after extracting k-mers for expected 
            level model lookup)
        alt_ref (:class:`tombo.tombo_stats.TomboModel`): an alternative     
            expected signal level model
    Note:
        Returned expected signal levels will be trimmed compared to the passed 
        sequence based on the `std_ref.kmer_width` and `std_ref.central_pos`.
    Returns:
        Expected signal level references
        1) ref_means (`np.array::np.float64`) expected signal levels
        2) ref_sds (`np.array::np.float64`) expected signal level sds
        3) alt_means (`np.array::np.float64`) alternate expected signal levels
        4) alt_sds (`np.array::np.float64`) alternate expected signal level sds
    """
    
    seq_samp_type = th.seqSampleType(DNA_SAMP_TYPE, False)
    std_ref = ts.TomboModel(seq_samp_type=seq_samp_type)
    
    seq_kmers = [seq[i:i + std_ref.kmer_width]
                 for i in range(len(seq) - std_ref.kmer_width + 1)]

    try:
        ref_means = np.array([std_ref.means[kmer] for kmer in seq_kmers])
        ref_sds = np.array([std_ref.sds[kmer] for kmer in seq_kmers])
    except KeyError:
        th.error_message_and_exit(
            'Invalid sequence encountered from genome sequence.')
    
    return ref_means

    

