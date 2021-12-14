"""
Deletes existing segmentation tables on Wick et al. data,
resquiggles the data using the reference sequences and reports
the results in txt file.
"""

import os
import sys
import glob
import multiprocessing as mp

from tombo import tombo_helper as th
from tombo import tombo_stats as ts
from tombo import resquiggle as tr
from tombo._default_parameters import  DNA_SAMP_TYPE, MIN_EVENT_TO_SEQ_RATIO, MAX_RAW_CPTS
from collections import namedtuple
import h5py
import mappy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import read
from normalization import normalize_signal_wrapper
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

def segment(genome_seq, read_id, all_raw_signal, norm_signal):
    """Resquiggle the raw data
    
    Args:
        genome_seq (str): DNA sequence that corresponds to the provided raw signal
        read_id (str): read id 
        all_raw_signal (np.array): raw signal
        norm_signal (np.array): normalized signal
    """
    
    outlier_thresh = None
    const_scale = None
    max_raw_cpts = MAX_RAW_CPTS

    seq_samp_type = th.seqSampleType(DNA_SAMP_TYPE, False)
    std_ref = ts.TomboModel(seq_samp_type=seq_samp_type)
    rsqgl_params = ts.load_resquiggle_parameters(seq_samp_type)

    align_info = th.alignInfo('insert_1', 
                              'insert_resquiggle', 
                              0, 0, 0, 0, len(genome_seq), 0)
    genome_loc = th.genomeLocation(0, '+', read_id)

    mean_q_score = 15
    start_clip_bases = None

    map_res = th.resquiggleResults(align_info=align_info, 
                                   genome_loc=genome_loc, 
                                   genome_seq=genome_seq,
                                   mean_q_score=mean_q_score, 
                                   start_clip_bases=start_clip_bases)

    map_res = map_res._replace(raw_signal = all_raw_signal)

    # compute number of events to find
    # ensure at least a minimal number of events per mapped sequence are found
    num_mapped_bases = len(map_res.genome_seq) - std_ref.kmer_width + 1
    num_events = ts.compute_num_events(
        map_res.raw_signal.shape[0], num_mapped_bases,
        rsqgl_params.mean_obs_per_event, MIN_EVENT_TO_SEQ_RATIO)
    # ensure that there isn't *far* too much signal for the mapped sequence
    # i.e. one adaptive bandwidth per base is too much to find a good mapping
    if num_events / rsqgl_params.bandwidth > num_mapped_bases:
        raise th.TomboError('Too much raw signal for mapped sequence')

    ## here dont get the normalized signal from segment signal, use our
    ## own passed as an argument

    valid_cpts, _, new_scale_values = tr.segment_signal(
        map_res, num_events, rsqgl_params, outlier_thresh, const_scale)

    event_means = ts.compute_base_means(norm_signal, valid_cpts)

    dp_res = tr.find_adaptive_base_assignment(
        valid_cpts, event_means, rsqgl_params, std_ref, map_res.genome_seq,
        start_clip_bases=map_res.start_clip_bases,
        seq_samp_type=seq_samp_type, reg_id=map_res.align_info.ID)
    # clip raw signal to only part mapping to genome seq
    norm_signal = norm_signal[dp_res.read_start_rel_to_raw:
                              dp_res.read_start_rel_to_raw + dp_res.segs[-1]]
    norm_signal = norm_signal.astype(np.float64)

    segs = tr.resolve_skipped_bases_with_raw(
        dp_res, norm_signal, rsqgl_params, max_raw_cpts)

    norm_params_changed = False

    sig_match_score = ts.get_read_seg_score(
        ts.compute_base_means(norm_signal, segs),
        dp_res.ref_means, dp_res.ref_sds)
    if segs.shape[0] != len(dp_res.genome_seq) + 1:
        raise th.TomboError('Aligned sequence does not match number ' +
                            'of segments produced')

    map_res = map_res._replace(
        read_start_rel_to_raw=dp_res.read_start_rel_to_raw, segs=segs,
        genome_seq=dp_res.genome_seq, raw_signal=norm_signal,
        scale_values=new_scale_values, sig_match_score=sig_match_score,
        norm_params_changed=norm_params_changed)
    
    return map_res

def write_table(file_path, segmentation_arr, read_start_rel_to_raw):
    """Write the segmentation table into a fast5 file
    
    Args:
        file_path (str): path to the fast5 file
        segmentation_arr (np.array): array with segmentation info
        read_start_rel_to_raw (int): relativity of array to raw data
    """
    
    corr_grp_slot = 'RawGenomeCorrected_000/BaseCalled_template/Events'
    with h5py.File(file_path, 'r+') as fh:
        try:
            analysis_grp = fh['/Analyses']
            del analysis_grp[corr_grp_slot]
        except KeyError:
            analysis_grp = fh.create_group('Analyses')
        corr_events = analysis_grp.create_dataset('RawGenomeCorrected_000/BaseCalled_template/Events', data=segmentation_arr, compression="gzip")
        corr_events.attrs['read_start_rel_to_raw'] = read_start_rel_to_raw

def segment_read(file_path, reference, alignment, q, p):
    """Wrapper to segment a read
    """
    
    read_data = read.read_fast5(file_path)
    read_id = list(read_data.keys())[0]
    read_data = read_data[read_id]
        
    st = alignment['r_st']
    nd = alignment['r_en']
    strand = '+' if alignment['strand'] == 1 else '-'
    chrom = alignment['ctg']
    
    
    norm_signal = normalize_signal_wrapper(read_data.raw, 
                                           offset = read_data.offset, 
                                           range = read_data.range, 
                                           digitisation = read_data.digitisation, 
                                           factor = 1) 
    
    try:
        seg_res = segment(reference, read_id, read_data.raw, norm_signal)
    except:
        s = file_path + '\t' + read_id + '\t' + 'Failed_to_segment' + '\t' + str(st) + '\t' + str(nd) + '\t' + str(strand) + '\t' + str(chrom)
        q.put(s)
        return file_path, read_id, 'Failed to segment'
    
    means = list()
    lens = list()
    r = seg_res.read_start_rel_to_raw
    for i in range(len(seg_res.segs)-1):
        s = seg_res.segs[i]
        n = seg_res.segs[i+1]
        means.append(np.mean(norm_signal[r + s:r + n]))
        lens.append(n-s)
        
    segmentation_arr = np.zeros(len(seg_res.segs)-1,
                            dtype=[('norm_mean', '<f8'),
                                   ('norm_stdev', '<f8'),
                                   ('start', '<u4'), 
                                   ('length', '<u4'),
                                   ('base', '<S1')])

    segmentation_arr['norm_mean'] = means
    segmentation_arr['norm_stdev'] = [np.nan] * len(means)
    segmentation_arr['start'] = seg_res.segs[:-1]
    segmentation_arr['length'] = lens
    segmentation_arr['base'] = list(reference[2:-3])
    
    
    try:
        write_table(file_path, segmentation_arr, r)
    except:
        s = file_path + '\t' + read_id + '\t' + 'Failed_to_write' + '\t' + str(st) + '\t' + str(nd) + '\t' + str(strand) + '\t' + str(chrom)
        q.put(s)
        return file_path, read_id, 'Failed to write'

    s = file_path + '\t' + read_id + '\t' + 'Success' + '\t' + str(st) + '\t' + str(nd) + '\t' + str(strand) + '\t' + str(chrom)
    q.put(s)
    
    s = '>' + str(read_id) + '\n' + str("".join(read_data.segmentation['base'].tolist()))
    p.put(s)
    
    return file_path, read_id, 'Success'

def listener_writer(q, output_file):
    """Listens to outputs on the queue and writes to a file
    """
    
    with open(output_file, 'a') as f:
        while True:
            m = q.get()
            if m == 'kill':
                break
            f.write(str(m) + '\n')
            f.flush()
            
def main(fast5_path, reference_file, genome_file, output_file, n_cores, verbose = True):
    """Process all the reads with multiprocessing
    Args:
        fast5_path (str): path to fast5 files, searched recursively
        reference_file (str): fasta file with read references
        genome_file (str): fasta file with genome reference
        output_file (str): output txt file to write outcome of resquiggle
        n_cores (int): number of parallel processes
        verbose (bool): output a progress bar
    """
    
    # queue for writing to a report
    manager = mp.Manager()
    q = manager.Queue()  # report queue
    p = manager.Queue()  # fasta queue
    pool = mp.Pool(n_cores) # pool for multiprocessing
    watcher = pool.apply_async(listener_writer, (q, output_file))
    fasta_file = os.path.join("/".join(output_file.split('/')[:-1]), 'read_references_benchmark.fasta')
    watcher_fasta = pool.apply_async(listener_writer, (p, fasta_file))
    
    print('Reading reference')
    references = read.read_fasta(reference_file)
    
    print('Resquiggleling reads in: ' + fast5_path)
    
    processed_reads = list()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                processed_reads.append(line.split('\t')[0])
    else:
        with open(output_file, 'w') as f:
            s = 'file'+'\t'+'read_id'+'\t'+'result'+'\t'+'align_st'+'\t'+'align_nd'+'\t'+'align_strand'+'\t'+'align_chrom'+'\n'
            f.write(s)
    processed_reads = set(processed_reads)
    
    if len(processed_reads) > 0:
        print('Reads already processed: ' + str(len(processed_reads)))
    
    print('Reading genome file: ' + str(genome_file))
    aligner =  mappy.Aligner(genome_file, preset=str('map-ont'), best_n=1)
    
    print('Mapping')
    jobs = list()
    file_list = glob.glob(fast5_path + '/**/*.fast5', recursive=True)
    for file_path in tqdm(file_list, disable = not verbose):
        if file_path in processed_reads:
            continue
        
        file_name = file_path.split('/')[-1].split('.')[0]
        try:
            ref = references[file_name]
        except KeyError:
            s = file_path + '\t' + 'unknown' + '\t' + 'No_reference' + '\t' + 'NaN' + '\t' + 'NaN' + '\t' + 'NaN' + '\t' + 'NaN'
            q.put(s)
            continue

        try:
            alignment = list(aligner.map(str(ref)))[0]
        except IndexError:
            s = file_path + '\t' + 'unknown' + '\t' + 'Failed_to_align' + '\t' + 'NaN' + '\t' + 'NaN' + '\t' + 'NaN' + '\t' + 'NaN'
            q.put(s)
            continue

        
        alignment_results = {"r_st":alignment.r_st, 
                             "r_en":alignment.r_en,
                             "strand":alignment.strand,
                             "ctg":alignment.ctg}

        job = pool.apply_async(segment_read, (file_path, ref, alignment_results, q, p))
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
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast5-path", type=str, help='Path to fast5 files, it is searched recursively')
    parser.add_argument("--reference-file", type=str, help='Fasta file that contains references')
    parser.add_argument("--genome-file", type=str, help='Fasta file with genome sequence')
    parser.add_argument("--output-file", type=str, help='Text file that contains information on the result of the segmentation')
    parser.add_argument("--n-cores", type=int, help='Number of processes')
    parser.add_argument("--verbose", action='store_true', help='Output a progress bar')
    
    args = parser.parse_args()
    
    main(fast5_path = args.fast5_path, 
         reference_file = args.reference_file, 
         genome_file = args.genome_file,
         output_file = args.output_file, 
         n_cores = args.n_cores, 
         verbose = args.verbose)
    
    
    
    
    