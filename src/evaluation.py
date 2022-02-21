import re
import numpy as np
from scipy.stats import mannwhitneyu

from constants import BASES, GLOBAL_ALIGN_FUNCTION, LOCAL_ALIGN_FUNCTION, MATRIX, ALIGNMENT_GAP_OPEN_PENALTY, ALIGNMENT_GAP_EXTEND_PENALTY
from utils import find_runs

REPORT_COLUMNS = ['read_id', # id of the read
                    'len_reference', # length of the reference
                    'len_basecalls', # length of the basecalls
                    'que_start',
                    'que_end',
                    'ref_start', # number of insertions/deletions at the start of the alignment
                    'ref_end', # number of insertions/deletions at the end of the alignment
                    'comment'] 

ERRORS = list()
for b1 in BASES:
    for b2 in BASES + ['-']:
        for b3 in BASES + ['-']:
            for b4 in BASES:
                if b2 == '-' and b3 == '-':
                    continue
                ERRORS.append(b1 + b2 + '>' + b3 + b4)
REPORT_COLUMNS += ERRORS

for b in BASES:
    REPORT_COLUMNS.append('homo_'+b+'_counts')
    REPORT_COLUMNS.append('homo_'+b+'_errors')

REPORT_COLUMNS.append('phred_error_test')
REPORT_COLUMNS.append('phred_mismatch_test')
REPORT_COLUMNS.append('phred_insertion_test')
REPORT_COLUMNS.append('phred_mean')
REPORT_COLUMNS.append('phred_median')
REPORT_COLUMNS.append('phred_std')
REPORT_COLUMNS.append('phred_mean_correct')
REPORT_COLUMNS.append('phred_median_correct')
REPORT_COLUMNS.append('phred_std_correct')
REPORT_COLUMNS.append('phred_mean_error')
REPORT_COLUMNS.append('phred_median_error')
REPORT_COLUMNS.append('phred_std_error')

def align(que, ref, local = True):
    """Wrapper function to align two sequences
    """
    if local:
        return LOCAL_ALIGN_FUNCTION(que, ref, ALIGNMENT_GAP_OPEN_PENALTY, ALIGNMENT_GAP_EXTEND_PENALTY, MATRIX)
    else:
        return GLOBAL_ALIGN_FUNCTION(que, ref, ALIGNMENT_GAP_OPEN_PENALTY, ALIGNMENT_GAP_EXTEND_PENALTY, MATRIX)

def elongate_cigar(short_cigar):
    cigar_counts = re.split('H|X|=|I|D|N|S|P|M', short_cigar)
    cigar_strs = re.split('[0-9]', short_cigar)
    
    cigar_counts = [c for c in cigar_counts if c != '']
    cigar_strs = [c for c in cigar_strs if c != '']
    
    assert len(cigar_strs) == len(cigar_counts)
    
    longcigar = ''
    for c, s in zip(cigar_counts, cigar_strs):
        longcigar += s*int(c)
    return longcigar, cigar_counts, cigar_strs

def shorten_cigar(long_cigar):

    for i, l in enumerate(long_cigar):
        if i == 0:
            short_cigar = ''
            c = 1
            prev_l = l
            continue
        
        if l == prev_l:
            c += 1
        else:
            short_cigar += str(c) + prev_l
            c = 1
            prev_l = l

    short_cigar += str(c) + prev_l

    return short_cigar

def make_align_arr(long_cigar, truth_seq, pred_seq, phredq = None, phredq_ref = None):
    """Makes an alignment array based on the long cigar
    
    Args:
        long_cigar (str): output from `elongate_cigar`
        truth_seq (str): sequence 1
        pred_seq (str): sequence 2
        phredq (str): quality scores for the predicted sequence
        phredq_ref (str): quality scores for the reference

    Returns:
        A np:array of shape [5, alignment_length]. The first dimensions are the
        reference, alignment chars, predicted sequence, phredq and phredq_ref if given.
    """

    if phredq is not None:
        if len(pred_seq) != len(phredq):
            raise ValueError('pred_seq ({}) and phredq ({}) lenghts are different'.format(len(pred_seq), len(phredq)))
    
    tc = 0
    pc = 0
    align_arr = np.full((5, len(long_cigar)), '')
    for i, c in enumerate(long_cigar):
        if c == 'D':
            align_arr[0, i] = truth_seq[tc]
            align_arr[1, i] = ' '
            align_arr[2, i] = '-'
            if phredq is not None:
                align_arr[3, i] = ' '
            if phredq_ref is not None:
                align_arr[4, i] = phredq_ref[tc]

            tc += 1
        elif c == 'I':
            align_arr[0, i] = '-'
            align_arr[1, i] = ' '
            align_arr[2, i] = pred_seq[pc]
            if phredq is not None:
                align_arr[3, i] = phredq[pc]
            if phredq_ref is not None:
                align_arr[4, i] = ' '

            pc += 1
        elif c == 'X':
            align_arr[0, i] = truth_seq[tc]
            align_arr[1, i] = '.'
            align_arr[2, i] = pred_seq[pc]
            if phredq is not None:
                align_arr[3, i] = phredq[pc]
            if phredq_ref is not None:
                align_arr[4, i] = phredq_ref[tc]

            pc += 1
            tc += 1
        elif c == '=':
            align_arr[0, i] = truth_seq[tc]
            align_arr[1, i] = '|'
            align_arr[2, i] = pred_seq[pc]
            if phredq is not None:
                align_arr[3, i] = phredq[pc]
            if phredq_ref is not None:
                align_arr[4, i] = phredq_ref[tc]

            pc += 1
            tc += 1
        elif c == 'M':
            align_arr[0, i] = truth_seq[tc]
            align_arr[2, i] = pred_seq[pc]
            if truth_seq[tc] == pred_seq[pc]:
                align_arr[1, i] = '|'
            else:
                align_arr[1, i] = '.'
            if phredq is not None:
                align_arr[3, i] = phredq[pc]
            if phredq_ref is not None:
                align_arr[4, i] = phredq_ref[tc]

            pc += 1
            tc += 1
            
    return align_arr

def count_longcigar_patches(long_cigar, local_st, local_nd):
    
    # calculate the length of the patches of each case
    prev_m = ''
    err_lens = {'=':list(), 'X':list(), 'D':list(), 'I':list()}
    c = 1
    local_cigar = long_cigar[local_st:local_nd]
    for i in range(len(local_cigar)):
        m = local_cigar[i]
        if m == prev_m:
            c += 1
        else:
            try:
                err_lens[prev_m].append(c)
            except KeyError:
                pass
            c = 1
            prev_m = m
            
    return err_lens

def eval_pair(ref, que, read_id, phredq = None, align_method = 'parasail'):
    """Align two sequences and evaluate the alignment
    
    Args:
        ref (str): reference sequence or aligner if using minimap2
        que (str): predicted sequence
        read_id (str): uuid of the read
        phredq (str): string with predq symbols
        align_method (str): whether to do alignment using 'parasail' or 'minimap2'
        
    Returns:
        results (dict): dictionary with metrics and long confusion matrix
    """

    # for the per read report, these are the columns at the top of the csv file
    # for the AA, AC... the first letter indicates reference and the second the basecall
    # therefore different letters indicate missmatches (AC is predicted a C but truth was an A)
    # - indicates insertions if on the left and deletions if on the right

    # mutational signature annotation
    # the first and lat bases are the context, the middle part
    # indicates from which base to which base the error
    # AA>CA mismatch
    # AA>-A deletion
    # A->AA insertion

    result = dict()
    for k in REPORT_COLUMNS:
        result[k] = None
    result['read_id'] = read_id

    
    if align_method == 'parasail':
        # align the two sequences
        alignment = align(que, ref, local = True)
        decoded_cigar = alignment.cigar.decode.decode()
        long_cigar, _, _ = elongate_cigar(decoded_cigar)
        
        que_st = alignment.cigar.beg_query
        que_nd = alignment.end_query
        ref_st = alignment.cigar.beg_ref
        ref_nd = alignment.end_ref

        # calculate the local ends
        local = np.where((np.array(list(long_cigar)) == 'X') | (np.array(list(long_cigar)) == 'M'))[0]
        local_st = local[0]
        local_nd = local[-1]
        decoded_cigar = decoded_cigar[local_st:local_nd+1]

        # create a 2D array with the alignment
        alignment_arr = make_align_arr(long_cigar, ref, que, phredq = phredq)
        # +1 to local_nd because the calculation gives the last position, so we 
        # need one more for slicing
        local_arr = alignment_arr[:, local_st:local_nd+1]
        len_ref = len(ref)
        len_que = len(que)

    elif align_method == 'minimap2':

        len_que = len(que)
        if len_que == 0:
            result['len_basecalls'] = len_que
            result['comment'] = 'no prediction'
            return result

        correct_match = False
        for alignment in ref.map(seq = que):
            if read_id == alignment.ctg:
                correct_match = True
                break

        if not correct_match:
            result['comment'] = 'failed mapping'
            return result
        
        que_st = alignment.q_st
        que_nd = alignment.q_en + 1
        ref_st = alignment.r_st
        ref_nd = alignment.r_en + 1
        len_ref = len(ref.seq(read_id))
        

        long_cigar, _, _ = elongate_cigar(alignment.cigar_str)

        if phredq is not None:
            phredq = phredq[que_st:que_nd]

        local_arr = make_align_arr(
            long_cigar = long_cigar, 
            truth_seq = ref.seq(read_id)[ref_st:ref_nd], 
            pred_seq = que[que_st:que_nd], 
            phredq = phredq
        )

        longcigar_arr = np.array(list(long_cigar))
        longcigar_arr[np.where(local_arr[1, :] == '.')[0]] = 'X'
        longcigar_fixed = "".join(longcigar_arr.tolist())

        local = np.where((np.array(list(longcigar_fixed)) == 'X') | (np.array(list(longcigar_fixed)) == 'M'))[0]
        local_st = local[0]

        local_nd = local[-1]
        longcigar_fixed = longcigar_fixed[local_st:local_nd+1]

        que_st += local_st
        que_nd -= local_nd
        ref_st += local_st
        ref_nd -= local_nd
        local_arr = local_arr[:, local_st:local_nd+1]

        decoded_cigar = shorten_cigar(longcigar_fixed)

    else:
        raise ValueError()
    
    result['len_reference'] = len_ref
    result['len_basecalls'] = len_que
    result['ref_start'] = ref_st
    result['ref_end'] = ref_nd
    result['que_start'] = que_st
    result['que_end'] = que_nd
    
    signatures = count_signatures(local_arr)
    result = {**result, **signatures}
    
    # count for each base the amount of bases in homopolymers
    # and how many errors in these regions
    homo_counts = dict()
    homo_errors = dict()
    for b in BASES:
        homo_counts[b] = 0
        homo_errors[b] = 0

    ref_arr = local_arr[0, :]
    for b in BASES:
        base_or_gap = (ref_arr == b) | (ref_arr == '-')
        sections = find_runs(base_or_gap)
        for t, st, l in zip(*sections):
            if not t:
                continue
            if l < 5:
                continue
            if np.sum(local_arr[0, st:st+l] == b) < 5:
                continue
            h_arr = local_arr[:, st:st+l]
            for j in range(h_arr.shape[1]):
                if h_arr[0, j] == '-' and h_arr[2, j] == b:
                    homo_errors[b] += 1
                elif h_arr[0, j] == b:
                    if h_arr[2, j] == b:
                        homo_counts[b] += 1
                    else:
                        homo_counts[b] += 1
                        homo_errors[b] += 1
    
    for b in BASES:
        result['homo_'+b+'_counts'] = homo_counts[b]
        result['homo_'+b+'_errors'] = homo_errors[b]

    # calculate significance of phredq scores with correctness
    if phredq is not None:
        correct = list()
        error = list()
        mismatch = list()
        insertion = list()

        for i in range(local_arr.shape[1]):

            phred_symbol = local_arr[3, i]
            align_symbol = local_arr[1, i]
            
            if phred_symbol == ' ':
                continue
            
            score = ord(phred_symbol) - 33
            if align_symbol == '|':
                correct.append(score)
            elif align_symbol == '.':
                error.append(score)
                mismatch.append(score)
            elif align_symbol == ' ':
                error.append(score)
                insertion.append(score)

        try:
            result['phred_error_test'] = mannwhitneyu(x = correct, y = error, alternative = 'greater').pvalue
        except:
            result['phred_error_test'] = -1
        try:
            result['phred_mismatch_test'] = mannwhitneyu(x = correct, y = mismatch, alternative = 'greater').pvalue
        except: 
            result['phred_mismatch_test'] = -1
        try:
            result['phred_insertion_test'] = mannwhitneyu(x = correct, y = insertion, alternative = 'greater').pvalue
        except:
            result['phred_insertion_test'] = -1

        result['phred_mean'] = np.mean(correct + error)
        result['phred_median'] = np.median(correct + error)
        result['phred_std'] = np.std(correct + error)
        result['phred_mean_correct'] = np.mean(correct)
        result['phred_median_correct'] = np.median(correct)
        result['phred_std_correct'] = np.std(correct)
        result['phred_mean_error'] = np.mean(error)
        result['phred_median_error'] = np.median(error)
        result['phred_std_error'] = np.std(error)


    result['comment'] = 'pass'

    return result

def count_signatures(arr):
    """Counts the different signatures in a local alingment array

    Args:
        arr (np.array): array with the alignment

    Returns:
        A dictionary with signatures as keys and counts as values
    """

    if arr[0, 0] == '-' or arr[0, -1] == '-':
        raise ValueError('The reference must start and end with bases, not insertions')

    # calculate the mutational signature style errors
    mut_dict = dict()
    for e in ERRORS:
        mut_dict[e] = 0

    # we iterate over the positions for which we can calculate a signature,
    # which are all but the first and last bases
    # for each of these positions we look for the closest base in the predictions
    # on the left side and right side
    # then we get which code should be based on the chunk of the array that 
    # we have
    r = np.array(arr[2, :])
    nogaps = r != '-'
    pos = np.arange(0, len(nogaps), 1)

    for i in np.arange(1, len(r) - 1, 1):
        st = pos[:i][np.where(nogaps[:i])[0]][-1]
        nd = pos[i+1:][np.where(nogaps[i+1:])[0]][0]

        code = arr[2, st] + arr[2, i] + '>' + arr[0, i] + arr[2, nd]

        mut_dict[code] += 1
    
    return mut_dict

def alignment_accuracy(y, p, alignment_function = GLOBAL_ALIGN_FUNCTION, matrix = MATRIX, 
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