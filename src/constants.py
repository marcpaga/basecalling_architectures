"""Contains constants used 
"""

import parasail

ALL_SPECIES = [ 'Stenotrophomonas_pavanii',
                'Klebsiella_pneumoniae',
                'Salmonella_enterica',
                'Pseudomonas_aeruginosa',
                'Morganella_morganii',
                'Moraxella_lincolnii',
                'Klebsiella_variicola',
                'Klebsiella_quasipneumoniae',
                'Haemophilus_parainfluenzae',
                'Escherichia_marmotae',
                'Escherichia_coli',
                'Enterobacter_kobei',
                'Comamonas_kerstersii',
                'Citrobacter_koseri',
                'Citrobacter_freundii',
                'Burkholderia_cenocepacia',
                'Acinetobacter_ursingii',
                'Acinetobacter_nosocomialis',
                'Klebsiella_aerogenes',
                'Acinetobacter_baumannii',
                'Staphylococcus_aureus',
                'Shigella_sonnei',
                'Serratia_marcescens',
                'Haemophilus_haemolyticus',
                'Acinetobacter_pittii',
                'Stenotrophomonas_maltophilia',
                'Lambda_phage',
                'Homo_sapiens' ]

HUMAN_CHROMOSOMES = ['chr1',
                     'chr2',
                     'chr3',
                     'chr4',
                     'chr5',
                     'chr6',
                     'chr7',
                     'chr8',
                     'chr9',
                     'chr10',
                     'chr11', 
                     'chr12',
                     'chr13',
                     'chr14',
                     'chr15',
                     'chr16',
                     'chr17',
                     'chr18',
                     'chr19',
                     'chr20',
                     'chr21',
                     'chr22',
                     'chrX',
                     'chrY',
                     'chrM']

HUMAN_CHROMOSOMES_EVEN = ['chr2',
                          'chr4',
                          'chr6',
                          'chr8',
                          'chr10',
                          'chr12',
                          'chr14',
                          'chr16',
                          'chr18',
                          'chr20',
                          'chr22']
HUMAN_CHROMOSOMES_ODD = ['chr1',
                         'chr3',
                         'chr5',
                         'chr7',
                         'chr9',
                         'chr11',
                         'chr13',
                         'chr15',
                         'chr17',
                         'chr19',
                         'chr21']

HUMAN_CHROMOSOMES_OTHER = ['chrX', 'chrY', 'chrM']

BASES = ['A', 'C', 'G', 'T']
BASES_CRF = 'N' + ''.join(BASES)

# parasail alignment configuration
ALIGNMENT_GAP_OPEN_PENALTY = 8
ALIGNMENT_GAP_EXTEND_PENALTY = 4
ALIGNMENT_MATCH_SCORE = 2
ALIGNMENT_MISSMATCH_SCORE = 1

# networks
CTC_BLANK = 0
CRF_STATE_LEN = 4
CRF_BIAS = True
CRF_SCALE = 5.0
CRF_BLANK_SCORE = 2.0
CRF_N_BASE = len(BASES)

# evalution
GLOBAL_ALIGN_FUNCTION = parasail.nw_trace_striped_32
LOCAL_ALIGN_FUNCTION = parasail.sw_trace_striped_32
MATRIX = parasail.matrix_create("".join(BASES), ALIGNMENT_MATCH_SCORE, -ALIGNMENT_MISSMATCH_SCORE)