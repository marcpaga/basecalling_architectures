"""
Here the seeds used for the random steps are defined to ensure reproducibility
And also the amount of reads for each task
"""

### INTER-SPECIES TASK 

# species selection
INTER_START_SEED = 1
INTER_INNER_SEED = 1

# read selection
INTER_READ_SEED = 1

# number of reads 
MAX_SPECIES = 12
MIN_SPECIES = 10
INTER_TRAIN = 50000
INTER_TRAIN_TEST = 5000
INTER_TEST = 20000

### GLOBAL TASK

# reads selection
GLOBAL_SEED = 1

# odd chromosomes for training in human
GLOBAL_TRAIN_ODD = 1

# number of reads
GLOBAL_TRAIN = 100000
GLOBAL_TEST = 50000

### HUMAN TASK
HUMAN_SEED = 1
HUMAN_TRAIN_ODD = 1

# number of reads
HUMAN_TRAIN = 50000
HUMAN_TEST = 25000

### DATA PREPARE 
DATA_PREPARE_READ_SHUFFLE = 1
GENOME_SPLIT = 0.5



