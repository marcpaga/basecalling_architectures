# Data download and annotation

After the data download and annotation is done the dir should look like this:

```
benchmark
 │
 ├─ jain
 │   ├─ genomes
 │   │   └─ Homo_sapiens.fna                    # human NA12878 genome
 │   └─ species                                 # 3 dirs, different sequencing runs
 │       ├─ fast5                               # Dir with .fast5 files
 │       ├─ fastq                               # Dir with .fastq files
 │       ├─ read_references_benchmark.fasta     # Reference for each read
 │       └─ segmentation_report.txt       # Results on the segmentation
 │
 ├─ wick
 │   ├─ genomes
 │   │   └─ species.fna                         # 36 .fna files with genomes
 │   └─ species                                 
 │       ├─ fast5                               # Dir with .fast5 files
 │       ├─ fastq                               # Dir with .fastq files
 │       ├─ read_references_benchmark.fasta     # Reference for each │ead
 │       └─ segmentation_report.txt             # Results on the segmentation
 │
 └- verm
    ├─ genomes
    │   └─ species.fna 
    └─ Lambda_phage 
        ├─ fast5                               # Dir with .fast5 files
        ├─ fastq                               # Dir with .fastq files
        ├─ read_references_benchmark.fasta     # Reference for each │ead
        └─ segmentation_report.txt             # Results on the 
```

## Download 

Download the data using the following script: `scripts/download/download_all.sh OUTPUT_DIR`.

All the data will be downloaded and decompressed in the `OUTPUT_DIR` directory. Inside `OUTPUT_DIR` there will be three subdirectories: `jain`, `verm` and `wick`. Inside each of these there will be several folders with the different datasets, and also a `tmp` folder. The `tmp` folder contains the compressed downloaded files and can be deleted at the user discretion. It can be useful however to keep it in case the processed data is lost, then it is not necessary to download the data again.

## Annotate

Scripts to annotate can be found in the `scripts/annotate` folder. These use tombo to align the raw data to the DNA sequence. To be able to run these scripts several dependencies must be installed. A conda environment file is provided in `envs/annotate.yml`.

## Dataset peculiarities

### Jain

The downloaded raw data is in multi-fast5 format. This format is not compatible with the `tombo` tool to annotate the raw data with their sequence. For this, first the data files must be split to single-fast5 files, then annotated with their respective basecalls and finally resquiggled.

### Wick

This dataset is already divided into train and test sets. The data of these two sets is organized differently. For this reason, the links files are divided into train and test. The train data is already annotated, therefore we know which sequence belongs each read. However, for the test data we do not know that, and we must also download the basecalls and genomes to annotate the data with `tombo`.

## Delete data

Not all of the data is necessary and some has been duplicated, besides the already mentioned tar files, there are other files that can be deleted as they are not used at the user discretion.

- `OUTPUT_DIR_WICK/*/sloika_hdf5s`: here there are many .hdf5 files that can be deleted
- `OUTPUT_DIR_JAIN/*/fastq/*.fastq`: all the fastq files have been combined to `all_basecalls.fastq`, the other fastq files can be deleted (KEEP `all_basecalls.fastq`)
- `OUTPUT_DIR_JAIN/*/fastq/{Norwich|Bham}`: these folders can be deleted.


