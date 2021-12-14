#!/bin/bash

# Usage
# bash annotate_verm.sh $VERM_DIR $N_CORES

main_dir=$1
n_cores=$2

tmp_dir="${main_dir}/Lambda_phage/tmp"
fast5_dir="${main_dir}/Lambda_phage/fast5"
fastq_dir="${main_dir}/Lambda_phage/fastq"

# join all basecalls into a single file
if [ ! -f "${fastq_dir}/all_basecalls.fastq" ]
then
    echo "Joining basecalls"
    cat "${tmp_dir}/VER5940/fastq_pass/"*".fastq" > "${fastq_dir}/all_basecalls.fastq"
fi

if [ ! -f "${main_dir}/segmentation_report.txt" ]
    then

    if [ ! -f "${fast5_dir}/filename_mapping.txt" ]
    then
        echo "  Spliting fast5"
        multi_to_single_fast5 -i "${tmp_dir}/VER5940/fast5_pass/" -s ${fast5_dir} --recursive -t $2
    fi

    if [ ! -f "${fast5_dir}/annotated.txt" ]
    then
        echo "  Merging fast5 and fastq"
        tombo preprocess annotate_raw_with_fastqs --fast5-basedir ${fast5_dir} --fastq-filenames "${fastq_dir}/all_basecalls.fastq" --overwrite --processes $n_cores
        echo "True" >> "${fast5_dir}/annotated.txt"
    fi

    if [ ! -f "${fast5_dir}/resquiggled.txt" ]
    then
        echo "  Resquiggleling"
        tombo resquiggle ${fast5_dir} "${main_dir}/genomes/Lambda_phage.fna" --processes $n_cores --dna --num-most-common-errors 5 --ignore-read-locks
        echo "True" >> "${fast5_dir}/resquiggled.txt"
    fi

    echo '  Reporting results'
    python annotate_jain.py --fast5-path ${fast5_dir} --output-file "${main_dir}/segmentation_report.txt" --n-cores $n_cores --report --verbose
fi


