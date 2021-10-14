#!/bin/bash

# Usage
# bash annotate_jain.sh $JAIN_DIR $N_CORES

main_dir=$1
n_cores=$2

array=($main_dir/*/) # end with / to list only dirs

for dir in ${array[@]}; do
    
    fname=$(basename $dir) # get deepest folder name
    dir=${dir%?}
    # skip genomes and tmp folder as we do not need them
    if [ "$fname" == "genomes" ]; then
        continue
    fi
    if [ "$fname" == "tmp" ]; then
        continue
    fi

    echo "Processing files in ${dir}"
    # untar all the fastq files and join them into a single fastq file
   
    if [ ! -f "${dir}/fastq/all_basecalls.fastq" ]
    then
        echo "  Decompressing fastq"
        find "${dir}/fastq" -name '*.fastq.gz' -exec gzip -d '{}' \;
        find "${dir}/fastq" -name '*.fastq' -exec mv {} "${dir}/fastq" \;

        echo "  Joining fastq"
        cat "${dir}/fastq/"*".fastq" > "${dir}/fastq/all_basecalls.fasta"
        rm "${dir}/fastq/"*".fastq"
        rm -r "${dir}/fastq/Norwich"
        rm -r "${dir}/fastq/Bham"
        mv "${dir}/fastq/all_basecalls.fasta" "${dir}/fastq/all_basecalls.fastq"
    fi

    if [ ! -f "${dir}/segmentation_report.txt" ]
    then

        if [ ! -f "${dir}/fast5/filename_mapping.txt" ]
        then
            echo "  Spliting fast5"
            multi_to_single_fast5 -i "${dir}/fast5" -s "${dir}/fast5" --recursive -t $2
        fi

        echo "  Deleting multifast5 files"
        rm -r "${dir}/fast5/Norwich"
        rm -r "${dir}/fast5/Bham"

        if [ ! -f "${dir}/fast5/annotated.txt" ]
        then
            echo "  Merging fast5 and fastq"
            tombo preprocess annotate_raw_with_fastqs --fast5-basedir "${dir}/fast5" --fastq-filenames "${dir}/fastq/all_basecalls.fastq" --overwrite --processes $n_cores
            echo "True" >> "${dir}/fast5/annotated.txt"
        fi

        if [ ! -f "${dir}/fast5/resquiggled.txt" ]
        then
            echo "  Resquiggleling"
            tombo resquiggle "${dir}/fast5" "${main_dir}/genomes/Homo_sapiens.fna" --processes $n_cores --dna --num-most-common-errors 5 --ignore-read-locks
            echo "True" >> "${dir}/fast5/resquiggled.txt"
        fi

        echo '  Reporting results'
        python annotate_jain.py --fast5-path "${dir}/fast5" --output-file "${dir}/segmentation_report.txt" --n-cores $n_cores --report --verbose
    fi


done
