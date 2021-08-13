#!/bin/bash

wick_dir=$1
spe_train_file=$2
n_cores=$3

while read a;
do    
    spe=$a
    
    ref="${spe}/read_references.fasta"
    out="${spe}/segmentation_report.txt"
    
    if [ -f $out ]; then
        echo 'Skipping'
    else
        IFS='/' read -r -a spename <<< "${spe}"
        spename="${spename[-1]}"

        IFS='_' read -r -a array <<< "${spename}"
        genome_file="${wick_dir}/genomes/${array[0]}_${array[1]}.fna"

        echo "Processing ${spe}"
        python annotate_wick.py --fast5-path $spe --genome-file $genome_file --reference-file $ref --output-file $out --n-cores $2 --verbose
    fi
    
done < $spe_train_file