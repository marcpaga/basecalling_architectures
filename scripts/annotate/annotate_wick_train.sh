#!/bin/bash

wick_dir=$1
spe_train_file=$2
n_cores=$3

while read a;
do    
    spe=$a
    
    ref="${wick_dir}/${spe}/read_references.fasta"
    out="${wick_dir}/${spe}/segmentation_report.txt"
    
    if [ -f $out ]; then
        echo 'Skipping'
    else
        IFS='/' read -r -a spename <<< "${spe}"
        spename="${spename[-1]}"

        IFS='-' read -r -a array <<< "${spename}"
        genome_file="${wick_dir}/genomes/${array[0]}.fna"

        echo "Processing ${spe}"
        python annotate_wick.py --fast5-path "${wick_dir}/$spe" --genome-file $genome_file --reference-file $ref --output-file $out --n-cores $n_cores --verbose
    fi

    
done < $spe_train_file