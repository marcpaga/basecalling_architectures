#!/bin/bash

wick_dir=$1
spe_test_file=$2
n_cores=$3

while read a;
do    

    spe=$a
    echo $spe
    
    ref="${wick_dir}/genomes/${spe}.fna"
    out="${wick_dir}/${spe}/segmentation_report.txt"
    fastq_file="${wick_dir}/${spe}/fastq/basecalls.fastq"

    if [ ! -f "${wick_dir}/${spe}/fast5/annotated.txt" ]
    then
        echo 'Annotating reads with basecalls'
        tombo preprocess annotate_raw_with_fastqs --fast5-basedir "${wick_dir}/${spe}/fast5" --fastq-filenames $fastq_file --processes $3 --overwrite
        echo "True" >> "${wick_dir}/${spe}/fast5/annotated.txt"
    fi

    if [ ! -f "${wick_dir}/${spe}/fast5/resquiggled.txt" ]
    then
        echo 'Resquiggleling'
        tombo resquiggle "${wick_dir}/${spe}/fast5" $ref --processes $3 --dna --num-most-common-errors 5 --overwrite
        echo "True" >> "${wick_dir}/${spe}/fast5/resquiggled.txt"
    fi

    if [ ! -f $out ]
    then
        echo 'Reporting results'
        python annotate_jain.py --fast5-path "${wick_dir}/${spe}/fast5" --output-file $out --n-cores $3 --report --verbose
    fi
    
done < $spe_test_file