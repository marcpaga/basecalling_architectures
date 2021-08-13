#!/bin/bash

wick_dir=$1
spe_test_file=$2
n_cores=$3

while read a;
do    
    spe=$a
    
    ref="${wick_dir}/genomes_test/${spe}.fna"
    out="${wick_dir}/${spe}/segmentation_report.txt"
    fastq_file="${wick_dir}/basecalls_test/${spe}/01_guppy_v2.1.3.fastq"
    
    if [ -f $out ]; then
        echo 'Skipping'
    else
        echo 'Moving fast5 files to reads folder'
        mkdir "${wick_dir}/${spe}/reads"
        mv "${wick_dir}/${spe}/"*".fast5" "${wick_dir}/${spe}/reads"

        echo 'Annotating reads with basecalls'
        tombo preprocess annotate_raw_with_fastqs --fast5-basedir "${wick_dir}/${spe}/reads" --fastq-filenames $fastq_file --processes $3 --overwrite

        echo 'Resquiggleling'
        tombo resquiggle "${wick_dir}/${spe}/reads" $ref --processes $3 --dna --num-most-common-errors 5 --overwrite

        echo 'Reporting results'
        python annotate_teng.py --fast5-path "${wick_dir}/${spe}/reads" --output-file $out --n-cores $3 --report --verbose
    fi
    
done < $spe_test_file