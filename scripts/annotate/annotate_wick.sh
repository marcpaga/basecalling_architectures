#!/bin/bash

wick_dir=$1
n_cores=$2

for d in "$1"/* ; 
do    
    spe=$d
    ref="${spe}/read_references.fasta"
    out="${spe}/segmentation_report.txt"
    
    echo "Processing ${spe}"
    python annotate_wick.py --fast5-path $spe --reference-file $ref --output-file $out --n-cores $2 --verbose
    

done