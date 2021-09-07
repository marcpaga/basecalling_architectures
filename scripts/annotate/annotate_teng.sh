#!/bin/bash

# Usage
# bash annotate_teng.sh $TENG_DIR $N_CORES

echo 'Processing Lambda phage data'
echo 'Deleting nanoraw segmentation tables'
python annotate_teng.py --fast5-path "${1}/Lambda_phage/pass" --n-cores $2 --delete --verbose

echo 'Resquiggleling'
tombo resquiggle "${1}/Lambda_phage/pass" "${1}/genomes/Lambda_phage.fna" --processes $2 --dna --num-most-common-errors 5 --overwrite

echo 'Reporting results'
python annotate_teng.py --fast5-path "${1}/Lambda_phage/pass" --output-file "${1}/Lambda_phage/segmentation_report.txt" --n-cores $2 --report --verbose

echo 'Processing Escherichia coli data'
echo 'Deleting nanoraw segmentation tables'
python annotate_teng.py --fast5-path "${1}/Escherichia_coli/pass" --n-cores $2 --delete --verbose

echo 'Resquiggleling'
tombo resquiggle "${1}/Escherichia_coli/pass" "${1}/genomes/Escherichia_coli.fna" --processes $2 --dna --num-most-common-errors 5 --overwrite

echo 'Reporting results'
python annotate_teng.py --fast5-path "${1}/Escherichia_coli/pass" --output-file "${1}/Escherichia_coli/segmentation_report.txt" --n-cores $2 --report --verbose