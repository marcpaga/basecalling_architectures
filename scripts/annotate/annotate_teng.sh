#!/bin/bash

echo 'Resquiggleling Lambda phage data'
tombo resquiggle "${1}/Lambda_phage/pass" "${1}/lambda_genome.fasta" --processes $2 --dna --num-most-common-errors 5 --overwrite

echo 'Resquiggleling Escherichia coli data'
tombo resquiggle "${1}/Escherichia_coli/pass" "${1}/ecoli_genome.fasta" --processes $2 --dna --num-most-common-errors 5 --overwrite
