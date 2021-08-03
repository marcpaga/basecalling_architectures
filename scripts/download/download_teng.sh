#!/bin/bash

output_dir=$1

ecoli_fast5_link="https://data.genomicsresearch.org/Projects/train_set_all/S10/pass.tar.gz"
ecoli_ref_link="https://data.genomicsresearch.org/Projects/train_set_all/S10/Reference/reference.fasta"
lambda_fast5_link="https://data.genomicsresearch.org/Projects/train_set_all/Lambda_old/pass.tar.gz"
lambda_ref_link="https://data.genomicsresearch.org/Projects/train_set_all/Lambda_old/Reference/lambda.fasta"


wget $ecoli_fast5_link -O "${output_dir}/ecoli_fast5.tar.gz" --no-check-certificate
wget $ecoli_ref_link -O "${output_dir}/ecoli_genome.fasta" --no-check-certificate

mkdir "${output_dir}/Escherichia_coli"
tar -xvzf "${output_dir}/ecoli_fast5.tar.gz" -C "${output_dir}/Escherichia_coli"

wget $lambda_fast5_link -O "${output_dir}/lambda_fast5.tar.gz" --no-check-certificate
wget $lambda_ref_link -O "${output_dir}/lambda_genome.fasta" --no-check-certificate

mkdir "${output_dir}/Lambda_phage"
tar -xvzf "${output_dir}/lambda_fast5.tar.gz" -C "${output_dir}/Lambda_phage"


