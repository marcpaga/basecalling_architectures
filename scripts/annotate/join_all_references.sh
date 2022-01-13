#!/bin/bash

# Usage
# bash join_all_references.sh $DATA_DIR
# where $DATA_DIR has the following structure
# jain/experiments/read_references_benchmark.fasta
# verm/experiments/read_references_benchmark.fasta
# wick/experiments/read_references_benchmark.fasta

data_dir=$1

find $1 -maxdepth 3 -type f -name read_references_benchmark.fasta -exec cat {} + > "${data_dir}/all_read_references.fasta"
