#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --job-name=evaluate
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.pagesgallego@umcutrecht.nl

source /home/cog/mpages/.bashrc
source activate babe


python /hpc/compgen/users/mpages/babe/scripts/benchmark/evaluate_basecalls.py \
--basecalls-path /hpc/compgen/projects/nanoxog/raw/benchmark/jain_benchmark/Homo_sapiens_FAB42828/fastq \
--references-path /hpc/compgen/projects/nanoxog/raw/benchmark/jain_benchmark/Homo_sapiens_FAB42828 \
--output-file /hpc/compgen/users/mpages/babe/eval1.csv \
--processes 32
