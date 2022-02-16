#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --job-name=evaluate
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.pagesgallego@umcutrecht.nl

source /home/cog/mpages/.bashrc
source activate babe

python /hpc/compgen/users/mpages/babe/scripts/benchmark/evaluate_basecalls.py \
--basecalls-path /hpc/compgen/projects/nanoxog/babe/analysis/mpages/models/grid_analysis/human/catcaller_sacall_ctc_True_2000 \
--references-path /hpc/compgen/projects/nanoxog/raw/benchmark/all_read_references.fasta \
--depth 2 \
--processes 32
