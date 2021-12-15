#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --job-name=dataprep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.pagesgallego@umcutrecht.nl

source /home/cog/mpages/.bashrc
source activate babe


python /hpc/compgen/users/mpages/babe/scripts/nn/data_prepare_numpy.py \
--fast5-list  /hpc/compgen/users/mpages/babe/doc/splits/human_task_train_reads.txt \
--output-dir  /hpc/compgen/projects/nanoxog/babe/analysis/mpages/train_input/human/4000.0 \
--total-files  100 \
--window-size 4000 \
--window-slide 0
