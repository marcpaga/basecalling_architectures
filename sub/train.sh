#!/bin/bash

#SBATCH --time=5-00:00:00
#SBATCH -c 4
#SBATCH --partition gpu
#SBATCH --gpus-per-node=RTX6000:1
#SBATCH --mem=32G
#SBATCH --job-name=trainhalcyon
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.pagesgallego@umcutrecht.nl

source /home/cog/mpages/.bashrc
source activate babe

export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64

python /hpc/compgen/users/mpages/babe/scripts/benchmark/train_original.py \
--data-dir /hpc/compgen/projects/nanoxog/babe/analysis/mpages/train_input/human/400.0 \
--output-dir /hpc/compgen/projects/nanoxog/babe/analysis/mpages/models/original \
--model halcyon \
--window-size 400 \
--task human \
--batch-size 64

