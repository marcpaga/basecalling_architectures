#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH -c 4
#SBATCH --partition gpu
#SBATCH --gpus-per-node=RTX6000:1
#SBATCH --mem=32G
#SBATCH --job-name=crf
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.pagesgallego@umcutrecht.nl

source /home/cog/mpages/.bashrc
source activate babe

export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64

python /hpc/compgen/users/mpages/babe/scripts/nn/train.py \
--config-file /hpc/compgen/users/mpages/babe/models/bonito_crf/config.py
