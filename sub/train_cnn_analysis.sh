#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH -c 4
#SBATCH --partition gpu
#SBATCH --gpus-per-node=RTX6000:1
#SBATCH --mem=32G
#SBATCH --job-name=babetrain
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.pagesgallego@umcutrecht.nl

source /home/cog/mpages/.bashrc
source activate babe

export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64

python /hpc/compgen/users/mpages/babe/scripts/benchmark/train_cnn_analysis.py \
--cnn-type urnano \
--rnn-size 256 \
--rnn-type lstm \
--num-layers 2 \
--bidirectional \
--window-size 4000 \
--task human


# 'bonito',
# 'catcaller',
# 'causalcall',
# 'halcyon',
# 'mincall',
# 'sacall',
# 'urnano'
