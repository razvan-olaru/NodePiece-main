#!/bin/bash

#SBATCH --job-name=test
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1

source /home/rolaru/.bashrc
source activate

# cd /home/NodePiece/nc

srun python -u /home/rolaru/NodePiece/lp_rp/run_lp.py python run_lp.py -loop slcwa -loss nssal -margin 15 -b 512 -data wn18rr -anchors 500 -sp 100 -lr 0.0005 -ft_maxp 50 -pool cat -embedding 50 -negs 20 -subbatch 2000 -sample_rels 4 -epochs 500

