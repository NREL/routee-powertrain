#!/bin/bash  --login

#SBATCH --job-name=routee-batch-train
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --account=mbap

module purge
. activate /home/$USER/.conda/envs/routee-powertrain

python batch-trainer.py config.yml

