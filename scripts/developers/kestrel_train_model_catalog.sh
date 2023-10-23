#!/bin/bash  --login

#SBATCH --job-name=routee-powertrain
#SBATCH --time=4:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --account=mbap
#SBATCH --mail-user=Nicholas.Reinicke@nrel.gov
#SBATCH --mail-type=ALL

module purge
module load anaconda3
. activate /home/$USER/.conda-envs/routee-powertrain

python train_model_catalog.py 