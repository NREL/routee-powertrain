#!/usr/bin/bash
#SBATCH --account=mbap
#SBATCH --time=3:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --mail-user=jacob.holden@nrel.gov
#SBATCH --mail-type=ALL


export PROJ_ENV="/home/jholden/.conda/envs/routee-powertrain"
export CODE_DIR=$(pwd)

srun bash -l <<EOF
module purge
module load conda
. activate "$PROJ_ENV"
~/.conda/envs/routee-powertrain/bin/python "$CODE_DIR"/model_trainer.py
EOF