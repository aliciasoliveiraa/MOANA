#!/bin/bash
#SBATCH --job-name=processing     # job name
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1         # number of MPI task per node
#SBATCH --output=processing.out  # std out
#SBATCH --error=processing.err   # std err

source path/to/env/bin/activate
module load Python/3.9.5-GCCcore-10.3.0
srun python preprocessing.py
