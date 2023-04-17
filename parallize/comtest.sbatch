#!/bin/bash

#SBATCH --job-name=comptest
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --mem=1GB 
#SBATCH --time=00:30:00 
#SBATCH --output=comptest.out
#SBATCH --verbose
#SBATCH --mail-type=END
#SBATCH --mail-user=qj2030@nyu.edu

module purge 
module load  amber/openmpi/intel/20.06 

mpiexec ./residualtest