#!/bin/bash

#SBATCH --job-name=serial
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=1GB 
#SBATCH --time=00:30:00 
#SBATCH --output=serial.out
#SBATCH --verbose
#SBATCH --mail-type=END
#SBATCH --mail-user=qj2030@nyu.edu

module purge 
module load  gcc/10.2.0

./serialres 4097 5