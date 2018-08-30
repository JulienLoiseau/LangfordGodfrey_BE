#!/bin/bash

#SBATCH -J serveur
#SBATCH --error=./err_go
#SBATCH --output=./out_go

#SBATCH --time=5-00:00:00

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8

srun ./serveur.sh 
