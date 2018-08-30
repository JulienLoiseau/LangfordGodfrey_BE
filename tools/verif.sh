#!/bin/bash

#SBATCH -J serveur
#SBATCH --error=./err_verif
#SBATCH --output=./out_verif

#SBATCH --time=01:00:00

#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 8
#SBATCH --gres=gpu:2

srun ./lecture 32768 >> resultat_27.txt 
