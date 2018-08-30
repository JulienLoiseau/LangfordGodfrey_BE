#!/bin/bash

#SBATCH -J lgf
#SBATCH --error=./err
#SBATCH --out=./out
#SBATCH --qos=besteffort
#SBATCH --account=besteffort
#SBATCH --partition=romeoBE
#SBATCH --time-min=00:15:00
#SBATCH --time=01:00:00

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --signal=9

export OMP_NUM_THREADS=8
srun ./main $1 $2
