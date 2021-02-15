#!/bin/bash
#SBATCH --chdir C:/Users/thiba/OneDrive/Bureau/emg1
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

slmodules -s x86_E5v2_Mellanox_GPU
module load gcc cuda cudnn mvapich2 openblas
source venvs/venvs/emg_tn/bin/activate
srun python emg1.py