#!/bin/bash

#SBATCH --account=p32294
#SBATCH -p gengpu
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=150G 
#SBATCH --job-name=alldata
#SBATCH --output=alldata122024 ## standard out and standard error goes to this file

module purge
module load python-miniconda3/4.12.0
conda init bash
source ~/.bashrc

eval "$(conda shell.bash hook)"

source activate /projects/p32294/conda_env/newfort3
python3 scripts/train_nn.py

# 4099


# ObjectFolder-Real	1417K 
# Invariant	   1165K 
# Panda-Probe	820K
# VisGel-Downsampled	726K
# Yuan18	494K
# NeuralFeels	404K
# Touch-and-Go	262K
# TVL	82K
# TarF	38K
# Calandra17	24K
# Braille-TD	20K
# Octopi	15K
# Tippur23	        12k
# CNC-Probe	9K
# YCB                         2k