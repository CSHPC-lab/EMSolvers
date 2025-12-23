#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --partition=all
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH -o ./log/%j.out
#SBATCH -e ./log/%j.err

./execute.sh

cd output
python a.py
python a_plot.py