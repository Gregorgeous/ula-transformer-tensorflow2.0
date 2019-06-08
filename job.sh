#!/bin/sh
#SBATCH --time 0           # time in minutes to reserve
#SBATCH --cpus-per-task 4  # number of cpu cores
#SBATCH --mem 32G           # memory pool for all cores
#SBATCH --gres gpu:4       # number of gpu cores
#SBATCH  -o transformer90k500eps.log      # write output to log file

echo "========= PYTHON VERSION (to establish if running through pure python or Anaconda =========== "
python --version
echo "=====START OF THE ACTUAL JOB ===="
export PYTHONIOENCODING=utf-8
srun -l python MAIN.py

