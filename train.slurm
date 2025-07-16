#!/bin/bash

#SBATCH --partition=v100
#SBATCH --job-name=MIPA_2B
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output=experiments/jobout/MIPA_2B_%j.o
#SBATCH --error=experiments/jobout/MIPA_2B_%j.e
#SBATCH --account=gen15621@v100
#SBATCH --mail-user=emilie.tardieu@inrae.fr
#SBATCH --mail-type=BEGIN,END

cd $MYHOME
module load gnu/11 mpi/openmpi/4 flavor/python3/cuda-12.2 python3/3.11.4

echo Execution of mipanet_2B
srun python /ccc/cont003/home/inp/tardieue/MY_MIPANet_2_branches/lancer_entrainement.py

echo ===DONE===