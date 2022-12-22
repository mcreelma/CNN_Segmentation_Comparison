#!/bin/bash 
#SBATCH -J pytorch      # job name 
#SBATCH --mail-type=end 
#SBATCH --mail-user=mitchcreelman@boisestate.edu 
#SBATCH -o log_slurm.o%j # outpt & err fname (%j expands to jobID) 
# Specify job not to be rerunable. 
#SBATCH --no-requeue 
# Job Name. 
#SBATCH --job-name="RawUnet" 
#SBATCH -n 48        # total number of tasks requested 
#SBATCH -N 1        # number of nodes you want to run on 
#SBATCH --gres=gpu:1   # request a gpu 
#SBATCH -p gpu   # queue (partition) 
#SBATCH -t 15:00:00   # run time (hh:mm:ss) - 15.0 hrs. 

# Load necessary modules  
module load cudnn8.0-cuda11.0/8.0.5.39  

# Activate the conda environment 
. ~/.bashrc 
conda activate NN2

###### Your code goes here #######
python UNET_Raw_Train.py  > UNETRaw/UNET_Results.txt

# To submit the batch file I type  
# sbatch conda-UNET-slurm.bash 
