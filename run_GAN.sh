#!/bin/bash -l
#SBATCH --job-name=GAN_Moon
#SBATCH --time=0-6:00:00
#SBATCH --partition=gpu # If gpu: set '-G <gpus>'
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH -G 1
#SBATCH --mem 64000
#SBATCH -C volta
#SBATCH -c 7 # multithreading per task
#SBATCH -o %x-%j.out # ./<jobname>-<jobid>.out

conda activate gdal

cd MoonProject

source_folder=$1
run_name=$2
target_save_path=$3
map_name=$4
source_folder_path=${source_folder}/${map_name}/${run_name}_map
save_path=${target_save_path}/SR/SR_${map_name}
model_path=/home/users/arichard/MoonProject/exp_spade/models/20220724-121426/epoch_6/
image_size=512
batch_size=12
stride=64

python3 process_full_tiles.py --source_folder_path ${source_folder_path}\
			      --map_name ${map_name}\
			      --save_path ${save_path}\
			      --model_path ${model_path}\
			      --batch_size ${batch_size}\
			      --image_size ${image_size}\
			      --stride ${stride}
