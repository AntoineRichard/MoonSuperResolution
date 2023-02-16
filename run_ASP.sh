#!/bin/bash -l
#SBATCH --job-name=ASP_Full_Stereo_Pipeline
#SBATCH --time=0-6:00:00
#SBATCH --partition=batch
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -o %x-%j.out # ./<jobname>-<jobid>.out

# Variables
exp_folder=$1
run_name=$2

# This scripts expects the following structure:
#
# + firstpair_secondpair
# 	+ firstpairLE.IMG
# 	+ firstpairRE.IMG
#	+ secondpairLE.IMG
#	+ secondpairRE.IMG
#
# Example:
# + M104318871_M104311715
#	+ M104311715LE.IMG
#	+ M104311715RE.IMG
#	+ M104318871LE.IMG
#	+ M104318871RE.IMG

# Exports the ASP path
export PATH=${PATH}:${HOME}/ASP/ASP_bins/bin
# (OPTIONAL) Loads conda on the HPC
module load lang/Anaconda3/2020.11
# Activates our conda environment
conda activate isis

# (OPTIONAL) When using multiple nodes you can get the list of nodes using the following code
cd $SLURM_SUBMIT_DIR
nodesList=$(mktemp -p $(pwd))
scontrol show hostname $SLURM_NODELIST | tr ' ' '\n' > $nodesList

# (OPTIONAL) Going to somewhere with large storage capacities
cd ${SCRATCH}/${exp_folder}

# Collects the name of the image pairs
for i in $(ls *.IMG)
do
	echo ${i%??.*} >> image_list.txt
done

sort image_list.txt | uniq > unique_images.txt
/bin/rm -rf image_list.txt

# Preprocessing and stiching of the pairs
for img in $(cat unique_images.txt)
do
	lronac2mosaic.py ${img}LE.IMG ${img}RE.IMG --threads 128
done

# Variables for the cubs
left_cub=$(head -n 1 unique_images.txt)"LE.lronaccal.lronacecho.noproj.mosaic.norm.cub"
right_cub=$(tail -n 1 unique_images.txt)"LE.lronaccal.lronacecho.noproj.mosaic.norm.cub"

# Remove unused files
/bin/rm unique_images.txt

# Run coarse stereo reconstruction
parallel_stereo ${left_cub} ${right_cub} --job-size-w 1024 --job-size-h 1024  --subpixel-mode 1 ${run_name}_nomap/run --processes 32 --threads-multiprocess 4 --threads-singleprocess 128 --keep-only 'PC.tif'

# Reconstructs a coarse DEM
point2dem --search-radius-factor 5 --tr 0.0013 ${run_name}_nomap/run-PC.tif --threads 128

# Map-projects the cub images onto the coarse DEM
mapproject --tr 0.000033 ${run_name}_nomap/run-DEM.tif ${left_cub} ${run_name}_left_proj.tif   --processes 128 --threads 1
mapproject --tr 0.000033 ${run_name}_nomap/run-DEM.tif ${right_cub} ${run_name}_right_proj.tif --processes 128 --threads 1

# Makes a fine stereo reconstruction
parallel_stereo --job-size-w 1024 --job-size-h 1024 --stereo-algorithm asp_mgm --subpixel-mode 3 ${run_name}_left_proj.tif ${run_name}_right_proj.tif ${left_cub} ${right_cub} ${run_name}_map/run ${run_name}_nomap/run-DEM.tif --processes 32 --threads-multiprocess 4 --threads-singleprocess 128 --keep-only 'L.tif PC.tif F.tif' --corr-tile-size 1024 --sgm-collar-size 512

# Remove the coarse stereo reconstruction data.
/bin/rm -r ${run_name}_nomap

# Reconstructs the fine DEM
point2dem --nodata-value -32768 --tr 0.000033 ${run_name}_map/run-PC.tif --orthoimage ${run_name}_map/run-L.tif --threads 128  --dem-hole-fill-len 5 --orthoimage-hole-fill-len 5

# Remove the left-over data
/bin/rm ${run_name}_left_proj.tif ${run_name}_right_proj.tif
/bin/rm -fv $nodesList

