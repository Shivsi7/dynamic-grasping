#!/bin/bash
# Build ~/dyngrasp_rl/dyngrasp_rl.sif from dyngrasp_rl.def + requirements.lock (in the same directory as this script).
#   cd ~/dyngrasp_rl/repo/scripts/orcd && sbatch build_image.sbatch
#SBATCH -J dg_build
#SBATCH -p mit_normal
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o /home/conchell/dyngrasp_rl/build_%j.log
set -e
module load apptainer/1.5.2
unset APPTAINER_BINDPATH  # the module binds /orcd,/nfs, which don't exist in the image during %post
export APPTAINER_TMPDIR=/tmp APPTAINER_CACHEDIR=$HOME/dyngrasp_rl/apptainer_cache
df -h /tmp | tail -1
apptainer build --force --fakeroot $HOME/dyngrasp_rl/dyngrasp_rl.sif dyngrasp_rl.def
apptainer exec $HOME/dyngrasp_rl/dyngrasp_rl.sif python -c "import pydrake, jax, sbx, stable_baselines3, torch; print('image ok', jax.__version__)"
