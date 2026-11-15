#!/bin/bash
# Remote environments for fine-tuning runs (scripts/env_workers.py), CPU only, on mit_preemptable; requeued on
# preemption (the learners keep going with fewer environments meanwhile); ends when every run is DONE.
#   CODE_DIR=<checkout> sbatch [-c CPUS] [-J NAME] env_workers.sbatch RUN_DIR:N [RUN_DIR:N ...]
#SBATCH -J envw
#SBATCH -p mit_preemptable
#SBATCH -c 64
#SBATCH --mem-per-cpu=1500M
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o /home/conchell/dyngrasp_rl/rl2/.slurm/%x_%j.out
CODE_DIR=${CODE_DIR:-$HOME/dyngrasp_rl/repo_rl}
echo "== $(date) job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, restart ${SLURM_RESTART_COUNT:-0}, code $CODE_DIR: $*"
module load apptainer/1.5.2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEADLESS=1 PYTHONPATH=$CODE_DIR
cd "$CODE_DIR" || exit 1
exec apptainer exec "$HOME/dyngrasp_rl/dyngrasp_rl.sif" python scripts/env_workers.py "$@"
