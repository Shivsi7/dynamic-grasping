#!/bin/bash
# Any command in the image, in CODE_DIR (default ~/dyngrasp_rl/repo; export CODE_DIR=<checkout> to run a frozen one).
# CPU on mit_normal by default; for a GPU add -p mit_normal_gpu --gres=gpu:l40s:1 (the image then runs with --nv).
#   sbatch -J bc_demos -c 32 run.sbatch python scripts/make_bc_demos.py generate --out ... --workers 32
#   sbatch -J bc_train -p mit_normal_gpu --gres=gpu:l40s:1 run.sbatch python scripts/train_bc.py --data ... --run-dir ...
#   sbatch -J bc_eval run.sbatch python scripts/eval_rl.py RUN/best --scenarios scenarios/rl_experiment30.yaml --workers 8 --out ...
# Log: ~/dyngrasp_rl/runs/.slurm/<job name>_<job id>.out
#SBATCH -p mit_normal
#SBATCH -c 16
#SBATCH --mem-per-cpu=4G
#SBATCH -t 02:00:00
#SBATCH -o /home/%u/dyngrasp_rl/runs/.slurm/%x_%j.out
CODE_DIR=${CODE_DIR:-$HOME/dyngrasp_rl/repo}
echo "== $(date) job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, GPU ${CUDA_VISIBLE_DEVICES:-none}, code $CODE_DIR"
echo "== $*"
module load apptainer/1.5.2
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEADLESS=1
export PYTHONPATH=$CODE_DIR
cd "$CODE_DIR" || exit 1
NV=; [ -n "$CUDA_VISIBLE_DEVICES" ] && NV=--nv
apptainer exec $NV $HOME/dyngrasp_rl/dyngrasp_rl.sif "$@"
RC=$?
echo "== $(date) exit $RC"
exit $RC
