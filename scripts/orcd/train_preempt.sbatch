#!/bin/bash
# (D) SAC training on mit_preemptable: 1 GPU of any type, 66 CPUs, 1 day. Preemption = SIGTERM to every process,
# SIGKILL 30 s later, then Slurm requeues the job and train_rl.py resumes from the run's last periodic checkpoint.
#   sbatch [-c CPUS] ~/dyngrasp_rl/train_preempt.sbatch RUN [SEED=0] [N_ENVS=CPUS-2] [more train_rl.py --set KEY=VALUE ...]
# Writes ~/dyngrasp_rl/runs/RUN/ (slurm_<jobid>.out inside). Submitting the same RUN again resumes from its last checkpoint.
# 10 min before the time limit the trainer gets SIGUSR1 (to checkpoint).
#SBATCH -J dg_train
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH -c 66
#SBATCH --mem-per-cpu=2G
#SBATCH -t 1-00:00:00
#SBATCH --requeue
# B200/B300 nodes excluded: jax's cuBLAS 12.9 warns of silent data corruption on them (needs cuBLAS >= 13.2)
#SBATCH --exclude=node5500-c1,node5501-c1,node5502-c1,node5600-c1,node5601-c1,node5602-c1,node5700-c1,node5701-c1,node5702-c1,node5800-c1,node5801-c1,node5802-c1,node5900-c1
#SBATCH --signal=B:USR1@600
#SBATCH -o /home/conchell/dyngrasp_rl/runs/.slurm/%x_%j.out
RUN=${1:?usage: sbatch train_preempt.sbatch RUN [SEED] [N_ENVS] [KEY=VALUE ...]}; SEED=${2:-0}; N_ENVS=${3:-$((SLURM_CPUS_PER_TASK - 2))}
shift $(($# < 3 ? $# : 3))
RUN_DIR=$HOME/dyngrasp_rl/runs/$RUN
mkdir -p "$RUN_DIR"
exec >> "$RUN_DIR/slurm_$SLURM_JOB_ID.out" 2>&1
echo "== $(date) job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name --format=csv,noheader
module load apptainer/1.5.2
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEADLESS=1
export PYTHONPATH=$HOME/dyngrasp_rl/repo
cd $HOME/dyngrasp_rl/repo
apptainer exec --nv $HOME/dyngrasp_rl/dyngrasp_rl.sif python scripts/train_rl.py \
    --run-dir "$RUN_DIR" --set train.seed=$SEED train.n_envs=$N_ENVS "$@" &
PID=$!
trap 'echo "== $(date) time limit near: SIGUSR1 -> trainer"; kill -USR1 $PID' USR1
wait $PID; RC=$?
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done  # the trap interrupts wait: wait for the checkpoint too
echo "== $(date) trainer exited with $RC"
exit $RC
