#!/bin/bash
# RL PPO runs (scripts/ppo_rl.py, learners only: the environments are env_workers.sbatch --ppo jobs) sharing one GPU on
# mit_preemptable, one process per run.
#   CODE_DIR=<checkout> sbatch [-c CPUS] [--gres=gpu:TYPE:1] [-J NAME] ppo.sbatch PLAN
# PLAN: a text file, one run per line: RUN_NAME KEY=VALUE ... (ppo_rl.py --set; give each run
# ppo.val_workers so that the runs fit in CPUS). Runs go to ~/dyngrasp_rl/rl3/RUN_NAME/ (log: train.log inside).
# Preemption: SIGTERM is forwarded to every trainer (they checkpoint), Slurm requeues the job; 10 min before the time
# limit SIGUSR1 does the same and the job requeues itself. Resubmitting resumes every unfinished run (DONE: skipped).
#SBATCH -J ppo
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem-per-cpu=3G
#SBATCH -t 1-00:00:00
#SBATCH --requeue
# B200/B300 nodes excluded: jax's cuBLAS 12.9 warns of silent data corruption on them (needs cuBLAS >= 13.2)
#SBATCH --exclude=node5500-c1,node5501-c1,node5502-c1,node5600-c1,node5601-c1,node5602-c1,node5700-c1,node5701-c1,node5702-c1,node5800-c1,node5801-c1,node5802-c1,node5900-c1
#SBATCH --signal=B:USR1@600
#SBATCH -o /home/%u/dyngrasp_rl/rl3/.slurm/%x_%j.out
PLAN=${1:?usage: sbatch ppo.sbatch PLAN}
CODE_DIR=${CODE_DIR:-$HOME/dyngrasp_rl/repo_rl}
ROOT=$HOME/dyngrasp_rl/rl3
echo "== $(date) job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, restart ${SLURM_RESTART_COUNT:-0}, code $CODE_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader
module load apptainer/1.5.2
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEADLESS=1
export PYTHONPATH=$CODE_DIR
cd "$CODE_DIR" || exit 1
PIDS=(); NAMES=()
while read -r NAME ARGS; do
  [ -z "$NAME" ] || [[ "$NAME" == \#* ]] && continue
  [ -e "$ROOT/$NAME/DONE" ] && { echo "== $NAME: done"; continue; }
  mkdir -p "$ROOT/$NAME"
  echo "== $(date) start $NAME: $ARGS" | tee -a "$ROOT/$NAME/train.log"
  # shellcheck disable=SC2086
  apptainer exec --nv "$HOME/dyngrasp_rl/dyngrasp_rl.sif" python scripts/ppo_rl.py --run-dir "$ROOT/$NAME" --set $ARGS \
    >> "$ROOT/$NAME/train.log" 2>&1 &
  PIDS+=($!); NAMES+=("$NAME")
done < "$PLAN"
TERMINATED=
trap 'echo "== $(date) time limit near: SIGUSR1 -> trainers"; kill -USR1 "${PIDS[@]}"' USR1
trap 'echo "== $(date) SIGTERM -> trainers"; TERMINATED=1; kill -TERM "${PIDS[@]}"' TERM
FAIL=0
for k in "${!PIDS[@]}"; do
  while true; do wait "${PIDS[$k]}"; RC=$?; kill -0 "${PIDS[$k]}" 2>/dev/null || break; done  # traps interrupt wait
  echo "== $(date) ${NAMES[$k]} exited with $RC"
  [ $RC -ne 0 ] && FAIL=1
done
UNFINISHED=0
for NAME in "${NAMES[@]}"; do [ -e "$ROOT/$NAME/DONE" ] || UNFINISHED=1; done
if [ $FAIL -eq 0 ] && [ $UNFINISHED -eq 1 ] && [ -z "$TERMINATED" ]; then
  echo "== $(date) unfinished runs: requeueing job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"
fi
exit $FAIL
