#!/bin/bash
# (D) fine-tuning runs (scripts/finetune_rl.py) on a CPU-only node of mit_normal (JAX on CPU), one process per run:
# the hedge when the GPU queue is long.  Same plan format and run layout as ft.sbatch.
#   CODE_DIR=<checkout> sbatch [-c CPUS] [-J NAME] ft.sbatch PLAN
# PLAN: a text file, one run per line: RUN_NAME KEY=VALUE ... (finetune_rl.py --set; give each run's ft.n_envs and
# ft.val_workers so that the runs fit in CPUS). Runs go to ~/dyngrasp_rl/rl2/RUN_NAME/ (log: train.log inside).
# Preemption: SIGTERM is forwarded to every trainer (they checkpoint), Slurm requeues the job; 10 min before the time
# limit SIGUSR1 does the same and the job requeues itself. Resubmitting resumes every unfinished run (DONE: skipped).
#SBATCH -J ftcpu
#SBATCH -p mit_normal
#SBATCH -c 56
#SBATCH --mem-per-cpu=3G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH -o /home/%u/dyngrasp_rl/rl2/.slurm/%x_%j.out
PLAN=${1:?usage: sbatch ft.sbatch PLAN}
CODE_DIR=${CODE_DIR:-$HOME/dyngrasp_rl/repo_rl}
ROOT=$HOME/dyngrasp_rl/rl2
echo "== $(date) job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, restart ${SLURM_RESTART_COUNT:-0}, code $CODE_DIR"
echo "CPU-only learner"
module load apptainer/1.5.2
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEADLESS=1
export PYTHONPATH=$CODE_DIR
cd "$CODE_DIR" || exit 1
PIDS=(); NAMES=()
while read -r NAME ARGS; do
  [ -z "$NAME" ] || [[ "$NAME" == \#* ]] && continue
  [ -e "$ROOT/$NAME/DONE" ] && { echo "== $NAME: done"; continue; }
  mkdir -p "$ROOT/$NAME"
  echo "== $(date) start $NAME: $ARGS" | tee -a "$ROOT/$NAME/train.log"
  # shellcheck disable=SC2086
  apptainer exec "$HOME/dyngrasp_rl/dyngrasp_rl.sif" python scripts/finetune_rl.py --run-dir "$ROOT/$NAME" --set $ARGS \
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
