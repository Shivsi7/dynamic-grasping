#!/bin/bash
# (D) behaviour cloning as one job on mit_preemptable (1 GPU): train_bc.py (with its closed-loop runs on the 50
# validation scenes), then eval_rl.py on the 30-scene experiment set and compare_rl.py against (A)/(B)/(C).
#   CODE_DIR=<checkout> sbatch [-J NAME] bc.sbatch RUN_DIR DATASET [KEY=VALUE ...]   (KEY=VALUE: train_bc.py --set)
# Requeued on preemption; the rerun skips what finished (training: RUN_DIR/run.json, written last; evaluation:
# RUN_DIR/eval30/compare.txt) and redoes an unfinished step from its start (training is deterministic and short).
#SBATCH -J bc
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
# B200/B300 nodes excluded: jax's cuBLAS 12.9 warns of silent data corruption on them (needs cuBLAS >= 13.2)
#SBATCH --exclude=node5500-c1,node5501-c1,node5502-c1,node5600-c1,node5601-c1,node5602-c1,node5700-c1,node5701-c1,node5702-c1,node5800-c1,node5801-c1,node5802-c1,node5900-c1
#SBATCH -c 32
#SBATCH --mem-per-cpu=3G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH -o /home/%u/dyngrasp_rl/runs/.slurm/%x_%j.out
RUN_DIR=${1:?usage: sbatch bc.sbatch RUN_DIR DATASET [KEY=VALUE ...]}; DATA=${2:?dataset .npz}; shift 2
CODE_DIR=${CODE_DIR:-$HOME/dyngrasp_rl/repo}
echo "== $(date) job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname), $SLURM_CPUS_PER_TASK CPUs, code $CODE_DIR"
module load apptainer/1.5.2
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEADLESS=1
export PYTHONPATH=$CODE_DIR
cd "$CODE_DIR" || exit 1
run() { apptainer exec --nv "$HOME/dyngrasp_rl/dyngrasp_rl.sif" "$@"; }
if [ ! -f "$RUN_DIR/run.json" ]; then
  run python scripts/train_bc.py --data "$DATA" --run-dir "$RUN_DIR" \
    --set "bc.closed_loop_workers=$((SLURM_CPUS_PER_TASK - 2))" "$@" || exit 1
fi
if [ ! -f "$RUN_DIR/eval30/compare.txt" ]; then  # measured latency, 8 episodes in parallel, as (A)/(B)/(C)
  run python scripts/eval_rl.py "$RUN_DIR/best" --scenarios scenarios/rl_experiment30.yaml --workers 8 \
    --out "$RUN_DIR/eval30" || exit 1
  run python scripts/compare_rl.py "$RUN_DIR/eval30/results.csv" --name "$(basename "$RUN_DIR")" \
    > "$RUN_DIR/eval30/compare.tmp" && mv "$RUN_DIR/eval30/compare.tmp" "$RUN_DIR/eval30/compare.txt" || exit 1
fi
cat "$RUN_DIR/eval30/compare.txt"
echo "== $(date) done"
