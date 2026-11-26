#!/bin/bash
# A (D) policy on the 30-scene experiment set as (A)/(B)/(C) were run (measured latency, 8 episodes in parallel; CPU
# only), then compare_rl.py against them. Writes OUT/results.csv and OUT/compare.txt (a finished OUT is skipped).
#   CODE_DIR=<checkout> sbatch [-J NAME] eval30.sbatch POLICY_DIR OUT [NAME]
#SBATCH -J eval30
#SBATCH -p mit_preemptable
#SBATCH -c 10
#SBATCH --mem-per-cpu=3G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH -o /home/%u/dyngrasp_rl/rl2/.slurm/%x_%j.out
POLICY=${1:?usage: sbatch eval30.sbatch POLICY_DIR OUT [NAME]}; OUT=${2:?OUT}; NAME=${3:-$(basename "$OUT")}
CODE_DIR=${CODE_DIR:-$HOME/dyngrasp_rl/repo_rl}
echo "== $(date) job $SLURM_JOB_ID on $(hostname), code $CODE_DIR, policy $POLICY"
module load apptainer/1.5.2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEADLESS=1 PYTHONPATH=$CODE_DIR
cd "$CODE_DIR" || exit 1
run() { apptainer exec "$HOME/dyngrasp_rl/dyngrasp_rl.sif" "$@"; }
[ -f "$OUT/compare.txt" ] && { cat "$OUT/compare.txt"; exit 0; }
run python scripts/eval_rl.py "$POLICY" --scenarios scenarios/rl_experiment30.yaml --workers 8 --out "$OUT" || exit 1
run python scripts/compare_rl.py "$OUT/results.csv" --name "$NAME" > "$OUT/compare.tmp" && mv "$OUT/compare.tmp" "$OUT/compare.txt" || exit 1
cat "$OUT/compare.txt"
