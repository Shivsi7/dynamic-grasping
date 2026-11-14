# ORCD (MIT Engaging) training environment

`~/dyngrasp_rl/dyngrasp_rl.sif`: python:3.12.16-bookworm + venv `/opt/venv` from `requirements.lock` (Drake 1.57, jax[cuda12] 0.11.2, sbx 0.28, CPU-only torch). Tested on L40S, L4, A100, H100, H200 and RTX Pro 6000; `train_preempt.sbatch` excludes the B200/B300 nodes (cuBLAS 12.9 data-corruption warning there). The repo is not in the image: it lives in `~/dyngrasp_rl/repo`, and the job scripts put it on `PYTHONPATH`.

- Sync the repo (Mac, repo root): `rsync -az --delete --exclude={.venv,results,runs,__pycache__,.pytest_cache,.ruff_cache,.claude,.DS_Store} ./ orcd:dyngrasp_rl/repo/`
- Rebuild the image (apptainer `--fakeroot`, ~5 min): `cd ~/dyngrasp_rl/repo/scripts/orcd && sbatch build_image.sbatch`. After changing deps, regenerate the lock with `pip freeze --exclude-editable` inside the image.
- One-time setup: `mkdir -p ~/dyngrasp_rl/runs/.slurm && ln -sf ~/dyngrasp_rl/repo/scripts/orcd/train_*.sbatch ~/dyngrasp_rl/`
- Submit on L40S (16 CPUs, 6 h): `sbatch ~/dyngrasp_rl/train_normal.sbatch RUN [SEED] [N_ENVS] [KEY=VALUE ...]`
- Submit on preemptable (any GPU, 66 CPUs, 1 day, requeued on preemption): `sbatch [-c 32] ~/dyngrasp_rl/train_preempt.sbatch RUN ...`
- Outputs go to `~/dyngrasp_rl/runs/RUN/`. Resubmitting the same RUN resumes it. SIGUSR1 arrives 10 min before the time limit.
- Docker copy of the same environment (public, no repo code): `apptainer pull dyngrasp_rl.sif docker://victoriano012/dyngrasp-rl:<tag>`. To rebuild it on the Mac: `docker buildx build --platform linux/amd64 -t victoriano012/dyngrasp-rl:<date> .`
- Behaviour cloning (D) (any command, CPU or GPU): `scripts/orcd/run.sbatch`, examples in its header: demonstrations (`make_bc_demos.py generate`, then `merge`), training (`train_bc.py`, on a GPU), evaluation (`eval_rl.py`, then `compare_rl.py`).
