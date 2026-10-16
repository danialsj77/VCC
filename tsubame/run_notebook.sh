#!/bin/sh
# Run notebooks/01_local_evaluation_harness.ipynb unattended on a TSUBAME 4.0 GPU node.
#
# Submit from the project folder (the TSUBAME group goes on the command line, not in this file):
#   qsub -g <your-group> tsubame/run_notebook.sh
#   qsub -g <your-group> -v VCC_PROXY=h1_2025 tsubame/run_notebook.sh     # real-biology proxy (§4b)
# Watch it with `qstat`; the log is vcc_harness.o<jobid>.
# The executed notebook is saved after every cell as work/runs/01_local_evaluation_harness.<proxy>.<jobid>.ipynb.
#
# node_q = 1 × H100 (94 GB) + 48 cores + 192 GB RAM, charge coefficient 0.25.
# Cost = 0.25 × (0.7 × hours used + 0.1 × hours requested): e.g. 3 h used of 8 h requested ≈ 0.73 points (~45 JPY
# on campus). Requesting generous time is cheap; running out of time loses the remaining cells.
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=8:00:00
#$ -N vcc_harness
#$ -j y

# Environment modules (CUDA for the GPU scorer). A batch job's non-interactive shell may not have the `module` command
# yet: load its init script if needed, and stop loudly if it is still missing (otherwise CUDA is silently not loaded).
if ! command -v module >/dev/null 2>&1; then
    [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
fi
command -v module >/dev/null 2>&1 || { echo "module command missing: cannot load CUDA (no /etc/profile.d/modules.sh?)" >&2; exit 1; }
module purge
module load cuda/12.8.0 || { echo "module load cuda/12.8.0 failed (see 'module avail cuda')" >&2; exit 1; }

# Temporary files (the notebook's `vcc prep --dry-run` and the scorer write large ones): the node's local disk.
export TMPDIR="${T4TMPDIR:-$PWD/work/tmp}"
mkdir -p "$TMPDIR"
export PATH="$HOME/.local/bin:$PATH"         # where `uv tool install` puts the vcc CLI
echo "vcc CLI: $(vcc version 2>&1 | head -1); TMPDIR=$TMPDIR"

export VCC_MODE=tsubame_full                 # up to 300 perturbations, GPU scorer
export VCC_PROXY="${VCC_PROXY:-synthetic}"   # or h1_2025

# Pre-flight: the environment must exist (bash tsubame/setup_env.sh).
.venv/bin/python -c "import cell_eval2, anndata, papermill" || { echo "Run tsubame/setup_env.sh first." >&2; exit 1; }

# Bring the .ipynb up to date with its paired .py (in case the .py was edited outside Jupyter).
.venv/bin/jupytext --sync notebooks/01_local_evaluation_harness.py

# papermill writes the output notebook after every cell, so a crash or time-out still leaves the results so far.
# Executed copies go to work/runs/ (not tracked by git).
mkdir -p work/runs
.venv/bin/papermill -k vcc --cwd notebooks \
    notebooks/01_local_evaluation_harness.ipynb \
    "work/runs/01_local_evaluation_harness.${VCC_PROXY}.${JOB_ID}.ipynb"
