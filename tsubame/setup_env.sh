#!/bin/bash
# One-time setup of the VCC Python environment on TSUBAME 4.0 (safe to re-run).
#
# Run it ON A COMPUTE NODE, never on a login node (login nodes have a 4 GB/process limit and no GPU):
#   iqrsh -l h_rt=1:00:00                          # free interactive queue (on campus), or:
#   qrsh -g <your-group> -l gpu_1=1 -l h_rt=1:00:00
#   cd /gs/bs/<your-group>/$USER/VCC                # YOUR OWN clone on the group disk (home+work is only 125 GiB)
#   bash tsubame/setup_env.sh
set -euo pipefail

# Always work in the project folder (the parent of this script's folder), whatever the current directory is.
cd "$(dirname "$0")/.."
[ -d notebooks ] && [ -d tsubame ] || { echo "Run this from a clone of the project (notebooks/ and tsubame/ missing)." >&2; exit 1; }
echo "Project folder: $(pwd)"

# Environment modules (CUDA). A non-interactive shell may not have the `module` command yet: load its init script if
# needed (with `set -u` off: the init script reads variables that may be unset), and stop if it is still missing.
if ! command -v module >/dev/null 2>&1; then
    set +u
    [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
    set -u
fi
command -v module >/dev/null 2>&1 || { echo "module command missing: cannot load CUDA (no /etc/profile.d/modules.sh?)" >&2; exit 1; }
set +u                           # module scripts may also read unset variables
module purge
module load cuda/12.8.0          # check what exists with: module avail cuda
set -u

# uv = fast Python package manager. System Python on TSUBAME is 3.9; cell-eval2 needs >= 3.11,
# so uv downloads its own Python 3.12 (needs outbound internet; TSUBAME compute nodes have it).
python3 -m pip install --user --upgrade uv
export PATH="$HOME/.local/bin:$PATH"

# --seed puts pip into the venv (TSUBAME's kernel helper may need it); --allow-existing makes re-runs safe.
uv venv .venv --python 3.12 --seed --allow-existing
# The same pinned packages as on the Mac (requirements.txt), plus the GPU extras: cupy (pseudobulk/rank kernels)
# and gpudge (the GPU DE engine the platform uses), pinned so every teammate's GPU scores are comparable.
uv pip install --python .venv/bin/python --torch-backend=cu128 \
    -r requirements.txt "cell-eval2[gpu,gpudge]==0.18.0" "gpudge==0.9.1"

# The vcc CLI (download data, vcc prep, vcc submit), pinned: 03_make_submission.py and the RUNBOOK rely on vcc 0.2.2's
# `prep` flags and defaults (--contexts, --cells-per-pert, --dry-run, -o, -f). If the platform requires a newer
# version, update the pin here and re-check those flags (vcc prep --help).
uv tool install 'vcc-cli==0.2.2' || uv tool install --force 'vcc-cli==0.2.2'   # --force: replace another installed version
vcc version

# Register the environment as the Jupyter kernel "vcc" for Open OnDemand. TSUBAME's helper records the loaded
# modules (CUDA) with the kernel; a plain `ipykernel install` cannot.
source .venv/bin/activate
if ! /apps/t4/rhel9/utils/bin/t4jpyregister -n vcc; then
    echo "WARNING: t4jpyregister failed; registering a plain kernel instead (no module environment)." >&2
    python -m ipykernel install --user --name vcc --display-name "VCC (project .venv)"
fi

# Quick check. On a GPU node the DE engine must be gpudge; this raises if gpudge is installed but unusable.
python - <<'EOF'
import importlib.metadata as md
for p in ("cell-eval2", "gpudge", "cupy-cuda12x", "torch", "anndata", "scanpy", "papermill"):
    try:
        print(f"{p:14s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p:14s} (not installed)")
import torch
print("CUDA available:", torch.cuda.is_available())
from cell_eval2.de_compute import _resolve_backend
print("DE engine (auto):", _resolve_backend("auto"))
EOF
echo "Done. Kernel list:"; jupyter kernelspec list
echo "In Open OnDemand > Jupyter, pick the 'vcc' kernel."
