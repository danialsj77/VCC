#!/bin/sh
# Run the whole model-v1 pipeline unattended on a TSUBAME 4.0 GPU node:
#   01_build_sources.py  →  02_validate.py --gpu  →  03_make_submission.py
#
# Submit from the project folder (the TSUBAME group goes on the command line, not in this file):
#   qsub -g <your-group> tsubame/run_pipeline.sh
#   qsub -g <your-group> -v SKIP_BUILD=1 tsubame/run_pipeline.sh        # reuse step 01's outputs (e.g. after 02 failed)
#   qsub -g <your-group> -v RESUME=1,SKIP_BUILD=1 tsubame/run_pipeline.sh   # also keep finished validation variants
#   qsub -g <your-group> -v SKIP_BUILD=1,RESUME=1,DRY_RUN=1 tsubame/run_pipeline.sh   # reuse 01/02; rebuild and
#                                                                       #   check the submission only, no .vcc
#   qsub -g <your-group> -v CONTROLS_DIR=controls_DEF tsubame/run_pipeline.sh   # FINAL PHASE bundle (D/E/F)
#   qsub -g <your-group> -v CONFIG=configs/my_variant.yaml tsubame/run_pipeline.sh
# Watch it with `qstat`; the job log is vcc_pipeline.o<jobid>, each step's full log is in work/logs/pipeline.<jobid>/.
#
# Which settings step 03 submits. 02 writes best_params.json (overall winner) and one file per generator. By default
# this job builds the submission from best_params_nb.json: the NB generator samples every count, whereas the template
# generator edits copies of the provided control cells (open rules question: "model inputs only — do not copy them
# into your prediction"). Only once the organizers have confirmed templates in writing:
#   qsub -g <your-group> -v ALLOW_CONTROL_TEMPLATES=1 tsubame/run_pipeline.sh   # overall winner (best_params.json)
#   qsub -g <your-group> -v PARAMS=work/validation/best_params_template.json,ALLOW_CONTROL_TEMPLATES=1 ...
#
# Before the first run (once): bash tsubame/setup_env.sh, the controls in controls/, and the data in data/
# (bash scripts/00_download_data.sh, best on the free interactive queue so this paid job doesn't download ~35 GB).
# The job NEVER submits to the leaderboard: 03 prints the `vcc submit` command for the team lead.
#
# node_q = 1 x H100 (94 GB) + 48 cores + 192 GB RAM, charge coefficient 0.25. Rough timing at full size: 01 ~1-2 h
# (reads the 34 GB of H1 twice), 02 ~2-5 h (20 stage-1 variants + reference + ~10 stage-2 variants, GPU scorer),
# 03 ~1 h (360k cells + vcc prep, which needs ~35 GB of RAM and ~20 GB of temporary disk in $TMPDIR).
# Cost = 0.25 x (0.7 x hours used + 0.1 x hours requested): 6 h of 12 ≈ 1.4 points.
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=12:00:00
#$ -N vcc_pipeline
#$ -j y

# ---- environment modules (CUDA for the GPU scorer) ----
# A batch job's non-interactive shell may not have the `module` command yet: load its init script if needed, and
# stop loudly if it is still missing (otherwise CUDA is silently not loaded and 02 --gpu fails hours later).
if ! command -v module >/dev/null 2>&1; then
    [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
fi
command -v module >/dev/null 2>&1 || { echo "module command missing: cannot load CUDA (no /etc/profile.d/modules.sh?)" >&2; exit 1; }
module purge
module load cuda/12.8.0 || { echo "module load cuda/12.8.0 failed (see 'module avail cuda')" >&2; exit 1; }

set -e                                  # any failing command (outside the explicit checks below) stops the job
CONFIG="${CONFIG:-configs/v1.yaml}"
PY=.venv/bin/python
LOGDIR="work/logs/pipeline.${JOB_ID:-manual}"
export PYTHONUNBUFFERED=1               # step logs fill in as the job runs (tail -f them)
export PATH="$HOME/.local/bin:$PATH"    # where `uv tool install` puts the vcc CLI
# Temporary files: `vcc prep` packages the whole uncompressed prediction (~17 GB) plus its .zst (~3 GB) in $TMPDIR.
# On TSUBAME the node's large local disk is $T4TMPDIR; /tmp may be too small.
export TMPDIR="${T4TMPDIR:-$PWD/work/tmp}"
mkdir -p "$TMPDIR" "$LOGDIR"

echo "[$(date '+%F %T')] vcc_pipeline on $(hostname), config $CONFIG, logs in $LOGDIR"
echo "TMPDIR=$TMPDIR: $(df -h "$TMPDIR" | tail -1)"

# ---- pre-flight: fail in seconds, not hours ----
[ -x "$PY" ] || { echo "No .venv: run 'bash tsubame/setup_env.sh' first." >&2; exit 1; }
"$PY" -c "import cell_eval2, vccpipe.sources, vccpipe.cis, vccpipe.model, vccpipe.proxy" ||
    { echo "Python environment incomplete (see the error above)." >&2; exit 1; }
[ -f controls/gene_names.csv ] || { echo "No controls/: download and unzip the VCC controls bundle." >&2; exit 1; }
if [ -n "${CONTROLS_DIR:-}" ] && [ ! -f "$CONTROLS_DIR/gene_names.csv" ]; then
    echo "CONTROLS_DIR=$CONTROLS_DIR has no gene_names.csv." >&2; exit 1
fi
command -v vcc >/dev/null 2>&1 || { echo "vcc CLI missing (bash tsubame/setup_env.sh installs vcc-cli==0.2.2)." >&2; exit 1; }
echo "vcc CLI: $(vcc version 2>&1 | head -1)"   # 03 relies on vcc 0.2.2's prep flags; recorded for the report
nvidia-smi -L || echo "WARNING: no GPU visible; 02 --gpu will fail." >&2

# The work folder of the config (default work/): where 02 writes the parameter files and 03 the submission.
WORK_DIR=$("$PY" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['paths']['work_dir'])" "$CONFIG")

# run_step NAME COMMAND...: run one step with its output in $LOGDIR/NAME.log; show the log's tail; stop on failure.
run_step() {
    name="$1"; shift
    log="$LOGDIR/$name.log"
    echo "[$(date '+%F %T')] >>> $name: $*"
    if "$@" > "$log" 2>&1; then
        tail -n 40 "$log"
        echo "[$(date '+%F %T')] <<< $name finished"
    else
        status=$?
        tail -n 80 "$log"
        echo "[$(date '+%F %T')] !!! $name FAILED (exit $status); full log: $log" >&2
        exit "$status"
    fi
}

if [ "${SKIP_BUILD:-0}" = 1 ]; then
    echo "SKIP_BUILD=1: reusing the outputs of 01_build_sources.py"
else
    run_step 01_build_sources "$PY" scripts/01_build_sources.py --config "$CONFIG"
fi

VALIDATE_ARGS="--gpu"
[ "${RESUME:-0}" = 1 ] && VALIDATE_ARGS="$VALIDATE_ARGS --resume"
run_step 02_validate "$PY" scripts/02_validate.py --config "$CONFIG" $VALIDATE_ARGS

# Step 03's parameter file (see the header): the NB winner unless control templates are explicitly allowed.
if [ "${ALLOW_CONTROL_TEMPLATES:-0}" = 1 ]; then
    PARAMS="${PARAMS:-$WORK_DIR/validation/best_params.json}"
    set -- --allow-control-templates
else
    PARAMS="${PARAMS:-$WORK_DIR/validation/best_params_nb.json}"
    set --
fi
# 03 writes to <work>/submission_<controls folder>/ (e.g. submission_controls_DEF), so the final-phase file never
# overwrites the A/B/C one. Arguments are collected in "$@" so a path with spaces stays one argument.
SUBMIT_OUT="$WORK_DIR/submission_$(basename "${CONTROLS_DIR:-controls}")"
[ -n "${CONTROLS_DIR:-}" ] && set -- "$@" --controls-dir "$CONTROLS_DIR"
[ "${DRY_RUN:-0}" = 1 ] && set -- "$@" --dry-run
run_step 03_make_submission "$PY" scripts/03_make_submission.py --config "$CONFIG" --params "$PARAMS" "$@"

echo "[$(date '+%F %T')] pipeline finished. Validation table: $WORK_DIR/validation/results.csv; settings used:"
echo "  $PARAMS; submission: $SUBMIT_OUT/ (the vcc submit command is in the 03 log)."
