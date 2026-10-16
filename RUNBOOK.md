# RUNBOOK: running model v1 on TSUBAME

For **Ken** (or whoever runs the cluster job). The commands are meant to be copied as they are. Replace only the
placeholders `<group>` (our TSUBAME group, `tga-…`) and `<login>` (your TSUBAME login name).

**Be aware: v1 has never run on real data.** Every piece was tested on small fake data on a laptop and reviewed
against the real files, but it has not run on them at full size, or on TSUBAME. This first cluster run *is* the
test, so expect at least one problem. [§9](#9-troubleshooting) covers the likely ones. When something fails, don't
patch the code on the cluster: send the logs ([§10](#10-what-to-send-back)) and we fix it in git.

What the run does (details: [docs/DESIGN_v1.md](docs/DESIGN_v1.md); reasoning: [METHOD.md](METHOD.md) §3
"What v1 implements"):

```
00_download_data.sh    public data, ~35.5 GB ─▶ data/replogle/, data/h1_2025/, data/annotation/ (GENCODE v32)
01_build_sources.py    response tables + H1 proxy ─▶ data/processed/{k562_gw,h1_2025}.npz, tss.csv, cis.json
                                                     work/h1_proxy/real.h5ad, targets.txt, real.kd.csv
02_validate.py --gpu   official scorer on held-out H1 targets ─▶ work/validation/results.csv, best_params.json,
                                                     best_params_nb.json, best_params_template.json
03_make_submission.py  the A/B/C submission ─▶ work/submission_controls/submission.vcc, submission.json
```

Two decisions are already made for this run, so you don't have to make them:

- **The submission uses the NB generator** (`best_params_nb.json`): every count is sampled from a model fitted to
  the controls. The other generator (`template`) edits copies of the challenge's control cells, and the data page
  says not to copy them into a prediction. We have asked the organizers (task #7). Until they confirm in writing
  that templates are allowed, 03 refuses to build a template submission, and 02 scores the template generator only
  for comparison.
- **The submitted model is the validated model.** Its only source is K562. H1 is the held-out line the two gains
  are fitted on, and it is never averaged into the prediction.

Time budget: about **1 hour of your time** spread over a day, plus 4–8 hours of unattended job time (estimate).

---

## 1. Prerequisites (check once)

| What | How to check |
|---|---|
| A TSUBAME 4.0 account with an **ed25519** SSH key registered in the [TSUBAME portal](https://portal.t4.gsic.titech.ac.jp/ptl/) | `ssh <login>@login.t4.gsic.titech.ac.jp` works |
| Membership of our TSUBAME group `<group>`, which has **points** (a faculty or staff member owns it) | `t4-user-info group point` |
| **About 130 GB free on the group disk** `/gs/bs/<group>`: ~36 GB of downloads, ~10 GB for the Python environment, ~5 GB for the H1 proxy, up to ~55 GB while 03 builds a submission (~17 GB of chunks plus the ~17 GB combined `.h5ad`, the 3–5 GB `.vcc`, and the other phase's `.vcc`), and **~20 GB of temporary files while `vcc prep` packages** (on the node's local disk when the job has one, otherwise in `work/tmp/`; see §5b). Home (125 GiB, not backed up) is too small | `t4-user-info disk group` |
| **GitHub**: you are a collaborator on the private repo `danialsj77/VCC`, and TSUBAME has its own SSH key for GitHub: `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github`, add the `.pub` to your GitHub account, and put `Host github.com` / `  IdentityFile ~/.ssh/id_ed25519_github` in `~/.ssh/config` | `ssh -T git@github.com` says "Hi <you>!" |
| **Virtual Cell Challenge**: you are registered, a member of our team, and have **your own API key** (web app → Account → Credentials). Never use someone else's key, and never put a key in a file, script, notebook or chat | `vcc whoami` (after §2) names our team |

The disk numbers are estimates. Please report the real ones (`du -sh data work` after the run).

## 2. One-time setup (~30 min)

Log in, then **move to a compute node**. Login nodes are for editing only (4 GB per process, no GPU), and
installing or running anything heavy there is not allowed.

```bash
ssh <login>@login.t4.gsic.titech.ac.jp
iqrsh -l h_rt=1:00:00                    # free interactive queue (on campus). If it is full or you are off campus:
                                         #   qrsh -g <group> -l gpu_1=1 -l h_rt=1:00:00     (costs a few points)
```

On the compute node:

```bash
export GROUP=<group>                     # e.g. tga-xxxxx. Put this line in ~/.bashrc so later sessions have it
mkdir -p /gs/bs/$GROUP/$USER && cd /gs/bs/$GROUP/$USER

# Your OWN clone (a shared clone breaks other people's .venv and outputs).
git clone git@github.com:danialsj77/VCC.git && cd VCC
git switch model-v1                      # v1 lives on this branch until it is merged; skip once it is on main
git log -1 --oneline                     # note this commit for the report (§10)

# Python 3.12 environment with the GPU scorer (cell-eval2 + gpudge), the vcc CLI, and the Jupyter kernel "vcc".
bash tsubame/setup_env.sh                # ~10–20 min; safe to re-run
export PATH="$HOME/.local/bin:$PATH"     # where `vcc` is installed; add this line to ~/.bashrc too
vcc version                              # must be 0.2.2 (the pipeline relies on its prep flags); if it is not:
                                         #   uv tool install --force 'vcc-cli==0.2.2'

# The challenge CLI needs YOUR key. There is no OS keychain on the cluster, so the key is kept in a private file
# (mode 0600) in your home directory. Paste it at the hidden prompt and press Enter.
vcc login --store-plaintext
vcc whoami                               # must name our team

# The challenge's control cells (~630 MB zip; not in git).
vcc datasets download controls -o controls.zip && unzip -o controls.zip -d controls
ls controls       # context_A.h5ad context_B.h5ad context_C.h5ad gene_names.csv manifest.json pert_counts.csv
```

If `git switch model-v1` says `invalid reference: model-v1`, the branch has not been pushed to GitHub yet. Stop
there and tell Daniyal; don't work on `main`, which has no pipeline.

**Shared data folder (recommended).** Large datasets are downloaded once for the whole team into the group's shared
folder and linked into each clone. Do this **before** §3, inside your clone:

```bash
mkdir -p /gs/bs/$GROUP/shared/data
ls -la data 2>/dev/null || ln -s /gs/bs/$GROUP/shared/data data      # only if the clone has no data/ yet
```

`data/processed/` (written by script 01) then also lands in the shared folder, so teammates can open notebook 02
without rebuilding it. Don't run two pipelines at the same time against the same shared folder.

If the interactive session ends (1 h limit) in the middle of a step, start a new one (`iqrsh …`, `cd` back into
the clone, `export PATH=…`) and repeat the step. Every step here is safe to re-run.

## 3. Download the public data (once, ~35.5 GB)

On a compute node (the free interactive queue is fine), from the clone:

```bash
cd /gs/bs/$GROUP/$USER/VCC
bash scripts/00_download_data.sh --list  # the plan and what is already on disk (no network)
bash scripts/00_download_data.sh         # everything: Replogle bulk ~0.55 GB, GENCODE ~30 MB, H1 ~34 GB
```

| Group | Files | Size | Used for |
|---|---|---|---|
| `replogle` | `data/replogle/K562_gwps_raw_bulk_01.h5ad` (+ K562 essential and RPE1, not used by v1 yet) | ~0.55 GB | v1's only source (K562 genome-wide) |
| `gencode` | `data/annotation/gencode.v32.basic.annotation.gtf.gz` | ~30 MB | gene start sites for the cis model. Release 32 has the same gene symbols as the challenge's gene list (10x GRCh38-2020-A); newer releases rename ~500 of them |
| `h1` | `data/h1_2025/adata_Test.h5ad` 12.0 GB, `adata_Validation.h5ad` 6.9 GB, `adata_Training.h5ad` 15.5 GB | ~34 GB | the held-out line: the proxy we score on, and the line the gains are fitted on |

- **Expected time** depends on the network. At 50–100 MB/s it takes about 6–12 minutes, at 10 MB/s about an hour.
  Watch the first H1 file's progress for a minute and extrapolate.
- **It is resumable.** Complete files are skipped and interrupted ones resume from their `.part` file. If the
  session ends or the script reports `INCOMPLETE`, run it again. You can also fetch the groups one at a time:
  `bash scripts/00_download_data.sh replogle gencode`, then `bash scripts/00_download_data.sh h1`.
- It is done when `--list` shows every file as `complete`.

## 4. Check the code on the cluster (~10–15 min, before spending points)

The laptop tests run the whole 01 → 02 → 03 chain on small **fake** data (with the real controls bundle and
`vcc prep --dry-run`), so they need no downloads and can run right after §2. They check the environment, the
scorer and the `vcc` CLI before a multi-hour job. Run them on a compute node (≤ 8 GB RAM; CPU scorer):

```bash
cd /gs/bs/$GROUP/$USER/VCC && mkdir -p work
for t in tests/test_*.py; do echo "== $t"; .venv/bin/python "$t"; done 2>&1 | tee work/test_report.txt
grep -oE "^(PASS|FAIL|SKIP)" work/test_report.txt | sort | uniq -c          # counts per outcome
```

Every line should be `PASS`. A `FAIL` comes with a traceback in `work/test_report.txt`: send that file (§10) and
wait for a fix before §5.

(Each script also has a `--smoke` mode that runs on the **real** files at tiny sizes and writes only to
`work/smoke/`. It is not much faster for step 01, which still has to read every H1 file, so the full job below is
the better use of time.)

## 5. The pipeline

### 5a. As one batch job (recommended)

```bash
cd /gs/bs/$GROUP/$USER/VCC
qsub -g $GROUP tsubame/run_pipeline.sh
qstat                                    # state: qw = waiting, r = running
```

- Runs on a `node_q` (1 × H100, 48 cores, 192 GB RAM) with a 12-hour limit. It checks the environment first
  (fails in seconds if `.venv`, `controls/`, the `vcc` CLI or the Python modules are missing), then runs 01 → 02
  (GPU scorer) → 03 and **stops at the first failing step**.
- **Logs:** `vcc_pipeline.o<jobid>` in the clone (summary and the tail of each step), and the full log of each
  step in `work/logs/pipeline.<jobid>/01_build_sources.log`, `02_validate.log`, `03_make_submission.log`.
  Follow a step live: `tail -f work/logs/pipeline.<jobid>/02_validate.log`.
- It **never submits** anything. 03 only prints the `vcc submit` command for Daniyal.
- Step 03 builds the submission from **`work/validation/best_params_nb.json`** (NB generator), whichever generator
  won validation overall. The job also points `TMPDIR` at a large disk for `vcc prep` and writes the free space and
  `vcc version` at the top of its log.
- **Template generator: only on Daniyal's word** that the organizers confirmed it in writing. Then:
  `qsub -g $GROUP -v SKIP_BUILD=1,RESUME=1,ALLOW_CONTROL_TEMPLATES=1,PARAMS=work/validation/best_params_template.json tsubame/run_pipeline.sh`.

Variants of the same job (`-v` passes settings to the script):

```bash
qsub -g $GROUP -v SKIP_BUILD=1 tsubame/run_pipeline.sh                    # reuse step 01's outputs (e.g. after 02 failed)
qsub -g $GROUP -v SKIP_BUILD=1,RESUME=1 tsubame/run_pipeline.sh           # also keep the variants 02 already scored
qsub -g $GROUP -v SKIP_BUILD=1,RESUME=1,DRY_RUN=1 tsubame/run_pipeline.sh # reuse 01/02; rebuild and check the submission, no .vcc
```

Always combine `DRY_RUN=1` with `SKIP_BUILD=1,RESUME=1`. On its own it rebuilds every output of 01 and rescores
every variant of 02 (3–7 hours of paid time) before the dry run.

Cost: `node_q` has charge coefficient 0.25, and points = 0.25 × (0.7 × hours used + 0.1 × hours requested). An
8-hour run with 12 hours requested costs about 1.7 points (~110 JPY on campus).

### 5b. Or step by step (to rerun one stage, or to watch it)

Open a `node_q` session and prepare it the same way the batch job does:

```bash
qrsh -g $GROUP -l node_q=1 -l h_rt=8:00:00
cd /gs/bs/$GROUP/$USER/VCC && module purge && module load cuda/12.8.0 && export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="${T4TMPDIR:-$PWD/work/tmp}" && mkdir -p "$TMPDIR" && df -h "$TMPDIR" | tail -1
```

The last line matters for step 3. `vcc prep` writes ~20 GB of temporary files (the whole uncompressed prediction
plus its compressed copy) to `$TMPDIR`, which is `/tmp` by default and may be small on a compute node. The line
points it at the node's large local disk (`$T4TMPDIR`) when the session has one, otherwise at `work/tmp/` on the
group disk. The `df` output should show at least ~25 GB free. (The batch job of §5a does the same by itself.)

| Step | Command | Produces | Rough time / peak RAM on `node_q` (estimates) |
|---|---|---|---|
| 1 | `.venv/bin/python scripts/01_build_sources.py --config configs/v1.yaml` (add `--reuse` to keep outputs that already exist) | `data/processed/k562_gw.npz` and `h1_2025.npz` (response tables), `tss.csv`, `cis.json`; `work/h1_proxy/real.h5ad` (H1 formatted like a VCC context: ~160–220 targets × 400 cells + 18,400 controls), `targets.txt`, `real.kd.csv`. Ends with a summary: panel coverage per source, the share of genes with a start site (should be ≥ 99%), cis effects, the number of proxy targets and how many were left out as too strong (see §6) | 1–2 h (reads the 34 GB of H1 twice) / < 30 GB |
| 2 | `.venv/bin/python scripts/02_validate.py --config configs/v1.yaml --gpu` (add `--resume` to keep variants already scored) | `work/validation/results.csv` (one row per variant); `best_params.json` (the overall winner), `best_params_nb.json` and `best_params_template.json` (the best variant of each generator), each with its gains refitted on all proxy targets; `harness_gpu/` (the scorer's reference bundle and per-variant outputs). An older `results.csv` is renamed `results.<time>.csv` | 2–5 h (scoring reference, ~20 stage-1 variants, ~10 stage-2 variants; §6) / ~50–65 GB (the scorer's documented peak at VCC scale) |
| 3 | `.venv/bin/python scripts/03_make_submission.py --config configs/v1.yaml --params work/validation/best_params_nb.json` (add `--dry-run` to check without packaging) | `work/submission_controls/submission.vcc` (360,000 cells, 3–5 GB), `submission.vcc.sha256` (its checksum), `submission.json` (what produced it). The ~17 GB `submission.h5ad` is deleted once packaging has succeeded (add `--keep-h5ad` to keep it). Prints, per context, how many testable genes the template alone moves for every target (should be a handful), and the `vcc submit` command. Without `--params` it would use `best_params.json`, the overall winner, and stop if that is a template variant | ~1 h / ~35–50 GB (`vcc prep` loads the whole file) |

`--gpu` matters: without it 02 uses the CPU engine, which is much slower and whose scores are not comparable with
the platform's GPU engine.

**Please record** the real wall time and peak memory of each step. The step logs print the time per stage, and
`qacct -j <jobid>` shows `ru_wallclock` and `maxvmem` for a finished batch job.

### 5c. Optional: notebook 02 (the model explained with plots)

After step 1 (and again after step 2, which adds its §9), render the walkthrough notebook unattended. It takes a
few minutes and < 16 GB RAM, so the free queue or a `gpu_1` is enough:

```bash
cd /gs/bs/$GROUP/$USER/VCC && mkdir -p work/runs
.venv/bin/jupytext --sync notebooks/02_effect_model_walkthrough.py      # creates the paired .ipynb
.venv/bin/papermill -k vcc --cwd notebooks notebooks/02_effect_model_walkthrough.ipynb \
    work/runs/02_effect_model_walkthrough.executed.ipynb
```

Its figures are also saved as PNGs in `work/figures/02_effect_model/`. To redo §4–8 of the notebook with the
settings validation chose, prefix the papermill command with `VCC_PARAMS_FROM=best`.

## 6. Reading `work/validation/results.csv`

The end of `02_validate.log` prints the same table ranked by `overall`. To print it again:

```bash
.venv/bin/python -c "import pandas as pd; r = pd.read_csv('work/validation/results.csv'); c = [x for x in ['variant','kind','parent','generator','overall','pds','mse','nmae','fid','reach','jac','beta_mean','gamma_mean','beta_scale','min_called_lfc'] if x in r]; print(r.sort_values('overall', ascending=False)[c].round(3).to_string(index=False))"
```

| Column | Meaning |
|---|---|
| `variant`, `kind`, `parent` | the settings, encoded in the name (e.g. `z0-3.5_null_p-0.05_nb`). `kind` says which part of the run the row comes from: `grid` (stage 1), `stage2` (`parent` names the stage-1 winner it starts from), `ablation` (`abl_template_only`: no target-specific part; `abl_no_cis`: no neighbour prior; diagnostics only, they can't win) or `reference` (`control_paste`: "predict no change" for every target) |
| `generator`, `spread`, `z0`, `null_p`, `beta_scale`, `min_called_lfc`, … | the cell generator (`template` edits copies of real control cells, `nb` samples every count) and the model settings. `beta`/`gamma` in these columns are the starting values; the fitted gains are in `beta_mean`, `gamma_mean` and `gains_per_fold` (one pair per cross-fitting fold) |
| `overall`, `pds`, `mse`, `nmae`, `fid`, `reach`, `jac` | **scaled** scores, exactly like the leaderboard: **0 = the context-mean baseline** (H1's true average response given to every target, an oracle) and **1 = a split-half replicate** of the real data. Higher is better in every scaled column (also for `mse` and `nmae`). `overall` is the mean of the six |
| `raw_pds` … `raw_jac` | the metrics before scaling (lower is better for `raw_mse` and `raw_nmae`) |

**How validation works.**
- **Held-out line.** H1 is never a source here: predictions use K562 only, exactly like the submission.
- **Proxy panel matched to the challenge.** H1's panel responds much more strongly than the challenge's: 16% of H1
  targets change more than 50 genes clearly in K562 (|z| > 4.5 among genes above 5 CPM), while no challenge
  target does (the maximum is 42). Step 01 therefore leaves those strong H1 targets out of the proxy, so the
  settings are chosen in the challenge's regime. H1 targets that K562 never measured are kept.
- **Cross-fitting.** The proxy targets are split into 2 folds. Each fold is predicted with gains fitted on the
  other fold's H1 answers, so no target is scored with gains that saw its own answer. The gains are fitted only on
  genes the scorer can test (control CPM > 5).
- **Two stages.** Stage 1 scores a grid: `z0` ∈ {2.5, 3.5, 4.5} × `null_p` ∈ {0.05, 1.0} × {template without
  spread, template with spread 0.5, nb}, plus the two ablations and `control_paste`. Stage 2 starts from the
  stage-1 winner of each generator and tries larger effect sizes: `beta_scale` ∈ {1, 2, 3} (a multiplier on the
  fitted gain of the target-specific part) × `min_called_lfc` ∈ {0, 0.4} (the smallest predicted change on a
  confident gene). Least squares tends to shrink effects, and the DE metrics punish effects too small to be called
  significant; stage 2 tests whether undoing that pays.
- **Same cells for every variant.** Cells are generated with one random seed per target, so all variants draw the
  same control cells and library sizes and differences between rows come from the settings, not from sampling luck.

**What to compare with.** These are proxy numbers on H1 with ~160–220 targets, not leaderboard numbers. They are
only roughly comparable, but they tell us whether v1 is in the right range:

| Reference | overall | Source |
|---|---|---|
| Leaderboard #1 (Illumina, likely proprietary data) | 0.47 | [brief §5](notes/VCC2026_brief.md) |
| Leaderboard ranks 2–10 | 0.30–0.33 | brief §5 |
| Leaderboard median team | 0.09 | brief §5 |
| Self-reported "transfer public responses" entries | 0.07–0.15 (with 0.00 scaled mse) | brief §5 |
| Synthetic proxy, a pipeline check and *not* biology: controls pasted / true means / denoised true means / true effects halved | −0.58 / 1.07 / 1.17 / 0.68 | notebook 01 §7 |
| `control_paste` row of this file | should be clearly negative | this run |

A plausible first v1 result is **0 to 0.2 overall**. Anything around or above the leaderboard's top ten on the
first try deserves suspicion before celebration.

**Please note in the report** the best `nb` row and the best `template` row (the same as
`best_params_nb.json` and `best_params_template.json`). Their gap tells us what the organizers' answer on control
templates is worth.

**Red flags** (send the files and stop; nothing gets submitted):
- **The log says `WARNING: the best variant does not beat control_paste`**, or every variant is below 0 overall.
  v1 is then worse than the line's average response, usually through `fid` (deeply negative: too few genes called,
  or wrong signs) or `nmae` (fold changes too large).
- **`nmae` near its floor (−6)**, or **`mse` exactly 0 for every variant.** Something produces absurd fold changes
  (e.g. genes with zero total counts), or the prediction carries no information beyond noise.
- **`pds` near 0** for every variant: predicted effects are no more target-specific than a constant.
- **The winner sits at the edge of the grid**: `z0` = 2.5 or 4.5, or `beta_scale` = 3. The real optimum is
  probably outside the grid. Tell us and we extend it.
- **Fitted gains** (`beta_mean`, `gamma_mean`, `gains_per_fold`) that are **negative, above ~3, or very different
  between the two folds.**
- **`abl_template_only` beats every full variant.** The target-specific part then adds noise on H1, which we need
  to know before submitting anything that relies on it.
- **Far fewer proxy targets than expected.** The 01 log says how many passed the knockdown filter and how many
  were left out as too strong; ~160–220 are expected in the proxy.

## 7. The submission file, and handing it over

Script 03 builds the A/B/C submission with the settings in **`work/validation/best_params_nb.json`**: the best
validated NB variant, with its gains refitted on all proxy targets. It uses exactly the model 02 validated (K562 as
the only source, K562's panel-average template). It takes the contexts from the controls bundle's manifest by
name, checks the file with `vcc prep --dry-run`, packages it into **`work/submission_controls/submission.vcc`**,
writes its checksum to `submission.vcc.sha256`, and then deletes the large `.h5ad`. The last lines of
`03_make_submission.log` show the `vcc submit` command.
**Do not run it** unless Daniyal tells you to (option B below).

**Template generator: only after the organizers' written answer.** If they confirm that control cells may be used
as templates, Daniyal will say so, and the template file is built with
`--params work/validation/best_params_template.json --allow-control-templates` (batch job: §5a). Never pass that
flag on your own.

**Daniyal controls every submission.** We have 2 scored submissions per day for the whole team (reset at 00:00 UTC,
one in flight at a time), and the leaderboard ranks our *most recent* entry. Your part:

1. On TSUBAME, check the file against the checksum 03 wrote:
   ```bash
   cd /gs/bs/$GROUP/$USER/VCC/work/submission_controls      # final phase: work/submission_controls_DEF
   sha256sum -c submission.vcc.sha256                       # must print: submission.vcc: OK
   cat submission.vcc.sha256                                # the hash Daniyal will refer to
   ls -lh submission.vcc
   ```
2. Send Daniyal the **small files** through a private channel: `submission.vcc.sha256`, `submission.json`,
   `work/validation/results.csv`, `best_params_nb.json` and the report of §10 (a few KB in total).
3. Then **one** of the two options below. Daniyal decides which, before the run.

`submission.vcc` is **~3–5 GB** (estimate from real control cells). **Never send it as a chat or e-mail
attachment**: chat apps and e-mail reject files this size, and the file must not leave the team.

**Option A: copy the file to Daniyal.** On **your own computer**, copy the files off the cluster with `rsync`, which
resumes an interrupted copy when you run the same command again:

```bash
SRC=<login>@login.t4.gsic.titech.ac.jp:/gs/bs/<group>/<login>/VCC/work/submission_controls   # final: submission_controls_DEF
rsync --partial --progress -e ssh "$SRC/submission.vcc" "$SRC/submission.vcc.sha256" "$SRC/submission.json" .
```

It takes about 1–2 minutes at 50 MB/s and 5–8 minutes at 10 MB/s. Then share `submission.vcc` with Daniyal as a
link from a private cloud folder that **only he** can open, and send him the link privately. He checks the file and
submits it (CONTRIBUTING.md §6) with the command printed at the end of the 03 log, using the local file name:

```bash
shasum -a 256 -c submission.vcc.sha256                  # Linux: sha256sum -c submission.vcc.sha256
vcc submit submission.vcc -m "model-v1-<UTC date>" --wait    # e.g. model-v1-20261017
vcc status <entry-id> --json                            # raw metric values
```

**Option B: you submit from TSUBAME, after Daniyal's go-ahead.** This avoids moving the file twice. You may run
`vcc submit` **only after Daniyal's explicit written go-ahead for that exact file**: his message must name the
file's sha256 (the hash you sent him in step 2) and the leaderboard name to use. It still uses one of the team's
2 daily submissions, so nobody else submits at the same time. With your own key, on a compute node (not a login
node):

```bash
iqrsh -l h_rt=1:00:00
cd /gs/bs/$GROUP/$USER/VCC/work/submission_controls      # final phase: work/submission_controls_DEF
export PATH="$HOME/.local/bin:$PATH"
cat submission.vcc.sha256                                # must equal the hash in Daniyal's go-ahead, character for character
sha256sum -c submission.vcc.sha256                       # must print: submission.vcc: OK
vcc whoami                                               # must name our team
vcc submit submission.vcc -m "<the name from Daniyal's go-ahead>" --wait
```

If the upload is interrupted, run `vcc submit --resume`. Scoring continues on the server even if the session ends
while waiting; `vcc status <entry-id>` shows the result later. Send Daniyal the entry id and the printed scores.

**Names and descriptions are public.** The `-m` name appears on the leaderboard, and other teams may read entry
descriptions. Use a neutral name like `model-v1-<UTC date>` and **no `-d` description**: the variant, the generator
and the proxy scores stay in `submission.json` and [submissions/LOG.md](submissions/LOG.md).

Daniyal records every submission in [submissions/LOG.md](submissions/LOG.md) (UTC time, entry id, commit, scaled
and raw scores). Only the `.vcc` is ever submitted, never the `.h5ad`.

## 8. The final phase (from Oct 22)

The final bundle has **new cell lines (D/E/F) and a new 300-target panel**. There is no leaderboard, and only our
last final entry counts.

```bash
cd /gs/bs/$GROUP/$USER/VCC
vcc datasets list                                    # find the final-phase controls bundle's name
vcc datasets download <final-controls-name> -o controls_DEF.zip && unzip -o controls_DEF.zip -d controls_DEF
ls controls_DEF                                      # a NEW folder: never mix it with controls/
cat controls_DEF/manifest.json                       # must list the contexts D, E, F
```

Then build the D/E/F file in its own folder, so it can't be confused with the A/B/C one:

```bash
qrsh -g $GROUP -l node_q=1 -l h_rt=3:00:00
cd /gs/bs/$GROUP/$USER/VCC && module purge && module load cuda/12.8.0 && export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="${T4TMPDIR:-$PWD/work/tmp}" && mkdir -p "$TMPDIR" && df -h "$TMPDIR" | tail -1
ls work/submission_controls/                         # if a submission.h5ad is still there (--keep-h5ad), delete it now
.venv/bin/python scripts/03_make_submission.py --config configs/v1.yaml \
    --controls-dir controls_DEF --params work/validation/best_params_nb.json
```

- The script reads the contexts, gene axis and target list from `controls_DEF/` and passes them to `vcc prep`
  (vcc 0.2.2 would otherwise assume A,B,C). Contexts are handled by **name**, never by position.
- If the new bundle's gene list differs from the one the response tables were built on, 03 maps the tables onto it
  by gene name and prints how many genes are new or dropped (new genes get "no change" from the sources). It no
  longer stops. Send that line with the report.
- Steps 00–02 don't need rerunning for the new panel. The response tables hold every target the sources measured,
  and the template is recomputed for whichever panel is predicted. Rerun 02 only if we change the model.
- The 03 log reports how many of the new targets have a source (`any source: …/300`). Expect a different coverage.
- As a batch job instead: `qsub -g $GROUP -v SKIP_BUILD=1,RESUME=1,CONTROLS_DIR=controls_DEF tsubame/run_pipeline.sh`.
  It reruns 02 quickly (every variant is already scored). Output goes to `work/submission_controls_DEF/`; the A/B/C
  file in `work/submission_controls/` is left untouched.
- The handover is the same as in §7 (from `work/submission_controls_DEF/`). The final entry is submitted by
  ~Nov 3; the deadline is **Nov 5, 23:59 UTC** (Nov 6, 08:59 JST).

## 9. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| The pre-flight stops the job within seconds | Its message names what is missing: `.venv` (run `bash tsubame/setup_env.sh`), `controls/` (§2), the `vcc` CLI, or a Python module (send the error) |
| `git switch model-v1` says `invalid reference` | The branch has not been pushed yet. Tell Daniyal and wait; don't run anything from `main` |
| `missing input files (run scripts/00_download_data.sh first)` | A download is incomplete: `bash scripts/00_download_data.sh --list`, then rerun the download (§3) |
| The 01 summary shows (or warns) that fewer than 99% of the gene list have a start site (TSS) | The wrong GENCODE release is on disk, or `tss.csv` was built from one (e.g. an older run with release 46). Check that `data/annotation/gencode.v32.basic.annotation.gtf.gz` exists, delete `data/processed/tss.csv` and `data/processed/cis.json`, and rerun 01 with `--reuse` (§5b): it rebuilds only those two |
| Step 03 stops with a message about the template generator or control templates | You ran 03 without `--params`, so it took `best_params.json` (the overall winner), which uses templates. That stop is on purpose. Run step 3 with `--params work/validation/best_params_nb.json` (§5b). Don't add `--allow-control-templates` unless Daniyal tells you the organizers confirmed it |
| Job killed, log ends abruptly, `MemoryError` or `Killed` | Out of memory. `qacct -j <jobid>` shows `maxvmem`. Rerun that step by hand (§5b) on a bigger slice: `qrsh -g $GROUP -l node_f=1 -l h_rt=4:00:00` (4 GPUs, 768 GB RAM; 4× the cost per hour). Tell us which step and how much memory it reached |
| `Exceeded h_rt`, or the job stops at the 12-hour limit | Finished outputs stay on disk. If it stopped in 02 or 03: `qsub -g $GROUP -v SKIP_BUILD=1,RESUME=1 tsubame/run_pipeline.sh` continues with the variants not yet scored. If it stopped in 01: run 01 by hand with `--reuse` (§5b), then the job with `SKIP_BUILD=1`. A longer limit: `qsub -g $GROUP -l h_rt=24:00:00 tsubame/run_pipeline.sh` (24 h is the maximum) |
| `WARNING: no GPU visible` in the job log, or gpudge / CUDA errors (`no CUDA-capable device`, `CUDA driver version is insufficient`, cupy import errors) in `02_validate.log` | The step ran without a GPU or without the CUDA module. Batch jobs from `run_pipeline.sh` load it themselves; by hand, run `module load cuda/12.8.0` first (§5b). Test on a GPU node: `.venv/bin/python -c "from cell_eval2.de_compute import _resolve_backend; print(_resolve_backend('auto'))"` must name a gpudge backend. If setup ran on a node without the module, rerun `bash tsubame/setup_env.sh` on a GPU node |
| Scores look wrong after the proxy was rebuilt with **different settings** (e.g. `proxy:` values in the config, or other H1 files) | 02 should notice a new proxy, rebuild the scorer's reference bundle and set aside rows scored on the old proxy (even with `--resume`). If in doubt, delete `work/validation/harness_gpu/` and rerun 02 without `--resume`. Rebuilding with the same settings gives the same proxy, so this is only needed after a change |
| `No space left on device` while `vcc prep` packages (end of 03) | Its temporary files went to a small `/tmp`. Set `TMPDIR` as in §5b and rerun step 3. Delete leftovers: `rm -rf work/tmp/*` |
| `vcc prep` fails in 03 for another reason | Read its message in `03_make_submission.log`: wrong contexts (final phase: did you pass `--controls-dir controls_DEF`?), gene order or count, a target missing or extra, not 400 cells per target, a cell above 1,000,000 counts, non-integer counts, or too many stored values. Format failures cost no submission slot. Send the log |
| `vcc: command not found` | `export PATH="$HOME/.local/bin:$PATH"` |
| `vcc version` is not 0.2.2 | `uv tool install --force 'vcc-cli==0.2.2'`. The pipeline relies on 0.2.2's `prep` flags and defaults |
| `vcc` says you are not logged in, or 401 | `vcc login --store-plaintext` again (generating a new key revokes the old one) |
| `git clone` asks for a password or says `Permission denied (publickey)` | The GitHub key on TSUBAME is missing or not registered (§1). `ssh -T git@github.com` must greet you |
| `Disk quota exceeded` | `t4-user-info disk group`. Delete `work/smoke/`, `work/tmp/`, old `work/runs/`, `work/validation/results.*.csv` you don't need, a kept `submission.h5ad`, and leftovers of failed runs in `work/submission_*/` (e.g. `submission_chunks/`). Ask the group owner for more group disk |
| A download stalls or fails | Rerun the same command; it resumes. If one source keeps failing, send the output: the config lists a fallback URL for the Replogle files |
| A copy of `submission.vcc` breaks off | Run the same `rsync` command again (§7): it continues from the partial file |
| Anything else | Don't edit the code on the cluster. Send the log and the command you ran (§10) |

## 10. What to send back

To Daniyal, after the run (or after a failure, whatever exists):

1. `work/validation/results.csv`, `best_params.json`, `best_params_nb.json` and `best_params_template.json`.
2. The logs: `vcc_pipeline.o<jobid>` and the folder `work/logs/pipeline.<jobid>/` (or the terminal output of each
   step if you ran them by hand), plus `work/test_report.txt` from §4.
3. `work/h1_proxy/targets.txt` and `work/h1_proxy/real.kd.csv` (small: which H1 targets made it into the proxy).
4. A short note:
   - the commit (`git log -1 --oneline`) and the CLI version (`vcc version`);
   - the job id(s) and node type;
   - the wall time and peak memory of each step (§5b);
   - the disk used (`du -sh data work`) and the free space `TMPDIR` had (the `df` line of §5b);
   - from the 01 log: the number of proxy targets, how many were left out as too strong, and the share of genes
     with a start site;
   - the best `nb` and the best `template` overall score (§6);
   - from the 03 log: the template sanity line of each context, and the `any source: …/300` coverage line;
   - anything you had to do differently from this runbook.
5. If you ran notebook 02: `work/runs/02_effect_model_walkthrough.executed.ipynb` and the PNGs in
   `work/figures/02_effect_model/`.
6. Only if script 03 finished: `submission.vcc.sha256` and `submission.json`, then the `.vcc` itself by option A
   or B of §7.

**Never send** the downloaded data, the controls, `.h5ad` files, or anything that contains your API key. **Never
attach** `submission.vcc` to a chat message or an e-mail.
