# VCC 2026: our team's workspace

Code, notes and tools for the [Virtual Cell Challenge 2026](https://virtualcellchallenge.org) (Arc Institute).

> **This repository is private. Keep it that way until the challenge ends** (winners announced late Nov 2026).
> The rules forbid sharing models, results or strategies with other teams, and the challenge data may only be
> used by registered participants. Never commit data or API keys (`.gitignore` blocks the usual suspects).

## The challenge in five lines

- **Task.** Predict how cells respond to 300 CRISPRi gene knockdowns in cell lines the model has never seen,
  given only each line's unperturbed (non-targeting control) cells. There is no training set; any public or
  proprietary data we have rights to may be used for training.
- **We submit cells, not summaries.** For each of 3 anonymized cell lines we submit 400 cells per target as
  raw integer counts (360,000 cells), as one `.h5ad` → `vcc prep` → `.vcc` → `vcc submit`.
- **Scoring.** Six metrics (pds, mse, nmae, fid, reach, jac) from Arc's `cell-eval2` scorer, each scaled so
  0 = the line's true average response and 1 = a replicate experiment, then averaged.
- **Timeline.** Validation (lines A/B/C, live leaderboard) until **Oct 22**. Final test (new lines D/E/F,
  **new 300 targets**, no leaderboard) **Oct 22 → Nov 5, 23:59 UTC**. Only our **last** final entry counts.
- **Limits.** 2 scored submissions per team per day (reset 00:00 UTC), one in flight at a time.

Full details, with sources: [notes/VCC2026_brief.md](notes/VCC2026_brief.md).

## Where we are

✅ built and tested · 🔧 built, not yet validated on real data · 🔜 planned

| Piece | Status |
|---|---|
| Local evaluation harness (official scorer on proxy data with known answers) | ✅ [notebooks/01](notebooks/01_local_evaluation_harness.ipynb), tested on a Mac with the synthetic proxy; production version `vccpipe/harness.py`. 🔧 H1 proxy (`vccpipe/proxy.py`), strength-matched to the VCC panel: written, not yet run on the real files |
| Cell generator (predicted profile → realistic count cells) | 🔧 `vccpipe/generator.py`: depth-aware negative-binomial generator (**the default for every submission**) and control-cell templates (used only if the organizers confirm in writing that templates are allowed); calibration on real cells (H1 proxy) pending |
| Public data pipeline (K562 genome-wide, H1 2025, GENCODE v32 TSS → response tables on the VCC gene axis) | 🔧 `scripts/00_download_data.sh`, `scripts/01_build_sources.py`; tested on fake data only |
| **Effect model v1** (K562 transfer: template-removed responses gated by z and rescaled by gains fitted on H1, K562 panel-mean template, CRISPRi neighbour prior) | 🔧 **built and reviewed; untested on real data.** First real run on TSUBAME pending → [RUNBOOK.md](RUNBOOK.md). Its rules status is pending the organizers' answer (below). Summary: [METHOD.md §3 "What v1 implements"](METHOD.md#what-v1-implements) |
| Questions to the organizers (control-cell templates; is v1-style transfer an ML prediction; licences) | 🔜 e-mail ready to send: [notes/organizer_email_draft.md](notes/organizer_email_draft.md) (task #7) |
| TSUBAME 4.0 setup + batch scripts | 🔧 [tsubame/](tsubame/) (`setup_env.sh`, `run_notebook.sh`, `run_pipeline.sh`), not yet run on the cluster |
| First leaderboard submission | 🔜 after the first cluster run of v1 (task #9), with the NB generator |
| Learned transfer model (v2: per-bin gains, then a regression over source responses and control-profile features) | 🔜 planned upgrade ([METHOD.md §3](METHOD.md#what-v1-implements)) |

**Quick links**
- **Run model v1 on TSUBAME, step by step: [RUNBOOK.md](RUNBOOK.md)** (for Ken, or whoever runs the cluster job)
- **Rules questions for the organizers, ready to send: [notes/organizer_email_draft.md](notes/organizer_email_draft.md)**
- Our approach, and why LangPert is not at its core: [METHOD.md](METHOD.md)
- Module interfaces of v1: [docs/DESIGN_v1.md](docs/DESIGN_v1.md)
- What every team member has to do, open tasks: [CONTRIBUTING.md](CONTRIBUTING.md) (§8)
- The challenge itself, exact scoring, leaderboard: [notes/VCC2026_brief.md](notes/VCC2026_brief.md)
- Submission history: [submissions/LOG.md](submissions/LOG.md)

## Quick start (Mac or Linux, CPU)

```bash
git clone git@github.com:danialsj77/VCC.git && cd VCC
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name vcc --display-name "VCC (project .venv)"
uv tool install 'vcc-cli==0.2.2' && vcc login             # paste your own API key at the hidden prompt
vcc datasets download controls -o controls.zip && unzip -o controls.zip -d controls
.venv/bin/jupyter lab notebooks/01_local_evaluation_harness.ipynb   # kernel "VCC (project .venv)" → Run All
```

## Repository layout

```
README.md                 this file
RUNBOOK.md                how to run model v1 on TSUBAME, copy-paste ready (download → validate → submission file)
CONTRIBUTING.md           team guide: onboarding, workflow, submission protocol, rules, open tasks
METHOD.md                 our method and the reasoning behind it
requirements.txt          pinned Python packages (CPU machines)
docs/
  DESIGN_v1.md            interfaces between the v1 modules and the data flow between the scripts
configs/
  v1.yaml                 paths, data URLs, model defaults and the validation grid for v1
vccpipe/                  the pipeline as a Python package (production versions of the notebook code)
  common.py               constants and expression helpers (CPM, means, fold changes)
  sources.py              public screens → response tables (ResponseSet) on the VCC gene axis
  cis.py                  CRISPRi neighbour (cis) prior from TSS distances
  model.py                effect model v1 (ModelParams, EffectModel, fit_gains)
  proxy.py                Arc's 2025 H1 data formatted like a VCC context (for scoring)
  generator.py            predicted profile → integer-count cells (NB, the default; control templates)
  harness.py              official scorer (cell-eval2, vcc2026 profile) on a proxy
  submission.py           build, check and package the submission file
scripts/
  00_download_data.sh     public data: Replogle bulk files, GENCODE v32, H1 2025 (~35 GB)
  01_build_sources.py     response tables, TSS table, cis model, H1 proxy
  02_validate.py          grid of model settings scored on held-out H1 targets → results.csv, best_params*.json
  03_make_submission.py   submission .h5ad → .vcc (via vcc prep); NB generator unless templates are allowed
tests/                    laptop tests on small fake data (tests/fakes.py builds the fakes)
notes/
  VCC2026_brief.md        the challenge: data, submission format, exact scoring, leaderboard, strategy
  LangPert_notes.md       reading notes on LangPert and how it fits (or doesn't) this challenge
  organizer_email_draft.md  rules questions for the organizers (task #7), ready for the lead to send
notebooks/
  01_local_evaluation_harness.ipynb   local "practice exam" (+ paired .py for code review)
  02_effect_model_walkthrough.py      model v1 explained with plots, on the real response tables (run after script 01)
tsubame/
  setup_env.sh            one-time GPU environment setup on TSUBAME 4.0
  run_notebook.sh         batch job running notebook 01 at full size
  run_pipeline.sh         batch job running scripts 01–03 (see RUNBOOK.md)
controls/   (not in git)  the challenge's control-cell bundle, downloaded with `vcc` (controls_DEF/ in the final phase)
data/       (not in git)  public datasets (data/replogle, data/h1_2025, data/annotation) and data/processed/ tables
work/       (not in git)  outputs: work/validation/, work/submission_<bundle>/, work/experiments.csv logs every scored test
submissions/LOG.md        one row per leaderboard / final submission
```
