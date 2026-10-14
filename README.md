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

| Piece | Status |
|---|---|
| Local evaluation harness (official scorer on proxy data with known answers) | ✅ [notebooks/01](notebooks/01_local_evaluation_harness.ipynb), tested on a Mac with the synthetic proxy; H1 proxy code written, not yet run |
| Cell generator (predicted profile → realistic count cells) | 🔧 built; calibration on real data (H1 proxy) pending; rules question open (it reuses control cells as templates) |
| TSUBAME 4.0 setup + batch scripts | ✅ [tsubame/](tsubame/), untested on the cluster yet |
| Effect model (predict each target's response in a new line) | 🔜 next: transfer from public screens ([METHOD.md](METHOD.md)) |
| First leaderboard submission | 🔜 after the first effect model |

Our approach, and why LangPert is not at its core: **[METHOD.md](METHOD.md)**.
What every team member has to do: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Quick start (Mac or Linux, CPU)

```bash
git clone git@github.com:danialsj77/VCC.git && cd VCC
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name vcc --display-name "VCC (project .venv)"
uv tool install vcc-cli && vcc login                        # paste your own API key at the hidden prompt
vcc datasets download controls -d . && unzip -o vcc_2026_controls.zip -d controls
.venv/bin/jupyter lab notebooks/01_local_evaluation_harness.ipynb   # kernel "VCC (project .venv)" → Run All
```

## Repository layout

```
README.md                 this file
CONTRIBUTING.md           team guide: onboarding, workflow, submission protocol, rules
METHOD.md                 our method and the reasoning behind it
requirements.txt          pinned Python packages (CPU machines)
notes/
  VCC2026_brief.md        the challenge: data, submission format, exact scoring, leaderboard, strategy
  LangPert_notes.md       reading notes on LangPert and how it fits (or doesn't) this challenge
notebooks/
  01_local_evaluation_harness.ipynb   local "practice exam" (+ paired .py for code review)
tsubame/
  setup_env.sh            one-time GPU environment setup on TSUBAME 4.0
  run_notebook.sh         batch job running notebook 01 at full size
controls/   (not in git)  the challenge's control-cell bundle, downloaded with `vcc`
data/       (not in git)  public datasets (e.g. Arc's 2025 H1 data)
work/       (not in git)  outputs of the notebooks; work/experiments.csv logs every scored test
submissions/LOG.md        one row per leaderboard / final submission
```
