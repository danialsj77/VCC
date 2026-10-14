# Team guide: what every member has to do

Team lead: **Daniyal Sajedi Hosseini** (owner of this repo and of our challenge team).
Read [README.md](README.md) first, then this guide, then [METHOD.md](METHOD.md).

---

## 1. Join the challenge (once, ~30 min)

1. **Check you are eligible** ([Rules](https://virtualcellchallenge.org/rules)). You must be 18+. **Not eligible:**
   current Arc employees or interns; current employees or interns of a sponsor (NVIDIA, 10x Genomics, Ultima
   Genomics); former Arc employees who left after April 2026; Arc service providers, consultants or contractors;
   students in Arc Core Investigator labs; students or employees in Arc *Innovation* Investigator labs (these PIs
   are at other universities, so check your lab); students advised by an Arc employee in or after January 2026;
   residents of Belarus, Cuba, Iran, North Korea, Russia, Crimea, Luhansk or Donetsk; anyone on a US
   restricted-party list. One ineligible member puts the whole team at risk. If in doubt, ask the lead first.
2. **Register** at <https://virtualcellchallenge.org> (Google, GitHub, Microsoft or e-mail link; use **one e-mail
   address** for all sign-in methods, or you create a duplicate account). Registration asks for your legal name,
   employer and university, and a one-time **selfie check** (Persona) to stop duplicate accounts.
3. **Participate in the same capacity as the rest of the team.** The rules don't allow one team to mix members
   participating *in a personal capacity* with members participating *as employees or university representatives*,
   and an employee team must come from a single organization (university members may come from several
   universities). Agree on this with the team lead **before** registering.
4. **Join our team**: ask Daniyal for an invite from the challenge web app and accept it. Teams have at most
   8 members, and **you can be on only one team**. Team changes go through help@virtualcellchallenge.org.
5. **Create your own API key**: web app → Account → **Credentials**. It is shown **once**; generating a new one
   revokes the old one. It is personal. **Never** paste it into code, notebooks, chat, issues, or any file in this
   repo; store it only with `vcc login` (below) or in a password manager.

## 2. Set up your computer (once, ~15 min)

Requirements: an Apple-silicon Mac (M1 or newer) or x86_64/ARM Linux (Intel Macs can't install the scorer's
dependencies; use TSUBAME), [uv](https://docs.astral.sh/uv/), Git with SSH access to this repo, ~25 GB free disk.
A 16–24 GB laptop runs the small notebooks; full-size work runs on TSUBAME (§4).

```bash
git clone git@github.com:danialsj77/VCC.git && cd VCC

# Python 3.12 environment with Arc's official scorer (cell-eval2) and the notebook tools
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name vcc --display-name "VCC (project .venv)"

# The challenge CLI, then log in with YOUR key
uv tool install vcc-cli
vcc login                                  # paste the key at the hidden prompt, press Enter; stored in your OS keychain
                                           # (macOS alternative: copy the key, then `pbpaste | vcc login --token-stdin`)
vcc whoami                                 # should say: team …, status: ready to submit

# The challenge's control cells (not in git: ~630 MB)
vcc datasets download controls -d .
unzip -o vcc_2026_controls.zip -d controls
```

## 3. Check that everything works (~10 min)

Open `notebooks/01_local_evaluation_harness.ipynb` with `.venv/bin/jupyter lab`, select the kernel
**VCC (project .venv)**, then *Kernel → Restart Kernel and Run All Cells*. It should finish without errors in about
10 minutes (~14 GB RAM peak) and end with `✓ dry run — no .vcc written`. The score table in §7 should look roughly like
this (synthetic proxy; your numbers will match closely, not exactly, if package versions differ):

| prediction | overall | fid | jac |
|---|---|---|---|
| control_paste | −0.58 | −1.84 | −0.02 |
| truth | 1.07 | 1.30 | 0.30 |
| truth_denoised | 1.17 | 1.24 | 1.10 |

If it fails, open an issue with the error and the last lines of `work/.../log_*.txt`.

## 4. TSUBAME 4.0 (members with Science Tokyo accounts)

Full-size experiments (300 targets, the real-biology H1 proxy, big public datasets, model training) run on TSUBAME
4.0 GPU nodes. Everything is in **notebook 01, §10**. In short:

1. Get a TSUBAME account and an ed25519 SSH key registered in the TSUBAME portal; ask Daniyal to add you to our
   TSUBAME group (`tga-…`), which pays for jobs with points.
2. So that TSUBAME can clone from GitHub, create a **second** key there and add its `.pub` to your GitHub account:
   `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github`, then add to `~/.ssh/config`:
   `Host github.com` / `  IdentityFile ~/.ssh/id_ed25519_github`.
3. On a compute node (`iqrsh -l h_rt=1:00:00`), make **your own** clone (shared clones break each other's `.venv`
   and outputs), set it up, log in to vcc (there is no OS keychain on the cluster, hence `--store-plaintext`, a
   private 0600 file), and get the controls:
   ```bash
   mkdir -p /gs/bs/<group>/$USER && cd /gs/bs/<group>/$USER
   git clone git@github.com:danialsj77/VCC.git && cd VCC && bash tsubame/setup_env.sh
   vcc login --store-plaintext
   vcc datasets download controls -d . && unzip -o vcc_2026_controls.zip -d controls
   ```
   Large public datasets are downloaded **once** into `/gs/bs/<group>/shared/data` and linked into each clone
   (`ln -s /gs/bs/<group>/shared/data data`).
4. Batch run: `qsub -g <group> -v VCC_PROXY=h1_2025 tsubame/run_notebook.sh` (executed notebook lands in `work/runs/`).

Rules of thumb: never run heavy work or Jupyter on login nodes; delete idle Open OnDemand sessions (they bill while
idle); keep data on the group disk (`/gs/bs/<group>/…`), not in your 125 GiB home.

## 5. How we work

**Pick a task.** Open tasks are GitHub issues (see §8 for the current list). Assign yourself before starting.

**Branches and pull requests.**
- Never push to `main` directly. Work on a branch (`git switch -c <your-name>/<topic>`), open a pull request, and
  get one teammate's review. `main` must always run.
- Keep commits small and messages descriptive. Never commit data, keys, `.h5ad`/`.vcc` files or `work/` outputs
  (`.gitignore` blocks most; check `git status` before every commit).

**Notebooks.**
- One notebook per stage, numbered: `01_…`, `02_…`. Each is paired with a `.py` copy by *jupytext*; commit **both**
  (reviewers read the `.py`). If you edit the `.py` outside Jupyter, run `jupytext --sync notebooks/<name>.py`.
- Comment generously: say *why*, not just *what*. Someone else will run your notebook at 2 am before a deadline.
- Read settings from §1-style constants or environment variables; no hard-coded personal paths.

**Experiments must be comparable.**
- Score every candidate with the harness (notebook 01's `score()`), and paste the scores table into your pull
  request, stating the **proxy** (synthetic / h1_2025 / which held-out line) and **mode** (CPU/GPU engine).
- Only compare numbers from the same proxy and mode. Synthetic-proxy numbers test code; they don't choose models.
- **Never train on the line you evaluate on** (e.g. don't train on H1 and then score on the H1 proxy).
- Fix random seeds; record dataset versions, commit hashes and settings.

**Data.** Download public datasets into `data/<dataset>/` (not in git). Record the source URL, size, license and any
processing in `notes/datasets.md` (create it if missing) so others can reproduce. Respect licenses: some datasets
(e.g. X-Atlas, CC BY-NC-SA) and Arc's State/Stack/PIE checkpoints are **non-commercial only**; check before using them
in a prize entry.

## 6. Submission protocol (leaderboard and final)

We have **2 scored submissions per day for the whole team** (reset 00:00 UTC = 09:00 JST), one in flight at a time.
Only the team lead submits, so we never collide or waste slots.

1. A candidate must **beat our current best on the local harness** (H1 proxy and/or held-out public lines).
2. Open an issue titled `Submission: <model name>` with the commit hash, how to build the file, and the harness
   scores.
3. The submitter builds the file on a big machine (TSUBAME `node_q`, ~50 GB RAM peak), packages it, and submits
   **the `.vcc`** (never the `.h5ad`: `vcc submit`'s own built-in check only knows contexts A,B,C):
   ```bash
   vcc prep sub.h5ad -g controls/gene_names.csv --perts controls/pert_counts.csv --contexts A,B,C -o sub.vcc
   vcc submit sub.vcc -m "<model name>" -d "<what changed>" --wait
   ```
   then records it in [submissions/LOG.md](submissions/LOG.md) (UTC time, entry id, commit, raw + scaled scores
   from `vcc status <id> --json`).
4. Never submit placeholders or untested files. The leaderboard ranks our **most recent** submission, not our best.

**Final phase (Oct 22 → Nov 5, 23:59 UTC = Nov 6, 08:59 JST).** New lines **and** new targets; no feedback.
- Unzip the D/E/F bundle into **`controls_DEF/`** (git ignores it); in notebooks set
  `SUBMIT_DIR = PROJECT / "controls_DEF"`; run `vcc prep sub.h5ad -g controls_DEF/gene_names.csv
  --perts controls_DEF/pert_counts.csv --contexts D,E,F -o sub.vcc` (vcc 0.2.2 defaults to A,B,C), then submit the `.vcc`.
- Map context labels by **name**, never by position: a swapped label looks like a weak model and nobody would notice.
- Submit a solid entry **early** (by ~Nov 3) and only replace it with something better-validated; **only our last
  final entry counts**.

## 7. Rules everyone must follow

- **Machine-learning predictions only.** We may train on any data we have rights to (public papers, H1 2025, other
  screens), but we must not submit measured lab/literature results, or use them to *revise* model outputs.
  Measured data enters **only as training data for a fitted model**. No component may pass measured values through
  as predictions: no copied or hand-weighted averages of fold changes from papers or public screens. Non-learned
  components (the cell generator, thresholds) may shape model outputs and must be disclosed.
- **Open question: the challenge's control cells.** Arc calls them "model inputs only — do not copy them into your
  prediction". Our generator uses them as templates (§3 of METHOD.md). Until the organizers confirm this is
  acceptable (task #7), we also build a fallback generator that doesn't (task #8).
- **Finalists must publish a description** of: training datasets, data processing and model inputs, models used or
  adapted (incl. public ones), architecture of non-public models, any non-learned components blended with model
  outputs, training/adaptation/inference phases, and (optionally) compute. Keep this information in
  [METHOD.md](METHOD.md) as you go.
- **Don't share** models, results, strategies or this repo with anyone outside the team, and don't try to access
  other teams' work. Social-media posts about our experience are fine; posting our method or numbers is not (until
  the challenge ends).
- **Exploits are off-limits.** If you find a scoring quirk (e.g. the known MSE "layout" residual, issues #294/#348 in
  `cell-eval2`), report it in an issue; don't use it. Arc can disqualify or re-score suspicious entries.
- Questions about the rules go to the team lead, who asks help@virtualcellchallenge.org when needed.

## 8. Open tasks (October 2026)

| # | Task | Needs | Size |
|---|---|---|---|
| 1 | **Target coverage census**: which of the 300 targets were measured (with a real response, enough cells) in K562 genome-wide, RPE1, HepG2, Jurkat, CD4 T cells, the full X-Atlas library, H1-2025 | laptop | ½ day |
| 2 | **H1 proxy run** on TSUBAME; tune the generator's `spread` (on the DE metrics only) and the "no change" confidence threshold on real cells | TSUBAME | 1 day |
| 3 | **Public data pipeline**: download + format K562 genome-wide (bulk files first), CD4 T `DE_stats`, Replogle K562/RPE1, Nadig HepG2/Jurkat onto the VCC gene axis; leave-one-line-out proxies (H1 and K562-GW at full VCC format, others at pseudobulk level) | TSUBAME | 2–3 days |
| 4 | **Effect model v1** (notebook 02): a *learned* cross-line transfer model trained on public screens ([METHOD.md](METHOD.md) §3) | 1–3 | 3–4 days |
| 5 | **Context template model**: predict a line's generic perturbation response from its control cells | 3 | 2–3 days |
| 6 | **Response-strength / null classifier**: which targets do (almost) nothing in a given line | 1, 3 | 2 days |
| 7 | **Ask the organizers** (urgent, before our first submission): may control cells serve as generator templates? Is a learned cross-line transfer of a target's public responses an ML prediction? Plus the other questions in brief §6.5 | lead | ½ hour |
| 8 | **Fallback generator** that does not reuse control cells: fitted negative-binomial per gene, library sizes drawn from the controls; compare with the template generator on the harness | laptop | 1 day |
