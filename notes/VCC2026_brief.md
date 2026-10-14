# Virtual Cell Challenge 2026 — Reference Brief

Compiled 2026-10-13. Sources: the official site (Evaluation, Data, FAQ, Rules rev. 2026-09-16, Leaderboard, 2025 page), the vcc CLI wiki and CLI reference, the `cell-eval2` scorer repo (commit `0ce25cf`, v0.18.0, 2026-10-01) including the `vcc2026` metric spec, config, code and tests, Arc's State, Stack and PIE repos and model cards, and the literature.
Tags: **[code]** = verified in scorer source or tests · **[spec]** = official spec or site · **[inferred]** = our inference · **[self-reported]** = community claim, not verified.

---

## 1. Timeline and where we stand

| Date (2026) | Event |
|---|---|
| Aug 20 | Validation phase opened (contexts A/B/C, live leaderboard) |
| **Today, Oct 13** | Validation still open: ~9 days, so up to ~18 scored submissions left (2 per day, reset at 00:00 UTC) |
| **Oct 22** | Final test released: control profiles for 3 **new** cell lines (D/E/F) and a **new 300-gene panel** |
| Oct 22 – Nov 5 | Final entries: 2 per day, **no leaderboard shown**, final scores revealed only when winners are announced |
| **Nov 5, 23:59 UTC** | Final deadline. **Only the last final entry counts** for prizes |
| Late Nov | Winners announced. Prizes: $100k / $50k / $25k, half cash and half NVIDIA Brev credits |

Consequences:
- The final phase is effectively **blind**, with different cell lines and different targets.
- Ranking is decided only by D/E/F. We need a **local evaluation harness** we trust more than the validation leaderboard.
- The validation leaderboard ranks each team's **most recent** submission, not its best.

Account status: the CLI (`vcc-cli` 0.2.2) is installed and logged in, with the token in the macOS Keychain. The team is currently a single member. Teams can have up to 8 members and each person can be on only one team.

## 2. Task and data

- **Zero-shot.** Predict the CRISPRi knockdown responses (10x Flex scRNA-seq) of 300 target genes in each of 3 anonymized cell lines. The only per-context input is non-targeting control (NTC) cells. There is no Challenge training set.
- Arc's six lines (3 validation, 3 final) are stem, hTERT-immortalized and cancer lines from different tissues, all different from the 2025 H1 line. Arc used the same sgRNAs in every line and kept only perturbations with ≥80% median on-target knockdown. The challenge set is subsampled to 400 cells per perturbation and a median of ~20k UMI. [spec + Arc "Behind the data 2026"]
- **What we have locally** (`controls/`; checked by us):
  - `context_{A,B,C}.h5ad`: 18,400 NTC cells each (46 guides × 400, `obs.ntc_id`), raw integer counts, CSR float32, 18,533 genes in `gene_names.csv` order.
  - Median UMI per cell ≈ 20.0k (5th–95th percentile ~5–7k to ~33–39k). About 5.5–6.1k genes detected per cell.
  - Genes passing the DE filter (control mean CPM > 5, where CPM = counts per million): **9,929 / 9,626 / 10,124** for A/B/C. Between 2.3k and 2.9k genes are all-zero in controls.
  - All 300 targets are on the gene axis. Targets that pass the 5 CPM filter: 299 / 295 / 293.
  - `manifest.json`: panel `vcc2026-val-1`, `ground_truth_cells` 138,400 = 300×400 + 18,400. This strongly implies **the scoring control pool is the same 18,400 control cells we were given** [inferred; the code can't prove cell identity].
  - Note that `gene_names.csv` has a `gene_name` header row.
- **Target coverage in public data** (data census 2026-10-13):
  - K562 genome-wide: 272/300 [verified].
  - CD4 T cells: 294/300 listed in `perturbed_gene_name` (cells per target not checked).
  - H1-2025: 25/300 [verified].
  - X-Atlas/Orion: 148/300 in PIE's preprocessed subset; full-library coverage not yet checked.
  - Jiang: 7/300.
  - K562-essential, RPE1, HepG2, Jurkat: 0/300 [verified].
  - The targets look mostly **non-essential**, so effects are probably subtle.
- **The final panel is NEW.** Anything tuned to the specific A/B/C target genes will not carry over. Our method must generalize across *both* targets and contexts.

## 3. Submission format and CLI workflow

One `.h5ad` covering all three contexts → `vcc prep` → `.vcc` → `vcc submit`.

| Requirement | Value |
|---|---|
| `.obs` | `target_gene` (gene **symbol**, e.g. `ADNP`, never `ADNP-1`) and `context` (`A/B/C`; `D/E/F` in the final phase) |
| Perturbations | exactly those in `pert_counts.csv`: no extras, none missing, the same set in every context |
| Cells | exactly **400 per perturbation per context**, so 360,000 rows in total |
| Genes | exactly 18,533, in `gene_names.csv` order |
| `.X` | **raw counts**: non-negative, integer, finite. No `normalize_total`, no `log1p` |
| Per-cell total | ≤ 1,000,000 |
| Density | ≤ 4.75e9 stored entries (~13.2k per cell). **Store as CSR with no explicit zeros.** A dense matrix (6.67e9) is over the cap on its own |
| Controls | **no non-targeting rows**. The platform appends the real controls itself |

```bash
vcc datasets list
vcc datasets download controls -d <dir>
vcc prep pred.h5ad -g gene_names.csv --perts pert_counts.csv -o pred.vcc --dry-run   # validate only
vcc prep pred.h5ad -g gene_names.csv --perts pert_counts.csv -o pred.vcc
vcc submit pred.vcc -m "<model name>" -d "<description>" --wait
vcc status <entry-id> --json        # raw (unscaled) metric values are only shown here
vcc cancel [<entry-id>]             # frees the single in-flight slot; doesn't count against the daily limit
```

**Gotchas that cost points or slots:**
- `-p` means `--pert-col` (a column name) in `prep`, but `--perts` (the list file) in `sample`.
- In vcc-cli 0.2.2, `prep --contexts` defaults to **`A,B,C`** (verified in the installed code; it does not read `manifest.json`). In the final phase, pass `--contexts D,E,F`, and point `-g`/`--perts` at the D/E/F bundle unpacked into a **fresh directory**. (`vcc submit` with a raw .h5ad defaults to the server's current phase.)
- **A context swap (A↔B) is not detected by prep.** It quietly drives every metric toward chance and looks like a weak model. Join labels by filename, never by position, and assert them before submitting.
- Rounding mean profiles with `np.rint` zeroes out low-expressed genes. That produces log2FC ≈ −30 and pushes NMAE to its −6 floor. **Sample** counts instead.
- Only submissions that reach the scoring step count against the 2 per day. Format rejections and HTTP 409s are free.
- Upload an interrupted submission again with `--resume`. The CLI checks the daily limit before uploading.

## 4. Scoring, exactly (vcc2026 profile, rule_version 3, frozen since 2026-08-19)

### 4.1 Common machinery [code]
- **Controls.** Every metric uses the **real** control cells (`control_source: real`), the full 18,400 with no downsampling. This applies to the prediction's DE test, the reference's DE test, and the PDS/MSE control profile. Anything we would supply as controls is ignored.
- **Pseudobulk profile** (used by PDS and MSE): sum raw counts over a group's cells, then `log1p(5e4 · P_g / Σ_g P_g)` with the natural log.
- **DE test** (used by the four DE metrics):
  - Per gene, a two-sided Mann–Whitney/Wilcoxon test of the 400 predicted cells against the 18,400 real controls, on per-cell CPM. The official GPU engine is gpudge: midranks, tie-corrected variance, continuity correction 0.5, normal approximation, **no p-value floor**.
  - **Gene filter:** keep a gene only if its real-control mean CPM is strictly above 5. The filter depends on the reference alone, so the gene set is identical for all perturbations and for both sides.
  - Then BH-FDR within each perturbation (~9.6–10.1k tests); a gene is significant when p_adj < 0.05. There is no fold-change floor.
  - `lfc = log2((mean_pert_CPM + 1e-9)/(mean_ctrl_CPM + 1e-9))`, where each mean is the arithmetic mean of **per-cell** CPM (normalize first, then average). The prediction's LFC is computed against the **real** control mean.
- **Target exclusion.**
  - The perturbed gene's own row is excluded from all six metrics.
  - PDS removes **all 300 panel targets** from the feature space.
  - In the DE tables, the on-target gene still counts in the BH family; it is dropped only when the metrics are computed.

### 4.2 The six metrics

| Leaderboard name | What it is (as implemented) | Aggregate |
|---|---|---|
| **pds** | For each perturbation: the cosine distance between the predicted effect (pred profile − real control profile) and every real effect, with all 300 target genes removed. `PDS_p = 1 − k_p/299`, where k_p is the midrank of the true match. A constant or pasted-control prediction scores exactly 0.5. | mean over 300 |
| **mse** | `Σ_p N_p / Σ_p D_p`. N_p = (‖b̂_p − b_p‖² − ρ·min(Ĉ_p, C_p) − C_p)/G_p; D_p = (‖b_p − b_ctrl‖² − C_p − C_ctrl)/G_p. The C terms are delete-one-cell jackknife noise corrections. ρ = min(1, B/claim) caps the correction by the prediction's own spread across perturbations. Only each row's own target gene is excluded. Raw 1 means no skill. | **ratio of sums**: the strongest perturbations dominate |
| **nmae** | Over genes the reference calls significant (with finite LFC; perturbations with fewer than 10 such genes are skipped): Σ\|lfĉ − lfc\| / Σ\|lfc\|. A missing or non-finite predicted LFC counts as 0. Raw 1 means predicting no change. | mean over 209–272 scored perturbations |
| **fid** | `k / max(n_pred, n_real)`. n_pred = genes the prediction calls significant (the reference doesn't have to call them). k = those whose sign matches the reference LFC. n_real = the reference's significant count. **Calling fewer genes than the reference is penalized; calling more costs only through wrong signs.** Calling nothing → 0. If n_real = 0 and n_pred > 0 it reduces to k/n_pred. | mean, dropping 0/0 cases |
| **reach** | Rank the reference-significant genes by the prediction's own confidence (predicted significant first, then p_adj, then p, then \|lfc\| descending). k* = the deepest prefix whose sign purity is ≥ 0.9. reach = k*/n_real. If sign accuracy is ≥ 0.9 everywhere, reach = 1; otherwise the best possible is accuracy/0.9. | mean over n_real > 0 |
| **jac** | Jaccard index of the significant gene sets \|R∩R̂\|/\|R∪R̂\|. An empty union counts as 1. **Over-calling and under-calling both cost.** | mean over all 300 |

### 4.3 Scaling and overall score [code + spec]

For each metric and each context, `s = (u − b)/(r − b)`:
- **b** = the "context mean" baseline. It is an **oracle**: the true mean response of that context's perturbed cells, tiled identically onto every perturbation.
- **r** = the mean of 5 disjoint split-halves of the real data (seeds from `SeedSequence(0)`). Each half has its own ~9.2k controls. NMAE uses a special estimator whose gene gate comes from the full reference.
- **Overall score** = mean of the 6 scaled metrics in each context, then the mean over the 3 contexts. No weights.

| Metric | Baseline b | Replicate r | One raw unit = this many scaled points | Perfect prediction scores | Floor |
|---|---|---|---|---|---|
| pds ↑ | 0.500 | 0.927–0.984 | ~2.1–2.3 | 1.03–1.17 | −1.03 to −1.17 |
| mse ↓ | 0.986–0.992 | 0.028–0.045 | ~1.05 | 1.00 (clamped) | 0 (clamped) |
| nmae ↓ | ~1.001 | 0.369–0.431 | ~1.6–1.75 | 1.58–1.75 | −6 (raw ≥ ~4.4–4.8); raw 2.0 → about −1.6 |
| fid ↑ | 0.505–0.522 | 0.795–0.832 | **~3.0–3.6** | 1.51–1.72 | −1.55 to −1.85 |
| reach ↑ | 0.047–0.097 | 0.958–0.978 | ~1.1 | 1.02–1.05 | ≥ −0.12 |
| jac ↑ | 0.021–0.037 | 0.375–0.423 | **~2.6–2.9** | **2.48–2.85** | −0.05 to −0.11 |

Validation and final scores are **not comparable**: they cover different lines and panels, and PDS is a rank within each panel.

## 5. Leaderboard landscape (API pull, 2026-10-13, 1,476 teams, anchors `vcc2026-val{A,B,C}-r4`)

| | overall | pds | mse | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|
| #1 Illumina AI "iCell" | **0.466** | 0.883 | **0.782** | 0.417 | 0.183 | 0.457 | 0.078 |
| #2 – #10 | 0.304–0.331 | 0.75–0.83 | 0.43–0.47 | 0.21–0.29 | 0.04–0.09 | 0.25–0.34 | 0.00–0.09 |
| 99th percentile / 90th / median | 0.297 / 0.236 / 0.091 | | | | | | |

- The leader's raw values are pds 0.898, mse 0.245, nmae 0.750, fid 0.565, reach 0.483 and jac 0.059. Ranks 2–10 have raw mse ≈ 0.55.
- **Almost the entire field sits near chance on fid and jac.** Those two metrics have the largest leverage per raw point (fid) and the most headroom (jac, up to ~2.5–2.85). [inferred: the field hasn't calibrated cell-level generation]
- Illumina's gap is mostly MSE (+0.3 scaled). One plausible source is its proprietary **Billion Cell Atlas** (CRISPR screens in more than 200 cell lines); training on such data is allowed. [inferred]
- Self-reported community entries:
  - LFC transfer from public screens scores about 0.07–0.15 overall, with **0.00 scaled MSE**.
  - A fine-tuned State model scored **below** the control.
  - Arc says State "is not able to win."

## 6. What this means for strategy

### 6.1 Two separable sub-problems
1. **Effect model (pseudobulk level).** For each (context, target), predict per-gene expected expression/LFC, which genes respond, and the overall response strength. This drives pds, mse, nmae and the signs used by fid and reach.
2. **Cell generator (count level).** Turn that into 400 integer-count cells whose **Wilcoxon significance against the 18,400 real controls** matches what the reference would show. This drives jac, fid coverage and the reach ordering. Four of the six metrics depend heavily on this step, and the field mostly gets it wrong.

### 6.2 Cell generator findings
These come from the DE agent's simulation on context-A controls, which reimplements the gpudge test, gate, BH and metrics [simulation; directional only]:
- 400 cells against 18,400 controls detects AUC shifts of about 0.06. **Any mismatch in zero fraction, dispersion or library-size distribution on null genes becomes "significant".**
- Using oracle means on a medium-strength perturbation (n_real = 304):

| Generator | Calls on null genes | jac | fid | nmae |
|---|---|---|---|---|
| **Control-cell templates (resampled real control cells) + binomial thinning (down) / Poisson-added counts (up)** | 9 | **0.74** | **0.91** | 0.19 |
| Negative binomial with fitted dispersion, library sizes from controls | 85 | 0.56 | 0.88 | 0.19 |
| Plain Poisson | ~860–1,050 | ~0.2 | ~0.6 | 0.19 |
| Library sizes ×0.5 or ×2 | ~7,000 | 0.03 | 0.52 | 0.2 |
| Identical cells (rounded mean) | 8,242 | 0.03 | 0.51 | **9.3** |

- The significant set can be steered **independently of the mean**: the same mean shift spread across cells in different ways gives 105 vs 286 vs 330 calls at identical NMAE. So set means to the best LFC estimate, and use the *cell-level spread* to express confidence that a gene is significant.
- A global composition shift of 10% or more (from changing a few large genes) creates ~1,700 false calls. Keep library composition stable.
- Never emit a zero total for a gene that passes the filter (log2FC ≈ −30, NMAE → −6). Avoid all-zero cells.
- For a perturbation we believe is null, call nothing: jac then scores 1, and fid/reach drop it from their averages.

### 6.3 Effect model findings
- **Beating the baseline means beating an oracle context mean.** The model must predict the context's generic response (the "template": proliferation and p53/stress programs, plus the effect of the CRISPRi machinery itself) from NTC cells alone, *and then* add target-specific effects.
- Evidence on what transfers across lines:
  - Noise-corrected knockdown-effect correlations across lines are 0.40–0.75 (Nadig 2025). They are highest between lineage-similar lines (K562–Jurkat 0.74, HepG2–RPE1 0.75) and lowest for K562–RPE1 (0.40).
  - Response variance splits roughly into template 28%, conserved perturbation component 29%, line × perturbation interaction 24% and noise 19% (Molina & Zhang 2026).
  - Nobody has recovered the interaction term zero-shot.
- **What has worked:**
  - Simple, regularized models across **multiple** source lines: linear/ridge on perturbation embeddings learned from other lines (Ahlmann-Eltze 2025), TxPert, TabPFN with cross-line features, and similarity-weighted transfer (2025's 3rd place, TransPert).
  - Weighting sources by inferred lineage similarity to each context.
  - 2025 winners were hybrid models (DL + statistical features) trained on pseudobulk with metric-aware losses.
- **What has failed:** frozen foundation-model embeddings (Geneformer, scGPT), and State used as-is.
- **LLM-chosen neighbours (LangPert):** at most a gated fallback for targets nobody has measured. Knowledge-graph and DepMap neighbours match or beat LLM picks. Details and the experiments to run are in [LangPert_notes.md](LangPert_notes.md).
- **MSE** is a ratio of sums, so getting the **strong responders** right matters most. Over-shrinking hurts DE; over-amplifying hurts NMAE linearly.
- **PDS** ignores all target genes. Only the off-target *direction* of each predicted effect matters.
- **Promoter-neighbor effects:** CRISPRi silences genes sharing a bidirectional promoter with the target. These count, because only the target's own row is excluded. [community; plausible]
- **Arc models:**
  - **PIE** (released 2026-10-05) is Arc's best fit for this task. It predicts per-gene `p_de`, LFC and Δexpression for an unseen context from control means plus knowledge embeddings (ESM2, STRING, DepMap, gene text), which cover essentially all 300 targets. It reports Jaccard 0.18 and direction 0.82 on leave-one-line-out Replogle.
  - Its non-commercial license is separate from the State/Stack FAQ policy, so check with the organizers before using it.
  - Released State checkpoints silently fall back to control output for unknown targets, which scores ≈ 0.
  - Stack covers 213/300 targets and 13k genes. It is a possible secondary generator.
- **Candidate training data** (§2 has coverage figures):
  - Platform-matched (Flex + CRISPRi): H1-2025 (Arc), CD4 T cells (MIT license), X-Atlas/Orion and Pisces (non-commercial; check eligibility).
  - Also: Replogle K562 GW/K562/RPE1, Nadig HepG2/Jurkat, Jiang (6 lines, Parse chemistry), Tahoe (drugs).

### 6.4 Local evaluation harness (needed for the blind final)
- Leave-one-cell-line-out on public CRISPRi data formatted like VCC: about 300 shared targets, 400 cells per perturbation, ~20k median UMI, 18,533-gene axis, controls labelled `non-targeting`. Score with the official `cell-eval2 --preset vcc2026 --pert-col target_gene` (`baseline --save-pred` → `prep-real-bundle` → `run` → `score --real-bundle`).
- **Hardware:** this Mac (M5 Pro, 24 GB) is below the documented peaks (47–65 GiB at VCC scale). Options:
  - a Linux CUDA box with `cell-eval2[gpu,gpudge]`, which matches the official engine (H100 ≈ 34 s per 300-perturbation set);
  - a chunked CPU reimplementation of the DE metrics, which the agent's simulation scripts already started.
- Use the validation leaderboard to *calibrate* the harness: once raw values come back from `vcc status --json`, solve for each context's b and r.

### 6.5 Don'ts and gray zones
- **Don't use the MSE "layout" residual.** That means pinning group sums to the prediction while scattering cells so the noise correction pays out (worth up to ~0.9 of MSE's range). The maintainers call it an exploit (#294/#348), have a fix ready (rule_version 4), and the Rules let Arc disqualify suspicious entries or re-score. Also avoid the #291 reach head-miss quirk.
- Closed exploits (don't rely on them):
  - #343: spiking the other targets for PDS
  - #351: DE filter sign leak
  - #172: own-target row in MSE
  - #247: inflating dispersion to buy correction
  - #282: alphabetical tie ordering
  - #248: guide-level labels
- **Rules (rev. Sep 16):** never submit measured lab or literature data, or use it to revise model predictions. Training on it is fine. Finalists must disclose datasets, models, and **any non-learned/statistical components blended with model outputs**, so those are allowed but must be disclosed.
- **Questions for help@virtualcellchallenge.org:**
  1. Is PIE covered by the same policy as State/Stack?
  2. Are non-commercial datasets (X-Atlas) OK for a prize entry?
  3. Is using the provided control cells as *templates* for generated perturbed cells acceptable? The spec says controls are "model inputs only".
  4. Is inferring a context's lineage from its NTC profile OK?
  5. Will the final phase show teams their own scores?
  6. Will rule_version 4 land before the D/E/F bundles are built?
  7. Is a *learned* model that transfers a target's measured responses from public screens to a new line an ML
     prediction under the Sep 16 rule? (We never pass measured values through directly.)

## 7. Key links
- Site: https://virtualcellchallenge.org — [Evaluation](https://virtualcellchallenge.org/evaluation) · [Data](https://virtualcellchallenge.org/datasets) · [FAQ](https://virtualcellchallenge.org/faq) · [Rules](https://virtualcellchallenge.org/rules) · Leaderboard API: `https://virtualcellchallenge.org/api/leaderboard?get_final=false&is_generalist=false`
- CLI guide: https://vcc-cli-wiki.virtualcellchallenge.org/ (plus `/cli-reference`) · PyPI `vcc-cli`
- Scorer: https://github.com/ArcInstitute/cell-eval2. Spec: `docs/vcc2026_metrics/vcc2026-metrics.md`; config: `src/cell_eval2/configs/vcc2026.yaml`. DE engine: https://github.com/ArcInstitute/gpudge
- Arc models: https://github.com/ArcInstitute/state · https://github.com/ArcInstitute/stack · https://github.com/ArcInstitute/pie
- Arc news: [VCC 2026 announcement](https://arcinstitute.org/news/virtual-cell-challenge-2026) · [Behind the data 2026](https://arcinstitute.org/news/behind-the-data-virtual-cell-challenge-2026) · [2025 wrap-up](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up)
- Cell 2026 commentary: doi:10.1016/j.cell.2026.08.004 (paywalled). PRiMeFlow (recommended by Arc): arXiv 2604.13986
- Transfer evidence: Nadig 2025 (doi:10.1038/s41588-025-02169-3) · Ahlmann-Eltze 2025 (doi:10.1038/s41592-025-02772-6) · Molina & Zhang 2026 (bioRxiv 10.64898/2026.07.24.740459) · TxPert (arXiv 2505.14919)
- Community repos [self-reported]: MMatinGerami/zero-shot-perturbation · kaipengm2/Virtual-Cell-Challenge-2026 (AtlasShift)
- Support: help@virtualcellchallenge.org · Discord (registered participants)
