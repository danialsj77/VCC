# Our method

*Living document: keep it current. A cleaned-up version becomes the method description the rules require from
finalists.* Status tags: ✅ built and tested · 🔧 built, not yet validated on real data · 🔜 planned.

## 1. What the scoring rewards (the constraints that shape everything)

We are given only the **non-targeting control cells** of each unseen cell line and the list of 300 target genes.
We must return **400 single cells per target** as raw counts. Arc's scorer then computes six metrics (formulas:
[notes/VCC2026_brief.md §4](notes/VCC2026_brief.md)):

- **pds, mse** compare pseudobulk profiles: is our predicted effect pointing the right way (pds), and how far is our
  profile from the truth (mse, dominated by the strongest-responding targets)?
- **nmae, fid, reach, jac** come from a **Wilcoxon test of our 400 cells against 18,400 real control cells**, gene
  by gene. They ask: are fold changes the right size (nmae), in the right direction (fid, reach), and do we call the
  same responding genes (jac)?

Two facts follow, and our design is built around them:

1. **The cells matter as much as the averages.** With 400 vs 18,400 cells, a small mismatch in how our cells look
   (zero fraction, spread, depth) makes many unaffected genes "significant" and hurts four metrics. Almost every team
   on the leaderboard sits near chance on fid and jac, so this looks like the cheapest place to gain.
2. **Score 0 is the line's *true* average response.** To score above 0 we must predict that average (from controls
   alone) *and* the target-specific deviations. Effect sizes must be right, not shrunk. fid's scale is the narrowest
   of the six, so direction errors and too few significant genes cost a lot.

## 2. Architecture: three separable parts

```
 control cells of a new line ─┐
 public Perturb-seq screens ──┼─▶ (3) EFFECT MODEL ──▶ predicted mean expression per target & gene
 gene knowledge (DepMap, ...) ┘        + confidence            │
                                                                ▼
                                     (2) CELL GENERATOR ──▶ 400 count cells per target ──▶ submission
                                                                ▲
                (1) EVALUATION HARNESS: official scorer on held-out real data, tells us which choices win
```

Keeping "what changes" (effect model) separate from "how it looks as cells" (generator) lets us improve and test
each one on its own, and lets any effect model plug into the same generator.

## 3. The parts

### (1) Evaluation harness: `notebooks/01`

A "practice exam" we can take any number of times: a held-out cell line with known answers, formatted like a VCC
context (sizes differ slightly; see the notebook), scored by Arc's official `cell-eval2` with the `vcc2026` profile
and the same scaling formula as the leaderboard. Our baseline arm approximates Arc's; their exact construct filter
isn't public.

- ✅ **Synthetic proxy**, built from our own control cells. It tests code, not biology.
- 🔧 **H1 hESC from the 2025 challenge** (public, same 10x Flex chemistry, thinned to ~20k UMI/cell). Code is written
  but not yet run (task #2). This is our first real-biology proxy.
  - **Matched to the challenge's response strength.** H1's panel responds far more strongly than the VCC panel.
    Counting, per target, the genes with |z| > 4.5 in K562 (among genes above 5 CPM in K562 controls): the 272 VCC
    validation targets K562 measured have median 3 and maximum 42 such genes; the H1 targets have median 4, 95th
    percentile 181, and 16% have 50 or more. Settings tuned on the strong targets would be tuned for a regime the
    challenge doesn't have (more real signal at every threshold, so a lower `z0` would look better than it is). The
    proxy therefore leaves out H1 targets with more than 50 such genes. Targets K562 never measured are kept.
- 🔜 **K562 genome-wide** at full VCC format. CD4 T cells and the Nadig/Replogle lines are usable at pseudobulk level
  only: too few cells per target, shallower sequencing, fewer genes.

This matters most in the final phase, which gives no feedback, changes both the cell lines and the target panel, and
counts only our last entry.

### (2) Cell generator: `vccpipe/generator.py` (teaching version: `generate_cells()` in notebook 01)

🔧 Built: `TemplateGenerator` and `NBGenerator`; not yet calibrated on real cells. **Every submission uses
`NBGenerator` until the organizers confirm in writing that control cells may be used as templates** (see "Open
rules question" below).

`TemplateGenerator` turns a predicted mean profile into 400 cells by **editing real control cells of the same
line**: binomial thinning for decreased genes, added Poisson counts for increased genes. Each template cell keeps
its own library size and zero pattern. An optional `spread` lets cells respond unevenly; the averages are preserved
exactly, which we checked numerically.

What we know so far:
- **Simulation** (one medium-strength knockdown, context A, by a separate re-implementation of the scorer's test):
  the template approach made 9 false "significant" genes, against ~860–1,050 for plain Poisson sampling and 8,242
  for identical cells.
- **Harness, synthetic proxy** (30 targets, CPU engine). This test is circular, because the synthetic truth is made
  with this same generator; it checks the code, not realism. Given the measured means of the true cells it scores
  1.07 overall; given only their clear changes (|z| > 3) it scores 1.17.
- **On that proxy:**
  - Halving every effect size cut the overall score from 1.07 to 0.68, mostly through fid and nmae.
  - Feeding the generator noisy small changes on many genes cut jac from 1.10 to 0.30.

  So the effect model must (a) **not shrink** effect sizes and (b) **predict "no change" wherever it isn't confident**.

**Open rules question.** The data page calls the controls "model inputs only — do not copy them into your
prediction". Genes our model predicts as unchanged keep the template cell's control counts, and across a submission
each control cell would appear about 6.5 times, nearly verbatim. We have drafted the question to the organizers
([notes/organizer_email_draft.md](notes/organizer_email_draft.md), task #7). Until they confirm, the 🔧 **NB
generator** (`NBGenerator`, task #8) is our default, and `scripts/03_make_submission.py` refuses the template
generator unless `--allow-control-templates` is passed. The NB generator samples every count from a negative
binomial fitted per gene to the controls:
- **Depth-aware.** Control cells are split into 10 library-size deciles, and each decile gets a per-gene factor
  H[d] = (mean CPM in decile d) / (overall mean CPM). A generated cell takes the library size and decile of a
  randomly drawn control cell, with mean `predicted CPM × H[decile] × library size / 10^6`. Real per-gene CPM varies
  with depth, and the scorer's pseudobulk is a ratio of summed counts, so ignoring depth adds the same offset to
  every target. On context A's controls, the squared distance between the pseudobulk of a generated "no change"
  prediction and the real control pseudobulk falls from about 18 to about 15 with the deciles (real 400-cell
  subsets of the controls: about 14.5).
- In the earlier simulation (before the deciles) it made ~85 false genes: worse than templates, far better than
  Poisson.
- It still copies one number per cell from the controls (the library size). The draft e-mail asks about that too.
- 🔜 A richer generative model fitted to the controls (a negative-binomial factor model with per-cell size factors
  and a few latent factors, or scVI) could win back most of the template generator's realism without copying any
  cell: the controls would then be training data, not output. It would be scored against both generators on the
  H1 proxy.

**Calibration (task #2).** `spread` and the confidence threshold are chosen with the official scorer on the H1
proxy (see "Choosing the settings" under [What v1 implements](#what-v1-implements)). We check that a winning setting
gains on the four DE metrics and not through scaled mse, so it can't gain through the MSE noise correction.

### (3) Effect model: learned zero-shot transfer across cell lines

🔧 **v1 built** (a deliberately simple transfer baseline, described honestly in
[What v1 implements](#what-v1-implements) below); reviewed and tested on fake data only, never run on real data
yet. 🔜 The learned model described in 3a–3d (v2 onwards).

For a new line we have its control cells, plus the responses of the same or related knockdowns measured in *other*
lines. **The goal for v2 onwards: every piece below is a fitted model trained on public data.** Measured responses
are then training data only, so no submitted value is a copy, or a hand-weighted average, of measured data. **v1
does not meet this yet**: it rescales the same target's measured K562 response (see What v1 implements and §5).

**3a. Context template.** Every knockdown in a line shares part of its response: CRISPRi machinery, stress and
proliferation programs. In published analyses this "template" explains ~28% of response variance (Molina & Zhang
2026). We learn to predict it from a line's control profile (e.g. ridge regression), trained on all public screens
where it can be measured.

**3b. Target-specific response.** A regularized regression, fitted with leave-one-line-out, predicts a target's
response in a held-out line. Its inputs:
- the target's template-removed responses in the source lines, as features;
- similarity between the source lines and the new line, from control profiles only (no cell-line names);
- the reproducibility of each source response.

The model learns how much to trust each source and how to rescale magnitudes. This follows the best simple baseline
in the literature: linear models over perturbation embeddings learned from other lines' Perturb-seq (Ahlmann-Eltze
et al. 2025).

Evidence for and against:
- **For:**
  - Noise-corrected knockdown effects correlate 0.40–0.75 across lines, highest between similar lineages (Nadig et
    al. 2025).
  - ~30% of the response is a conserved component (Molina & Zhang 2026).
  - Linear transfer models and knowledge-graph models such as TxPert beat single-cell foundation models in
    benchmarks.
- **Against:**
  - Self-reported VCC 2026 entries that transfer each target's public responses score only ~0.07–0.15 overall, with
    0 scaled mse.
  - One such entry found that weighting sources by control-profile similarity *hurt* on H1.

  So transfer alone is not enough. The gains have to come from 3a, 3d and the generator, and every weighting choice
  must win on held-out lines.

Coverage of the 300 validation targets in public screens:
- K562 genome-wide: 272/300.
- CD4 T cells: 294/300 listed (cells per target not yet checked).
- X-Atlas: 148/300 in PIE's preprocessed subset; the full library probably covers more (to check). License is
  CC BY-NC-SA.
- H1-2025: 25/300.
- Usable coverage is task #1.

**3c. Targets nobody measured.** Here we rely on knowledge-based neighbours:
- DepMap co-essentiality (genes whose knockouts affect cell lines alike);
- STRING/CORUM network neighbours;
- response similarity in K562 genome-wide.

A learned model takes mainly a *direction* from them, with each neighbour's own knocked-down gene masked out, gated
by the strength model (3d). Where evidence is weak, it predicts "no change".

**3d. Strength and confidence.** A classifier/regressor predicts how strongly each target responds in the new line,
and which genes are confidently changed. Features include source-line response sizes, DepMap dependency in similar
lines, and the target's expression in the new line's controls. The generator then changes only confident genes, at
full (not shrunken) size. Magnitudes are calibrated on held-out lines.

**Decision rule for every choice above:** keep it only if it improves the official composite score on held-out lines
(H1 and K562-GW at full VCC format, plus pseudobulk-level checks on the other lines), with bootstrap confidence
intervals. The leaderboard (2 shots per day, validation lines only) confirms; it does not tune.

### What v1 implements

v1 is the simplest effect model that runs end to end: public data → prediction → cells → official scorer →
submission file. Its job is to make the whole pipeline work on real data and to give us a first honest score; the
learned pieces of 3a–3d come next. Interfaces: [docs/DESIGN_v1.md](docs/DESIGN_v1.md). How to run it:
[RUNBOOK.md](RUNBOOK.md). Walkthrough with plots: `notebooks/02_effect_model_walkthrough.py`.

**What v1 is, plainly.** For a target that K562 measured, v1 predicts the **same target's published K562
response**: K562's panel-average response is removed, the remaining deviation is kept only where its z-score is
large, and it is rescaled by a gain fitted on H1 times a scale multiplier chosen on the H1 proxy. K562's
panel-average response, rescaled by a second gain, is added back. The other pieces are fixed rules: a
distance-binned prior for neighbour silencing (targets K562 didn't measure), a fixed knockdown of the target's own
gene, and an expression floor. Cells come from the NB generator by default. v1 is a transfer baseline, not yet the
learned model of 3a–3d; a learned per-bin / regression transfer (v2) is the planned upgrade.

**Inputs.** Each public screen becomes a *response table* on the VCC 18,533-gene axis: for every target, the mean
expression of its knockdown cells, the control mean, and the standard error of each log2 fold change. v1's
predictions use **one source**, the K562 genome-wide screen (Replogle 2022 bulk files). Arc's 2025 H1 data, in the
same format, is the held-out line that v1's two gains are fitted on and its settings are judged on. **H1 is not a
source of the submission** (source weight 0). Averaging it in would change both the template and the specific
response in ways validation never saw: H1 measured only ~25 of the 300 validation targets, so its panel average
would be the mean of a handful of targets, added to every target, alone on the ~10k genes K562 never measured.

**The prediction**, as a log2 fold change for target `t` and gene `g` in a new line `C`, given only C's controls:

1. **Template `T(g)`.** The mean K562 log2 fold change over the predicted panel's targets that K562 measured (each
   target's own gene left out). Every target gets `gamma · gamma_scale · T(g)`. Only K562 contributes
   (`template_sources = ["k562_gw"]`). This is a stand-in for 3a: v1 does not predict the template from the new
   line's control cells.
2. **Specific response `D(t,g)`.** K562's response to this target minus K562's template, `lfc_K562(t,g) − T(g)`,
   kept only where this *deviation* is confident: `z_dev = (lfc_K562 − T) / se`, weight
   `clip((|z_dev| − z0) / z_soft, 0, 1)` (a soft ramp; `z_soft = 0` is a hard threshold). Gating on the deviation,
   not on the raw response, keeps noise on genes with a large template out and keeps real deviations on genes whose
   raw change is near 0. Targets whose knockdown had no detectable effect in K562 (Replogle's energy-test p above
   `null_p`) get no specific response at all. Every target gets `beta · beta_scale · D(t,g)`. This is 3b without
   the learned similarity weighting, and 3d in its simplest form: a confidence gate, not a trained classifier.
3. **Gains.** `beta` and `gamma` are fitted by weighted least squares of H1's log2 fold changes on `[D, T]`, over
   (target, gene) pairs that H1 measured and the scorer can test (control CPM > 5 in H1), the target's own gene
   left out. They are the only parameters fitted to responses. Least squares on a noisy `D` shrinks the gain
   (regression dilution: in a synthetic test where half of K562's genes transfer at full size, it gave
   `beta` ≈ 0.4–0.5), while the DE metrics punish predicted changes too small to be called significant. So
   validation also tries the multiplier `beta_scale` on top of the fitted `beta` (`gamma_scale` stays 1 in v1).
   The multipliers are chosen by the scorer, not fitted.
4. **Effect-size floor.** Where `D(t,g) ≠ 0` and the new line's control CPM is above 5 (genes the scorer tests),
   the predicted change is at least `min_called_lfc` in size, with its sign kept. With 400 vs 18,400 cells, a
   change of −0.2 log2 at 100 CPM is almost never called significant, so a confident K562 change shrunk below that
   would be invisible to fid, reach and jac.
5. **Unexpressed genes** of the new line (control CPM below 0.5) get no change. The floor is below the scorer's
   5 CPM gate, so it never removes a gene the scorer tests.
6. **Neighbour silencing (cis).** For targets that no source measured, genes whose start site lies within 50 kb of
   the target's get the median K562 log2 fold change of their distance bin (0–1, 1–2, 2–5, 5–10, 10–20, 20–50 kb).
   Start sites come from GENCODE release 32, whose gene names match the VCC axis (10x GRCh38-2020-A); newer
   releases rename ~500 of its genes. This replaces 3c's knowledge-based neighbours for now.
7. **The target itself** gets a fixed knockdown (log2FC −2.7, about 15% left); the scorer ignores this row.

**From prediction to cells.** The predicted fold changes move the new line's control means; the NB generator of
part (2) turns them into 400 cells per target (the template generator only if the organizers allow it).

**Choosing the settings** (`scripts/02_validate.py`, grids in `configs/v1.yaml`), with the official scorer on the
strength-matched H1 proxy (part (1)):
- **Stage 1:** `z0` ∈ {2.5, 3.5, 4.5} × `null_p` ∈ {0.05, 1.0} × {template, template with spread 0.5, nb}, plus
  two ablations (no specific part; no cis) and "predict no change" as a reference. A threshold of 1.5 is not tried:
  on K562 responses of VCC-like targets, ~93% of the genes passing |z| > 1.5 are noise (~2/3 at 2.5; ~0.1 noise
  genes per target at 4.5).
- **Stage 2:** around the stage-1 winner of each generator, `beta_scale` ∈ {1, 2, 3} × `min_called_lfc` ∈ {0, 0.4}.
- **Cross-fitting.** The proxy targets are split into two folds, and each fold is scored with gains fitted on the
  other fold's H1 answers, so no target is graded with gains fitted on its own answer.
- **Paired comparison.** Cells are generated with one random seed per target, so every variant is scored on the
  same control-cell and library-size draws, and differences between variants come from the settings.
- **Outputs.** `best_params.json` (overall winner), `best_params_nb.json` and `best_params_template.json` (best of
  each generator), each with its gains refitted on all proxy targets. The submission uses `best_params_nb.json`.

**Fitted, chosen, or fixed:**

| Component | How it is set |
|---|---|
| `beta`, `gamma` | fitted: weighted least squares on H1 |
| `z0`, `null_p`, `beta_scale`, `min_called_lfc`, generator, `spread` | chosen by the official scorer on the H1 proxy |
| cis effect per distance bin | median K562 log2 fold change per bin (bins set by hand) |
| NB dispersions and depth factors | estimated per gene on each context's own controls |
| template = K562 panel mean, K562 as the only source, `z_soft` = 1, `gamma_scale` = 1, expression floor 0.5 CPM, own-gene knockdown −2.7, the proxy's strength filter | fixed by hand |

**The upgrade (v2): learned transfer.** Each step below contains v1 as a special case and is chosen by the same
cross-fitting, so it should not cost score:
- **(a) Per-bin gains:** separate gains for bins of |z| × the new line's control expression, ridge-penalized
  toward the global gain. They replace the hand-set `z0`, `z_soft` and the scale multipliers.
- **(b) A regression over (target, gene) pairs:** H1's log2 fold change as the answer; as features the K562 log2
  fold change and z, the K562 template, the gene's control expression in the new line and in K562, target
  metadata (phenotype p, knockdown, cells) and the cis distance bin. Ridge or gradient boosting (monotone in the
  K562 response), trained leave-one-line-out as more line pairs are added (Replogle RPE1 and K562 essential, Nadig
  HepG2 and Jurkat).
- **(c) The template predicted from the new line's control profile (3a).** This also gives a principled
  prediction on genes K562 never measured.
- **(d) H1 only as a fitted training line**, never averaged in as a source.
- **(e) The own-gene knockdown estimated from data** (the median own-gene log2 fold change of the kept H1 targets).
- Then the rest of 3b–3d: several source lines with similarity weights, knowledge-graph neighbours for unmeasured
  targets, a trained strength/null model.

**Rules note.** v1 is the weakest point of our compliance story. On confident genes, its submitted value is roughly
`beta·beta_scale · lfc_K562(t,g) + (gamma·gamma_scale − beta·beta_scale) · T(g)`: the published K562 response of the
same target, gated and rescaled by global factors. If a challenge line is, or closely resembles, K562, v1 would
submit nearly the published measurements, rescaled. Whether this counts as an ML prediction is the first question we
are putting to the organizers (task #7; e-mail text in [notes/organizer_email_draft.md](notes/organizer_email_draft.md)).
Until they answer, v1 serves as a pipeline test and a leaderboard calibration point, and v2 is the planned final
entry. See §5.

**Status.** Built, reviewed against the real files, and unit-tested on small fake data. Never run on real data at
full size: the first TSUBAME run is its test.

## 4. How LangPert informed our design, and why it is not the core

[LangPert](https://openreview.net/forum?id=Tmx4o3Jg55) (Märtens et al., MLGenX 2025), suggested by Dr. Mehrshad on our team,
uses an LLM to pick 5–10 functionally similar genes whose responses were measured, and predicts the average of their
responses. It is a simple, low-cost idea, and we read it carefully (notes:
[notes/LangPert_notes.md](notes/LangPert_notes.md)). Two of its ideas shaped our design:
- Its option to **abstain** ("pick no genes if not confident") became our null / "no change" logic in 3d.
- Its **neighbour retrieval** became the 3c fallback for unmeasured targets.

We don't make it the core of the model, for six reasons:

1. **It addresses a different problem.** LangPert predicts an unmeasured knockdown in a cell line where many *other*
   knockdowns were measured, and averages those same-line responses. VCC gives no knockdown data at all in the target
   lines. Our hard part is transferring across lines.
2. **Its evidence comes from a favourable setting.** It was tested within K562 and RPE1, on essential genes, whose
   knockdowns cause strong responses that cluster by protein complex. Our targets look mostly non-essential (0 of 300
   are in the Replogle essential sets). For non-essential genes, knowledge-based predictions are much weaker (PRESAGE
   on K562 genome-wide: mean cosine ~0.05 vs ~0.25 for essential genes).
3. **Its benchmark metric rewards averaging; VCC's don't.** LangPert was scored on error over each perturbation's
   top-20 changed genes, where averaged predictions do well. On our harness, halving effects cost ~0.4 overall
   (fid and nmae fell, although jac rose slightly). Small noisy changes across many genes cut jac from 1.10 to 0.30.
   We expect averaging many neighbours to behave similarly, but we haven't tested that directly.
4. **Knowledge-graph neighbours do as well.** An independent re-test (Fawkes, Hodgson & Hartford 2026; within-line,
   like LangPert's own benchmark) found that a neighbour average over STRING, GO and two screen-derived graphs matched
   or outperformed LangPert. Restricting LangPert to genes that were also graph neighbours left performance
   unchanged. In Arc's PIE, DepMap and STRING were more useful than text. In the setting closest to VCC (unseen line
   *and* unseen target), knowledge sources didn't improve *which* genes respond, but did improve their direction.
5. **Specific risks for our panel.**
   - 41 of the 300 targets (14%) are ZNF/ZBTB/THAP/ZSCAN-family zinc-finger genes, each regulating its own
     idiosyncratic set of genes, so "similar genes" are a weak guide.
   - Some related genes push expression in opposite directions (e.g. the MED13/CDK8 kinase module vs core
     Mediator), which costs fid heavily.
   - Each neighbour's own knockdown leaks into the average as a false signal.
6. **Practical issues.**
   - Different LLMs pick different genes (two models shared ~2.5 of ~6.2 picks), though with similar accuracy;
     run-to-run variation is untested.
   - Only our last, blind final entry counts.
   - Describing an inferred cell line in the prompt is a rules gray zone.

**Where LangPert-style ideas still fit.** Both uses below get in only if they beat STRING/DepMap neighbours on
held-out lines, with pinned model versions and cached outputs, and both would be disclosed:
- An LLM could **re-rank** STRING/DepMap candidates in 3c. The final phase brings 300 new targets, so this bucket may
  matter.
- Its "null / weak / strong" guess could be one feature of the strength model (3d).

## 5. Rules compliance and disclosure

The rules (revision of Sep 16) allow training on any data we have rights to, forbid submitting measured lab or
literature results or using them to revise model outputs, and require finalists to disclose every non-learned
component blended with model outputs. Where we stand:

- **ML predictions only.**
  - **Planned (v2 onwards):** every submitted value is the output of a fitted model plus the cell generator.
    Measured responses from papers or public data are used only to train models, never passed through as
    predictions or used to correct outputs.
  - **v1, honestly:** it submits the same target's K562 responses (Replogle 2022), with K562's panel-average
    response removed, gated by the z-score of that template-removed deviation, and rescaled by two gains fitted by
    least squares on H1 times scale multipliers chosen on the H1 proxy; plus K562's panel-average template, a
    distance-binned cis prior, a fixed own-gene knockdown and an expression floor, with cells from the NB generator
    by default. On confident genes this is close to a rescaled copy of published measurements of the same target
    (details: §3, [What v1 implements](#what-v1-implements)). It may or may not count as an ML prediction.
  - **Question to the organizers (task #7):** we are putting this question, and the generator question below, to the organizers
    (e-mail text: [notes/organizer_email_draft.md](notes/organizer_email_draft.md)). Until they answer, v1 is a
    pipeline test and a leaderboard calibration point, and the learned per-bin / regression transfer (v2, §3) is
    the planned upgrade for the final entry. If they rule that v1-style transfer is not an ML prediction, v1 is
    never our final entry.
  - **Generator:** we also asked whether control cells may serve as templates. Until they confirm in writing,
    every submission uses the NB generator, which samples every count from a model fitted to the controls.
- **Training data:**
  - In v1: Replogle 2022 K562 genome-wide (CC BY 4.0), v1's only source; Arc's 2025 H1 data (no license stated; Arc
    allows training on any data we have rights to), used as the held-out line on which v1's two gains are fitted
    (cross-fitted during validation, refitted on all proxy targets for the submission) and on which its settings
    are chosen. H1 is never averaged into a prediction. GENCODE release 32 (gene start sites for the cis prior).
  - Planned: Nadig 2025 (GEO, public access, no explicit license); CD4 T-cell Perturb-seq (MIT according to its
    analysis repository; data terms to confirm); DepMap; STRING; possibly X-Atlas (CC BY-NC-SA; eligibility to
    confirm with the organizers); Replogle RPE1 and K562 essential.
- **Non-learned components to disclose** (v1; the list shrinks as v2 learns them):
  - the template: an unweighted mean of K562 log2 fold changes over the predicted panel's targets (each target's
    own gene left out), K562 only;
  - the confidence gate: a soft threshold on the z-score of the template-removed deviation (`z0`, `z_soft`), with
    `z0` chosen on the H1 proxy;
  - the phenotype gate: no target-specific response where Replogle's energy-test p exceeds `null_p` (chosen on the
    H1 proxy);
  - the scale multipliers `beta_scale` and `gamma_scale` on top of the fitted gains (chosen on the H1 proxy;
    `gamma_scale` fixed at 1);
  - the effect-size floor `min_called_lfc` on confident genes above 5 CPM (chosen on the H1 proxy);
  - the expression floor: no change predicted for genes below 0.5 CPM in the new line's controls;
  - the fixed own-gene knockdown: log2 fold change −2.7 (~15% left);
  - the cis prior: the median K562 log2 fold change per TSS-distance bin (0–50 kb), for targets no source measured;
  - the source weights (K562 1, H1 0) and template sources (K562 only);
  - the strength filter of the H1 proxy (which H1 targets the settings are chosen on);
  - the cell generator: NB (per-gene negative binomial fitted on each context's controls, with the library size and
    depth decile of a randomly drawn control cell), or, only if the organizers allow it, the template generator
    (binomial thinning / Poisson addition on control-cell templates, with `spread`);
  - planned for 3c: masking of neighbours' own genes.
- **Not used:** scoring exploits (e.g. the cell-eval2 MSE "layout" residual, issues #294/#348).

## 6. Timeline

| Dates (2026) | Goal |
|---|---|
| Oct 14–16 | Coverage census; H1 proxy code; NB generator; public data pipeline; effect model v1 built and reviewed; organizer questions drafted |
| Oct 16–21 | Organizer questions sent (task #7); v1 run on TSUBAME and validated on the H1 proxy; first leaderboard submissions (NB generator) |
| Oct 22 | D/E/F released: rebuild inputs for the new lines and new targets; dry runs |
| Oct 22 – Nov 3 | v2: learned per-bin / regression transfer, template from control profiles; then strength model, neighbour fallback; validated offline only |
| by Nov 3 | Solid final entry submitted, consistent with the organizers' answers; later replacements only if clearly better offline |
| Nov 5, 23:59 UTC | Deadline (only the last final entry counts) |

## References

- Challenge: <https://virtualcellchallenge.org>.
- Scorer: <https://github.com/ArcInstitute/cell-eval2> (spec `docs/vcc2026_metrics/vcc2026-metrics.md`).
- LangPert and its follow-up and re-test:
  - Märtens et al., *LangPert* (MLGenX 2025).
  - Fawkes, Hodgson & Hartford, arXiv 2606.08816.
  - Boubnovski Martell et al., *MechPert*, arXiv 2602.13791.
- Cross-line transfer and knowledge-based models:
  - Nadig et al., *Nat Genet* 2025 (doi:10.1038/s41588-025-02169-3).
  - Ahlmann-Eltze et al., *Nat Methods* 2025 (doi:10.1038/s41592-025-02772-6).
  - Molina & Zhang, bioRxiv 10.64898/2026.07.24.740459.
  - TxPert, arXiv 2505.14919.
  - PRESAGE, bioRxiv 10.1101/2025.06.03.657653.
  - Arc PIE, bioRxiv 10.64898/2026.10.02.756297.
