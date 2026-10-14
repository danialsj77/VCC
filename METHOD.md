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
- 🔜 **K562 genome-wide** at full VCC format. CD4 T cells and the Nadig/Replogle lines are usable at pseudobulk level
  only: too few cells per target, shallower sequencing, fewer genes.

This matters most in the final phase, which gives no feedback, changes both the cell lines and the target panel, and
counts only our last entry.

### (2) Cell generator: `generate_cells()` in notebook 01

🔧 Turns a predicted mean profile into 400 cells by **editing real control cells of the same line**: binomial thinning
for decreased genes, added Poisson counts for increased genes. Each template cell keeps its own library size and zero
pattern. An optional `spread` lets cells respond unevenly; the averages are preserved exactly, which we checked
numerically.

What we know so far:
- **Simulation** (one medium-strength knockdown, context A, by a separate re-implementation of the scorer's test):
  this approach made 9 false "significant" genes, against ~860–1,050 for plain Poisson sampling and 8,242 for
  identical cells.
- **Harness, synthetic proxy** (30 targets, CPU engine). This test is circular, because the synthetic truth is made
  with this same generator; it checks the code, not realism. Given the measured means of the true cells it scores
  1.07 overall; given only their clear changes (|z| > 3) it scores 1.17.
- **On that proxy:**
  - Halving every effect size cut the overall score from 1.07 to 0.68, mostly through fid and nmae.
  - Feeding the generator noisy small changes on many genes cut jac from 1.10 to 0.30.

  So the effect model must (a) **not shrink** effect sizes and (b) **predict "no change" wherever it isn't confident**.

**Open rules question.** The data page calls the controls "model inputs only — do not copy them into your
prediction". Genes our model predicts as unchanged keep the template cell's control counts. We are asking the
organizers whether this is acceptable (task #7). Until they confirm, we are building a **fallback generator** that
samples every count from a fitted negative-binomial model, with library sizes drawn from the controls (task #8). In
the same simulation it made ~85 false genes: worse than templates, far better than Poisson.

**Calibration (task #2).** We tune `spread` and the confidence threshold on the H1 proxy, using the four DE metrics
only. We check that scaled mse doesn't change, so the setting can't gain through the MSE noise correction.

### (3) Effect model 🔜: learned zero-shot transfer across cell lines

For a new line we have its control cells, plus the responses of the same or related knockdowns measured in *other*
lines. **Every piece below is a fitted model trained on public data.** Measured responses are training data only, so
no submitted value is a copy, or a hand-weighted average, of measured data.

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

- **ML predictions only.** Every submitted value comes from fitted models plus the cell generator. Measured responses
  from papers or public data are used only to train models, never passed through as predictions or used to correct
  outputs. Questions sent to the organizers (task #7):
  - Is the generator's use of the provided control cells as templates acceptable?
  - Is a learned cross-line transfer of a target's public responses an ML prediction?
- **Training data** (planned):
  - Replogle 2022 (CC BY 4.0).
  - Nadig 2025 (GEO, public access, no explicit license).
  - CD4 T-cell Perturb-seq (MIT according to its analysis repository; data terms to confirm).
  - DepMap; STRING.
  - Possibly X-Atlas (CC BY-NC-SA; eligibility to confirm with the organizers).
  - Arc's 2025 H1 data is our held-out **evaluation** proxy; it joins the training set only for the final model,
    after model selection (no license stated; Arc allows training on any data we have rights to).
- **Non-learned components to disclose:**
  - the cell generator (thinning / Poisson addition on control-cell templates, or the NB fallback);
  - the confidence threshold and `spread` (calibrated on the DE metrics only);
  - masking of neighbours' own genes.
- **Not used:** scoring exploits (e.g. the cell-eval2 MSE "layout" residual, issues #294/#348).

## 6. Timeline

| Dates (2026) | Goal |
|---|---|
| Oct 14–16 | Ask organizers; coverage census; H1 proxy on TSUBAME; generator calibrated; NB fallback; public data pipeline |
| Oct 16–21 | Effect model v1 (template + learned transfer) validated on held-out lines; first leaderboard submissions |
| Oct 22 | D/E/F released: rebuild inputs for the new lines and new targets; dry runs |
| Oct 22 – Nov 3 | Improve (strength model, neighbour fallback, ensembling), validated offline only |
| by Nov 3 | Solid final entry submitted; later replacements only if clearly better offline |
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
