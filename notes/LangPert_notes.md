# LangPert: reading notes and fit for VCC 2026

Paper: Märtens, Boubnovski Martell, Prada-Medina, Donovan-Maiye (Novo Nordisk). *LangPert: LLM-Driven Contextual Synthesis for Unseen Perturbation Prediction.* MLGenX workshop @ ICLR 2025 (Spotlight). Local PDF: `76_LangPert_LLM_Driven_Context.pdf`.
Code: https://github.com/novonordisk-research/LangPert (MIT license, last commit 2025-11-08). Compiled 2026-10-13 (paper read in full, plus a 3-agent follow-up and critique).

## 1. What the paper does
- **Task:** predict the pseudobulk response to an *unseen* single-gene knockout **in a cell line where many other perturbations were measured**.
- **Method:** an LLM never outputs numbers.
  - It is given the target gene, the list of genes measured in training, and a cell-line context string ("K562, CML model").
  - It returns about 5–10 functionally similar measured genes, or none if it is not confident.
  - The prediction is the **unweighted mean** of those genes' measured response vectors: a kNN whose neighbours are chosen by the LLM.
  - An optional second call asks the LLM to critique and refine its own list.
- **Evaluation:** Replogle K562 (1,092 perturbations) and RPE1 (1,543), from the GEARS-curated **essential** gene sets. 5-fold CV over perturbations *within* each line. MAE, MSE and Pearson on change-from-control, over each perturbation's top-20 DE genes.
- **Results:**
  - It beats GEARS, scGPT, GP+LLM and the non-control mean on MAE/MSE. K562 with 850 training perturbations: MAE 0.224 vs 0.265 (GP+LLM) vs 0.332 (mean).
  - Correlation gains are small: 0.731 vs 0.728 (GP+LLM) vs 0.671 (mean). On RPE1 with 50 training perturbations the mean wins on correlation.
- **LLM choice:** o1, o3-mini and Claude 3.5 Sonnet perform about the same, even though they agree on only ~2.5 of ~6.2 chosen genes. Refinement helped only o3-mini.

## 2. What happened after it (verified 2026-10-13)
- **No longer version** of LangPert exists. The code has one LLM call per gene, a plain mean, and K562 context hard-coded into one prompt template.
- **The same group's follow-ups:**
  - **MechPert** (arXiv 2602.13791): multi-agent consensus neighbours, +4–10% Pearson over LangPert. No mean baseline, no cross-line test, no DE metrics. The authors say Replogle is probably in the LLMs' training data.
  - **SynthPert** (arXiv 2509.25346): an LLM classifier for DE direction per gene. Its "87% on unseen RPE1" is close to the majority-class (not-DE) rate.
- **Independent re-test** (Fawkes, Hodgson & Hartford, arXiv 2606.08816):
  - A plain **knowledge-graph neighbour mean (STRING/GO)** beats LangPert and is second only to TxPert.
  - Only 30–50% of LLM picks are KG neighbours. Using only the KG∩LLM overlap left performance unchanged, so the extra LLM genes add nothing.
- **PIE** (Arc, Oct 2026), knowledge-source ablation:
  - Ranking: DepMap > STRING > NCBI annotations > ESM2 (≈0) > free-text description (slightly negative).
  - In the "both unseen" regime (VCC's setting), knowledge sources add **no** DE-set accuracy (AUPRC −0.001), only +0.075 in direction match.
- **Non-essential genes:** neighbour methods degrade sharply. In PRESAGE on K562 genome-wide, mean cosine is ~0.05 for non-essential genes vs ~0.25 for essential ones.
- **No VCC 2025 or 2026 entry** is known to use LLM-chosen neighbours.

## 3. Fit for VCC 2026 (verdict: not a core model; at most a gated fallback)
**Why it doesn't transfer directly:**
- VCC gives **no** measured perturbations in the target line, while LangPert's neighbours come from the same screen.
- Our targets are mostly non-essential, with weak effects. 41 of the 300 validation targets (14%) are ZNF/ZBTB/THAP/ZSCAN-family zinc-finger genes with idiosyncratic targets, which LLMs handle poorly.
- VCC's metrics punish what top-20-DE MAE rewards:
  - Averaging dilutes and shrinks effects, which hurts nmae and turns true nulls (worth jac = 1) into non-nulls.
  - Neighbours from strongly responding essential complexes inflate weak targets. MSE is a ratio of sums, so a few blow-ups sink a whole context.
  - Functionally related genes can respond with **opposite sign** (MED13/CDK8 vs core Mediator, NFE2L1 vs proteasome subunits, PCGF1 vs PCGF6). That costs fid at ~3–3.6 scaled points per raw unit.
  - Each neighbour's own knockdown row, and its bidirectional-promoter partners, leak into the average and become false DE calls against 18,400 controls.

**Where it could still help:**
- Targets with **no measurement in any public screen**. The final D/E/F panel is new, so this bucket matters, and LangPert needs no target-specific training.
- An LLM-elicited **strength or null class** (null/weak/moderate/strong) used as one feature in a learned strength model. That feeds the count generator's "how many genes to call" decision.

**If kept, design it like this** (the critic's recommendation):
1. Work in source-line-centred residual space. Mask each neighbour's own gene and its promoter partners before aggregating.
2. Retrieve 50–100 candidates from STRING/CORUM/DepMap and response similarity. The LLM only reranks or abstains, with no cell-line names in the prompt.
3. Restrict candidates to the target's strength class, and drop neighbours that anti-correlate with the consensus.
4. Aggregate by per-gene sign consensus (median magnitude). Take only the *direction* from neighbours; the magnitude comes from a separate strength model.
5. Blend with own-target cross-line transfer using precision weights fitted by leave-one-line-out (LOLO). With no own evidence and low agreement, predict null.
6. For reproducibility: pin the models, temperature 0, vote over ≥5 samples × 2 models, cache all lists, and disclose the component in the finalist write-up.

## 4. Experiments to run before deciding (cheapest first)
1. **Coverage census:** which of the 300 targets have a measured, non-null response in any source line (K562-GW, RPE1, HepG2, Jurkat, CD4 T, X-Atlas, H1-2025)? If ≥85–90% do, LangPert drops to fallback-only or is removed.
2. **Self-row leak audit:** count how many naive neighbour-mean genes are neighbours' own targets or promoter partners.
3. **LOLO bake-off** (pseudobulk level, about 2 days). Arms: own-transfer, LLM neighbours, KG neighbours, data-driven neighbours, random same-stratum neighbours (negative control), KG∩LLM, and blends. Score with VCC-metric proxies, then end-to-end through the count generator.
4. **Memorization check:** compare the LLM−KG gap on older (Replogle) and newer data (CD4 T, X-Atlas, H1-2025). Trust only the newer-data gap.
5. **Strength/null classifier test:** LLM class vs a learned regressor vs both. Score by AUROC for n_real < 10 and by the downstream jac/fid change.
6. Only if the offline gates pass, run a validation-leaderboard A/B using at most 2 submission slots.

Key sources: LangPert repo (above) · Fawkes et al. arXiv 2606.08816 · MechPert arXiv 2602.13791 · SynthPert arXiv 2509.25346 · PIE bioRxiv 10.64898/2026.10.02.756297 · PRESAGE bioRxiv 10.1101/2025.06.03.657653 · GenePert bioRxiv 10.1101/2024.10.27.620513.
