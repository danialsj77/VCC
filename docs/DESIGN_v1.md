# Model v1: design and interfaces

This file fixes the interfaces between modules so they can be built and tested independently. The *why* is in
[METHOD.md](../METHOD.md) (§3 "What v1 implements"); how to run everything is in [RUNBOOK.md](../RUNBOOK.md).

## Data flow

```
00_download_data.sh ──▶ data/replogle/*.h5ad, data/h1_2025/*.h5ad,
                        data/annotation/gencode.v32.basic.annotation.gtf.gz
01_build_sources.py ──▶ data/processed/k562_gw.npz        (ResponseSet: K562 genome-wide screen)
                       data/processed/h1_2025.npz        (ResponseSet: H1 pseudobulk; the truth line for the gains)
                       data/processed/tss.csv, cis.json  (gene TSS table, fitted CisModel)
                       work/h1_proxy/real.h5ad           (H1 formatted like a VCC context, for scoring;
                       work/h1_proxy/targets.txt          strength-matched to the VCC panel, see "Proxy")
                       work/h1_proxy/real.kd.csv
02_validate.py     ──▶ work/validation/results.csv       (official scorer on held-out H1 targets, one row per variant)
                       work/validation/best_params.json            (overall winner)
                       work/validation/best_params_nb.json         (best NB variant: what 03 builds by default)
                       work/validation/best_params_template.json   (best template variant)
03_make_submission ──▶ work/submission_<controls folder>/submission.vcc, submission.vcc.sha256, submission.json
                       (e.g. work/submission_controls/, final phase work/submission_controls_DEF/; the
                        intermediate submission.h5ad is deleted after packaging unless --keep-h5ad)
```

Everything lives on the **VCC gene axis** (`controls/gene_names.csv`, 18,533 symbols, the 10x GRCh38-2020-A
reference = GENCODE v32 names). A source's gene that is not on the axis is dropped; an axis gene not measured by a
source has `measured = False`.

## `vccpipe.common`

```python
CELLS_PER_PERT = 400                  # fixed by the challenge
CONTROL = "non-targeting"             # control label in obs['target_gene']
DE_GATE_CPM = 5.0                     # the scorer tests a gene only if its control mean CPM is > 5
```

## `vccpipe.sources`

```python
EPS_CPM = 0.5                         # pseudocount (CPM) for log2 fold changes everywhere in the model

@dataclass
class ResponseSet:
    name: str                         # "k562_gw", "h1_2025", ...
    genes: np.ndarray                 # (G,) str, the VCC gene axis
    measured: np.ndarray              # (G,) bool
    targets: list[str]                # (T,) target gene symbols (rows), unique
    target_cpm: np.ndarray            # (T, G) float32 mean CPM per target (0 where not measured)
    ctrl_cpm: np.ndarray              # (G,) float64 control mean CPM (0 where not measured)
    se_lfc: np.ndarray                # (T, G) float32 standard error of log2FC (np.inf where not measured)
    meta: pd.DataFrame                # index = targets; columns n_cells (int), kd_remaining (float, fraction of
                                      #   target expression left; nan if unknown), phenotype_p (float; nan if unknown)
    def lfc(self, eps=EPS_CPM) -> np.ndarray   # (T,G) log2((target_cpm+eps)/(ctrl_cpm+eps)); 0 where not measured
    def z(self, eps=EPS_CPM) -> np.ndarray     # lfc / se_lfc (0 where not measured)
    def subset(self, targets) -> "ResponseSet" # rows for the given targets (those present), same order
    def reindex_genes(self, genes) -> "ResponseSet"
        # columns mapped to a new gene axis BY SYMBOL (e.g. a final-phase bundle whose gene_names.csv differs).
        # Genes the source lacks get measured=False, cpm 0, se inf. Prints nothing (the caller reports counts).
    def save(self, path)                       # one .npz (+ meta as CSV string inside)
    @classmethod
    def load(cls, path) -> "ResponseSet"

def load_replogle_bulk(path, genes, name) -> ResponseSet
    # Replogle 2022 *_raw_bulk_01.h5ad: X = mean raw counts per cell per guide (rows), obs index
    # "<id>_<GENE>_<transcript>_<ENSG>", controls have GENE "non-targeting"; obs has num_cells_filtered,
    # fold_expr (fraction of target expression remaining), energy_test_p_value; var has gene_name.
    # The transcript field may itself contain "_" (a RefSeq id: "10598_ZNF735_NM_001159524_ENSG00000223614"
    # → ZNF735), so the symbol is parsed with the transcript tokens in mind, not with a plain rsplit.
    # Rows of one target are pooled weighted by num_cells_filtered. CPM is computed per row over all its genes.
    # SE: per-gene noise from non-targeting guides (each guide row = a pseudobulk of n_j cells):
    #   sigma2_g = mean_j(n_j * l_jg^2), l_jg = log2FC of guide j vs pooled controls;
    #   se(t,g) = sqrt(sigma2_g / n_t + sigma2_g / N_ctrl)   (plus a Poisson floor for low counts).
    # meta: n_cells = pooled cells, kd_remaining = cell-weighted mean fold_expr, phenotype_p = min energy p.

def h1_pseudobulk(h5ad_paths, genes, name="h1_2025", min_cells=50, max_cells_per_target=None, seed=0) -> ResponseSet
    # Arc VCC-2025 single-cell files (obs target_gene, control "non-targeting"; var index = symbols; raw counts).
    # Read in backed mode, one target at a time (bounded memory). Control cells appear in every file: use them
    # once. target_cpm / ctrl_cpm = arithmetic mean of per-cell CPM (the scorer's definition);
    # se via the delta method: se = (1/ln2) * sqrt(v_t/(n_t (x_t+eps)^2) + v_c/(n_c (c+eps)^2)).
    # meta: n_cells; kd_remaining = target's own mean CPM / control mean CPM; phenotype_p = nan.
```

## `vccpipe.cis`

CRISPRi (dCas9-KRAB) also silences genes whose promoters lie near the target's. Those genes are scored
(only the target's own row is excluded), so we predict them from genome distance.

```python
def load_tss(gtf_path) -> pd.DataFrame   # columns gene_name, chrom, tss (int, 1-based), strand; one row per
                                         # gene_name (gene records; gene_type protein_coding or lncRNA; first wins).
                                         # Use GENCODE v32 basic: its symbols match the VCC axis. Newer releases
                                         # rename ~500 axis genes (e.g. MARCH2 → MARCHF2), which then get no TSS.

@dataclass
class CisModel:
    bins_kb: list[float] = [0, 1, 2, 5, 10, 20, 50]      # distance bins between TSSs (kb)
    effect: list[float] | None = None                    # fitted median log2FC per bin (same length - 1)
    n_pairs: list[int] | None = None
    def fit(self, rs: ResponseSet, tss: pd.DataFrame, min_cells=50) -> "CisModel"
        # every (target, measured gene ≠ target) pair with TSS distance < 50 kb on the same chromosome;
        # effect[b] = median of the source's log2FC in bin b (targets with n_cells >= min_cells)
    def predict(self, targets, genes, tss) -> np.ndarray  # (T,G) log2FC prior (0 outside the window/unknown TSS;
                                                          #  the target's own column left 0)
    def to_json(self) -> str; @classmethod from_json(cls, s) -> "CisModel"
```

## `vccpipe.model`

Prediction for target `t`, gene `g`, new context `C` (controls only), as a log2 fold change vs C's controls:

```
lfc_C(t,g) = mask_C(g) · [ gamma·gamma_scale · T(g) · use_template + beta·beta_scale · D(t,g) · use_specific ]
then, where D(t,g) ≠ 0 and ctrl_cpm_C(g) > DE_GATE_CPM (5):
lfc_C(t,g) = sign(lfc_C(t,g)) · max(|lfc_C(t,g)|, min_called_lfc)                       (effect-size floor)
then        + use_cis · cis(t,g)                                                        (only where no source has t)
lfc_C(t,t) = kd_lfc                                                                     (the knockdown itself)
```

* `T(g)` = **template**: mean over the predicted panel's targets that a source measured, of that source's log2FC
  (each target's own gene left out), averaged over sources with their weights. Only the sources named in
  `template_sources` contribute (`None` = every source with weight > 0). v1: `["k562_gw"]`.
* `D(t,g)` = **specific response**, denoised: per source `soft(z_dev) · (lfc_s(t,g) − T_s(g))`, gated on the z of
  the **template-removed deviation**: `z_dev = (lfc_s(t,g) − T_s(g)) / se_s(t,g)` (non-finite → 0),
  `soft(z) = clip((|z| − z0) / z_soft, 0, 1)` (`z_soft <= 0` → hard gate `|z| > z0`); averaged over sources that
  measured `t` (source weights). Targets whose source `phenotype_p > null_p` get `D = 0` ("no detectable response"
  → predict no specific change).
* `mask_C(g)` = 1 if C's control mean CPM ≥ `expr_floor_cpm`, else 0 (no changes predicted for unexpressed genes).
  The floor (0.5 CPM) is below the scorer's gate (> 5 CPM), so it never removes a testable gene.
* `beta`, `gamma` are fitted by weighted least squares on a held-out line (H1) with known responses.
  `beta_scale`, `gamma_scale` and `min_called_lfc` sit on top of the fitted gains and are never touched by the fit;
  they, `z0`, `null_p`, the generator and its spread are chosen on the H1 proxy with the official scorer.

```python
@dataclass
class ModelParams:
    z0: float = 2.0; z_soft: float = 1.0; beta: float = 1.0; gamma: float = 1.0
    beta_scale: float = 1.0; gamma_scale: float = 1.0    # multipliers on the fitted gains (fit_gains ignores them)
    min_called_lfc: float = 0.0                          # effect-size floor where D != 0 and ctrl CPM > 5
    null_p: float = 0.05; expr_floor_cpm: float = 0.5; kd_lfc: float = -2.7
    use_template: bool = True; use_specific: bool = True; use_cis: bool = True
    source_weights: dict[str, float] = {}          # missing source → weight 1; weight 0 drops the source
    template_sources: list[str] | None = None      # None = every source with weight > 0; else only these names
    def to_json(self) -> str; @classmethod from_json(cls, s)   # round-trips the list and None

class EffectModel:
    def __init__(self, sources: list[ResponseSet], params: ModelParams, cis: CisModel | None = None,
                 tss: pd.DataFrame | None = None)
    def components(self, targets) -> dict          # 'template' (G,), 'specific' (T,G), 'has_source' (T,) bool,
                                                   # 'cis' (T,G)
    def predict_lfc(self, targets, ctrl_cpm) -> np.ndarray   # (T,G), the formula above
    def predict_cpm(self, targets, ctrl_cpm) -> np.ndarray   # (T,G) = apply_log2fc(ctrl_cpm, lfc, EPS_CPM)

def fit_gains(model: EffectModel, truth: ResponseSet, targets=None, max_weight=100.0) -> tuple[float, float]
    # Least squares of truth.lfc() on [template, specific] over (target, gene) pairs where truth measured the gene,
    # the gene passes truth's expression floor AND is DE-testable in the truth line (truth.ctrl_cpm > DE_GATE_CPM),
    # and the gene is not the target; weights min(1/se^2, max_weight). It estimates the RAW gains: internally
    # beta_scale = gamma_scale = 1 and min_called_lfc = 0. Returns (beta, gamma) and does NOT modify the model
    # (the caller sets params.beta / params.gamma and keeps its own scales and floor).
```

v1 configuration: sources = [K562] only (`source_weights: {k562_gw: 1.0, h1_2025: 0.0}`), so the submission is
the model that 02 validated; H1 is the truth line for `fit_gains` and the scoring proxy, never a source.

## `vccpipe.generator`

Both generators share one call signature: `generator(target_cpm (G,), n_cells, rng) -> CSR integer counts`, they
preserve the predicted mean CPM in expectation, and they guarantee the upload rules (non-negative integers,
< 1e6 counts per cell, every gated gene has at least one count).

```python
class TemplateGenerator:
    def __init__(self, ctrl_X, spread=0.0)
    # Edits randomly drawn real control cells: binomial thinning (down), added Poisson counts (up); spread > 0
    # makes cells respond unevenly (Beta / Gamma draws with the same mean). Open rules question: copies controls.
    # 03 refuses it unless --allow-control-templates.

class NBGenerator:                                 # the default for submissions
    def __init__(self, ctrl_X, max_phi=50.0)
    # Depth-aware negative binomial per gene, fitted on the controls (no control cell is copied):
    #   control cells are binned into 10 library-size deciles;
    #   H[d] = mean per-cell CPM in decile d / overall mean per-cell CPM (1 where the overall mean is 0);
    #   a generated cell copies the library size AND decile of a randomly drawn control cell, and
    #   mu = target_cpm * H[dec] * lib / 1e6; counts ~ NB(mu, phi_g) (method-of-moments phi, clipped to [0, max_phi]).
```

## `vccpipe.harness` and `vccpipe.submission`

```python
class Harness:                                     # official scorer (cell-eval2, vcc2026 profile) on a proxy
    def __init__(self, real_h5ad, workdir, gpu=False)
    def build_bundle(self, force=False); def score(self, name, pred_h5ad, keep_pred=False)
    def per_perturbation(self, name)

def write_prediction(path, targets, pred_cpm, generator, ctrl_X, genes, seed=0, cells_per_pert=400)
    # one block per target + the proxy's controls. RNG seeded PER TARGET: rng = np.random.default_rng([seed, i])
    # for target i, so every variant is scored on identical template / library-size draws (paired comparison).

def build_submission(predict_cpm, make_generator, controls_dir, out_h5ad, workdir, cells_per_pert=400,
                     chunk_targets=50, seed=0)
    # contexts by name from the bundle's manifest; chunks on disk; RNG seeded per (context, target):
    # np.random.default_rng([seed, context_index, target_index]).
def vcc_prep(sub_h5ad, controls_dir, contexts, out_vcc=None, dry_run=True, cells_per_pert=None)
    # vcc prep writes ~20 GB of temporary files to $TMPDIR while packaging (RUNBOOK §5b sets it).
```

## `vccpipe.proxy`

```python
def build_h1_proxy(h5ad_paths, genes, out_h5ad, n_ctrl=18400, kd_min=0.8, median_umi=20000, seed=0,
                   targets=None, max_targets=None) -> list[str]
    # VCC-format proxy (400 cells per kept target + n_ctrl controls, VCC gene axis, thinned to ~20k UMI);
    # returns the kept targets. Same logic as notebook 01 §4b, written directly to out_h5ad.
```

**Strength filter (applied by 01 through `targets=`).** H1's panel responds far more strongly than the VCC panel,
so H1 targets whose K562 response has more than `proxy.max_strong_genes` (50) genes with |z| > `proxy.strong_z`
(4.5), counted among genes with K562 control CPM > 5, are left out of the proxy. No VCC validation target reaches
50 (max 42). Targets K562 never measured are kept. 02 records `proxy: {n_targets, excluded_strong}` in its
best_params files.

## Scripts and config

`configs/v1.yaml` holds paths, data URLs (GENCODE v32 basic), model defaults (`source_weights: h1_2025: 0.0`), the
proxy filter, and the validation grids. Every script takes `--config` and `--smoke` (tiny sizes for a laptop test
with fake data). See RUNBOOK.md.

* **01_build_sources.py** (`--reuse`, `--skip-proxy`): response tables, TSS table, cis model, the strength-filtered
  H1 proxy.
* **02_validate.py** (`--gpu`, `--resume`): stage 1 = the grid `z0` ∈ [2.5, 3.5, 4.5] × `null_p` ∈ [0.05, 1.0] ×
  {template s0, template s0.5, nb} + ablations + `control_paste`; stage 2 = `beta_scale` ∈ [1, 2, 3] ×
  `min_called_lfc` ∈ [0.0, 0.4] around the stage-1 winner of each generator. Gains are cross-fitted (2 folds).
  Only `grid` and `stage2` variants can win (ablations are diagnostics). Writes `best_params.json` (overall winner)
  and `best_params_template.json` / `best_params_nb.json` (best per generator). `--resume` keeps only the rows
  scored on the same proxy with the same engine. Each has the keys `variant`, `model` (ModelParams JSON incl. the new fields, `template_sources =
  ["k562_gw"]`), `generator`, `spread`, `fitted_gains {beta, gamma}` (refitted on all proxy targets),
  `cv_scores {...}`, `validation_sources ["k562_gw"]`, `proxy {n_targets, excluded_strong}`.
* **03_make_submission.py** (`--controls-dir`, `--params`, `--out`, `--dry-run`, `--keep-h5ad`,
  `--allow-control-templates`): refuses generator `template` unless `--allow-control-templates` (open rules
  question) and points to `best_params_nb.json`; uses the params file's `template_sources`; reindexes the sources
  by symbol when the bundle's gene axis differs (prints new / dropped counts) instead of exiting; prints per
  context how many DE-testable genes the template alone moves by > 0.2 log2 for every target; packages with
  `vcc prep`, writes `<file>.vcc.sha256`, then deletes the `.h5ad` unless `--keep-h5ad`; prints a neutral submit
  command,
  `vcc submit <file>.vcc -m "<name>-<UTC date>" --wait` (no `-d`: names and descriptions are public). It never
  submits.
