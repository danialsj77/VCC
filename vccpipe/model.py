"""The effect model (v1): predicted log2 fold change per (target, gene) in a new cell line.

What it predicts
----------------
For a new context C we only see its control cells. For target t and gene g we predict the log2 fold change of the
mean CPM of t's knockdown cells vs C's controls (the quantity the scorer's DE metrics compare):

    x(t,g)     = mask_C(g) · [ gamma·gamma_scale · T(g) + beta·beta_scale · D(t,g) ]
    x(t,g)     = sign(x) · max(|x|, min_called_lfc)      where D(t,g) != 0 and ctrl_C(g) > 5 CPM ("called" genes)
    lfc_C(t,g) = x(t,g) + cis(t,g) · [no source measured t]
    lfc_C(t,t) = kd_lfc

* T(g), the **template**: the part of the response that every knockdown in a line shares (CRISPRi machinery,
  stress, proliferation programs; ~28% of response variance in published analyses). We estimate it as the mean
  response over the panel's targets in each public source line (only the lines in `template_sources`, if set).
  It applies to every target, measured or not.
* D(t,g), the **specific response**: what is particular to knocking down t, i.e. the source's response minus the
  source's own template, kept only where that deviation is confident (soft gate on its z-score), and set to 0 for
  targets whose knockdown had no detectable phenotype in the source. "Predict no change unless confident" protects
  the Jaccard and direction metrics: on our harness, noisy small changes on many genes cut jac from 1.10 to 0.30.
* mask_C(g): no change is predicted for genes C barely expresses. The scorer never tests them (its gene gate is
  control CPM > 5), and on near-zero counts a fold change only adds noise to the generated cells.
* cis(t,g): CRISPRi also silences genes whose promoters are near t's. A source that measured t already contains
  that silencing (it shows up in D), so the genome-distance prior is used only for targets no source measured.
* kd_lfc: the knockdown of the target itself. Its own row/column is excluded by every metric, but it still enters
  the pseudobulk normalization, so we give it a realistic CRISPRi value (2^-2.7 ≈ 15% remaining).
* beta and gamma rescale the transferred pieces to the new line (effects do not transfer at full size across
  lines). They are fitted with `fit_gains` on a held-out line with known answers (H1), never on the target line.
* beta_scale, gamma_scale and min_called_lfc are tuned on the scorer, not fitted (fit_gains ignores them and
  estimates the raw gains). Why they exist: least squares on noisy, partly transferring responses shrinks beta
  (regression dilution, ~0.4-0.5 in simulation), and a K562 change of -0.5 shrunk to -0.2 is rarely called by the
  scorer's Wilcoxon test with 400 cells. Under-calling is what the direction metric punishes most (it is 0 when we
  call nothing on a target the truth calls), so 02_validate.py tries larger scales and a minimum size for genes the
  model does call, around each generator's best grid variant ("stage 2").

Design reference: docs/DESIGN_v1.md (`vccpipe.model`); reasons: METHOD.md §3.

Memory
------
Sources can be large (K562 genome-wide: ~11k targets on the 18,533-gene VCC axis, ~0.8 GB per float32 matrix).
The model therefore never calls `lfc()` / `z()` on a whole source: it first takes `source.subset(panel targets)`
(≤ 300 rows) and works on that. All (T, G) work arrays are float32 (300 × 18,533 × 4 B ≈ 22 MB each), and only a
handful exist at once. `fit_gains` accumulates its sums in row chunks.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field, fields

import numpy as np
import pandas as pd

from .common import DE_GATE_CPM, apply_log2fc     # DE_GATE_CPM = 5: the scorer DE-tests genes above this control CPM

try:
    from .sources import EPS_CPM
except ImportError:  # sources.py is written in parallel; keep this value identical to vccpipe.sources.EPS_CPM
    EPS_CPM = 0.5    # pseudocount (CPM) for every log2 fold change in the model (DESIGN_v1.md)

__all__ = ["ModelParams", "EffectModel", "fit_gains", "soft_weight", "EPS_CPM"]

_FIT_CHUNK_ROWS = 64          # targets per chunk in fit_gains (bounds float64 temporaries to ~64 × G × 8 B each)


# ----------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class ModelParams:
    """All tunable settings of the effect model, serializable to JSON (best_params.json, configs).

    z0, z_soft      soft confidence gate on the z-score of a source's template-removed deviation,
                    z_dev = (lfc_s - T_s) / se_s:  weight = clip((|z_dev| - z0) / z_soft, 0, 1).
                    z_soft <= 0 means a hard gate (|z_dev| > z0).
    beta, gamma     gains on the specific response D and on the template T (fitted by `fit_gains`).
    beta_scale,     extra multipliers on top of the fitted gains (effective gains: beta·beta_scale and
    gamma_scale     gamma·gamma_scale). fit_gains never reads or changes them, so a value chosen on the scorer
                    survives refitting the gains (02_validate.py stage 2). 1.0 = use the fitted gains as they are.
    min_called_lfc  minimum |log2FC| of a "called" gene: where the specific response D(t,g) != 0 and the new
                    context's control CPM > 5 (genes the scorer DE-tests), the prediction keeps its sign and gets
                    at least this size. 0.0 = off. Must be >= 0.
    null_p          a source target whose phenotype_p > null_p gets D = 0 (no detectable response in that source).
                    nan p-values are kept. Set null_p = 1.0 to disable the gate (p-values never exceed 1).
    expr_floor_cpm  genes whose control mean CPM in the new context is below this get no predicted change
                    (except the cis prior and the target's own knockdown).
    kd_lfc          log2 fold change written into the target's own gene.
    use_*           switch components on/off (ablations).
    source_weights  {source name: weight >= 0}; a missing name means weight 1; weight 0 drops the source.
    template_sources  None = every source with weight > 0 contributes to the template T. A list of source names =
                    only those sources contribute to T (the others still give their specific response D, from which
                    their OWN template is removed). Used to keep the submitted template equal to the validated one
                    (["k562_gw"]) when more sources are loaded: gamma is fitted for that template and has no
                    meaning for another one. [] gives a zero template.
    """
    z0: float = 2.0
    z_soft: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    beta_scale: float = 1.0
    gamma_scale: float = 1.0
    min_called_lfc: float = 0.0
    null_p: float = 0.05
    expr_floor_cpm: float = 0.5
    kd_lfc: float = -2.7
    use_template: bool = True
    use_specific: bool = True
    use_cis: bool = True
    source_weights: dict[str, float] = field(default_factory=dict)
    template_sources: list[str] | None = None

    def __post_init__(self):
        # Normalize template_sources to a fresh list of str (or None). A tuple would otherwise not survive a JSON
        # round trip unchanged (it comes back as a list), and a list shared between two params objects (e.g. after
        # dataclasses.replace) could be changed through the other one. A bare string is refused: "k562_gw" would
        # otherwise be read as the sources "k", "5", "6", ... and silently give a zero template.
        ts = self.template_sources
        if ts is None:
            return
        if isinstance(ts, (str, bytes)):
            raise TypeError(f"ModelParams.template_sources must be a list of source names or None, got {ts!r} "
                            f"(write [{ts!r}] for a single source)")
        self.template_sources = [str(x) for x in ts]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)  # template_sources: a JSON list, or null for None

    @classmethod
    def from_json(cls, s) -> "ModelParams":
        """Parse a JSON string (a dict, e.g. from a YAML config, is accepted too).

        Unknown keys raise an error rather than being ignored: a typo such as "zo" in a config would otherwise
        silently fall back to the default and we would validate a different model than we think.
        """
        d = json.loads(s) if isinstance(s, (str, bytes, bytearray)) else dict(s)
        spec = {f.name: f for f in fields(cls)}
        unknown = sorted(set(d) - set(spec))
        if unknown:
            raise ValueError(f"unknown ModelParams field(s): {unknown}")
        kw = {}
        for name, v in d.items():
            default = getattr(cls(), name)
            if name == "source_weights":
                kw[name] = {str(k): float(x) for k, x in (v or {}).items()}
            elif name == "template_sources":
                kw[name] = v                    # None or a list of names; checked and normalized in __post_init__
            elif isinstance(default, bool):
                if not isinstance(v, (bool, int, np.bool_)):
                    raise TypeError(f"ModelParams.{name} must be a boolean, got {v!r}")
                kw[name] = bool(v)
            else:
                kw[name] = float(v)
        return cls(**kw)


def _check_params(p: ModelParams):
    for name in ("z0", "z_soft", "beta", "gamma", "beta_scale", "gamma_scale", "min_called_lfc",
                 "expr_floor_cpm", "kd_lfc"):
        if not np.isfinite(getattr(p, name)):
            raise ValueError(f"ModelParams.{name} must be finite, got {getattr(p, name)}")
    if p.min_called_lfc < 0:                    # a negative floor would be a silent no-op: most likely a sign typo
        raise ValueError(f"ModelParams.min_called_lfc must be >= 0, got {p.min_called_lfc}")
    for k, w in p.source_weights.items():
        if not (np.isfinite(w) and w >= 0):
            raise ValueError(f"source weight for {k!r} must be finite and >= 0, got {w}")
    if p.template_sources is not None and (isinstance(p.template_sources, (str, bytes))
                                           or not all(isinstance(x, str) for x in p.template_sources)):
        # __post_init__ normalizes at construction; this catches a value assigned later (params are mutable).
        raise TypeError(f"ModelParams.template_sources must be None or a list of source names, "
                        f"got {p.template_sources!r}")


def soft_weight(z, z0: float, z_soft: float) -> np.ndarray:
    """Soft confidence gate: 0 for |z| <= z0, rising linearly to 1 at |z| = z0 + z_soft (hard step if z_soft <= 0).

    Why soft: a hard cut makes predictions jump when a z-score crosses the line by a hair; the ramp lets weakly
    supported changes in at reduced size and makes z0 easier to tune on the proxy.
    The model applies it to z_dev = (lfc_s − T_s) / se_s, the z of the template-removed deviation (components()).
    """
    a = np.abs(np.asarray(z, dtype=np.float32))
    np.nan_to_num(a, copy=False, nan=0.0)
    return _soft_inplace(a, z0, z_soft)


def _soft_inplace(absz: np.ndarray, z0: float, z_soft: float) -> np.ndarray:
    """`soft_weight` on an |z| array that we own (overwritten to save a (T, G) copy)."""
    if z_soft > 0:
        absz -= np.float32(z0)
        absz /= np.float32(z_soft)
        np.clip(absz, 0.0, 1.0, out=absz)
        return absz
    return (absz > z0).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------------------------------------

def _as_target_list(targets) -> list[str]:
    if isinstance(targets, str):
        raise TypeError("targets must be a list of gene symbols, not a single string")
    out = [str(t) for t in targets]
    if len(set(out)) != len(out):
        dup = pd.Index(out)[pd.Index(out).duplicated()].unique().tolist()
        raise ValueError(f"targets must be unique; duplicated: {dup[:10]}")
    return out


class EffectModel:
    """Template + gated specific response transferred from public source lines, plus a cis prior.

    sources   list of ResponseSet (vccpipe.sources), all on the same gene axis (the VCC axis)
    params    ModelParams; read at every call, so the caller may change it (e.g. after fit_gains) in place
    cis, tss  optional CisModel (vccpipe.cis) and its TSS table; used only for targets that no source measured
    """

    def __init__(self, sources, params: ModelParams, cis=None, tss: pd.DataFrame | None = None):
        self.sources = list(sources)
        if not self.sources:
            raise ValueError("EffectModel needs at least one source (it defines the gene axis)")
        self.genes = np.asarray(self.sources[0].genes).astype(str)
        for s in self.sources[1:]:
            if not np.array_equal(np.asarray(s.genes).astype(str), self.genes):
                raise ValueError(f"source {s.name!r} is not on the same gene axis as {self.sources[0].name!r}")
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"source names must be unique (weights are keyed by name): {names}")
        self.params = params
        self.cis = cis
        self.tss = tss
        self._gidx = {g: i for i, g in enumerate(self.genes)}
        self._target_sets = [set(map(str, s.targets)) for s in self.sources]
        unknown = set(params.source_weights) - set(names)
        if unknown:
            warnings.warn(f"source_weights names no loaded source: {sorted(unknown)} (sources: {names})")
        if params.template_sources is not None:
            # A misspelled name here would silently drop that line from the template (or zero it entirely).
            unknown_t = sorted(set(params.template_sources) - set(names))
            if unknown_t:
                warnings.warn(f"template_sources names no loaded source: {unknown_t} (sources: {names})")
            if params.use_template and not any(self._in_template(s) and self._weight(s) > 0 for s in self.sources):
                warnings.warn(f"no loaded source with weight > 0 is in template_sources={params.template_sources}:"
                              f" the template is 0 everywhere")

    # -- helpers -------------------------------------------------------------------------------------------------

    def _weight(self, s) -> float:
        return float(self.params.source_weights.get(s.name, 1.0))

    def _in_template(self, s) -> bool:
        """Does source s contribute to the combined template T? (params.template_sources; None = every source)"""
        ts = self.params.template_sources
        return ts is None or s.name in ts

    def _own_columns(self, targets) -> np.ndarray:
        """Column of each target's own gene on the axis (-1 if the symbol is not on the axis)."""
        return np.fromiter((self._gidx.get(t, -1) for t in targets), dtype=np.int64, count=len(targets))

    def _check_ctrl(self, ctrl_cpm) -> np.ndarray:
        c = np.asarray(ctrl_cpm, dtype=np.float64).ravel()
        if c.shape != self.genes.shape:
            raise ValueError(f"ctrl_cpm has {c.size} genes, the model's axis has {self.genes.size}")
        if not np.all(np.isfinite(c)) or (c < 0).any():
            raise ValueError("ctrl_cpm must be finite and non-negative")
        return c

    @staticmethod
    def _source_template(L: np.ndarray, measured: np.ndarray, own: np.ndarray):
        """One source's template T_s(g): mean over its panel rows of lfc_s(t, g), on genes it measured.

        Leave-self-out: the entry lfc_s(t, t) (t's own gene, knocked down to ~-3) is left out of gene t's mean.
        A knockdown is not part of the *shared* response; with few panel targets in a source (H1: 25) it would
        otherwise pull T(t) down by ~3/25 = 0.12 and shift every other target's prediction at gene t.
        Returns (T_s (G,) float64, defined (G,) bool).
        """
        n = L.shape[0]
        total = L.sum(axis=0, dtype=np.float64)
        count = np.where(measured, float(n), 0.0)
        r = np.flatnonzero(own >= 0)
        c = own[r]
        np.subtract.at(total, c, L[r, c].astype(np.float64))
        np.subtract.at(count, c, measured[c].astype(np.float64))
        defined = count > 0
        tmpl = np.divide(total, count, out=np.zeros_like(total), where=defined)
        return tmpl, defined

    @staticmethod
    def _phenotype_gate(sub, null_p: float) -> np.ndarray:
        """Rows (of a source subset) whose knockdown showed no detectable phenotype: phenotype_p > null_p."""
        n = len(sub.targets)
        meta = getattr(sub, "meta", None)
        if meta is None or "phenotype_p" not in meta.columns:
            return np.zeros(n, dtype=bool)
        pv = pd.to_numeric(meta["phenotype_p"].reindex(list(map(str, sub.targets))), errors="coerce")
        return pv.to_numpy(dtype=np.float64, na_value=np.nan) > null_p     # nan > x is False: unknown → keep

    def _cis_prior(self, targets) -> np.ndarray:
        """CisModel.predict for every target (T, G) float32; zeros when no cis model / TSS table was given."""
        T, G = len(targets), self.genes.size
        if self.cis is None or self.tss is None:
            return np.zeros((T, G), dtype=np.float32)
        arr = np.asarray(self.cis.predict(targets, self.genes, self.tss), dtype=np.float32)
        if arr.shape != (T, G):
            raise ValueError(f"CisModel.predict returned shape {arr.shape}, expected {(T, G)}")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        own = self._own_columns(targets)
        r = np.flatnonzero(own >= 0)
        arr[r, own[r]] = 0.0                      # the knockdown itself is kd_lfc's job
        return arr

    # -- public API ----------------------------------------------------------------------------------------------

    def components(self, targets) -> dict:
        """The model's building blocks for a panel of targets (independent of the new context).

        Returns
          'template'   (G,)  float64  combined template T(g) (over the sources in params.template_sources)
          'specific'   (T,G) float32  combined specific response D(t,g) (0 on the target's own gene)
          'has_source' (T,)  bool     some (non-zero-weight) source measured t
          'cis'        (T,G) float32  cis prior for every target (predict_lfc applies it only where ~has_source)

        The template is computed over *these* targets: it is the panel's average response, so pass the full
        panel you are predicting (the same list at fitting and at prediction time).
        None of these depend on the gains, their scales or min_called_lfc (those act in predict_lfc only).
        """
        p = self.params
        _check_params(p)
        targets = _as_target_list(targets)
        T, G, S = len(targets), self.genes.size, len(self.sources)
        pos = {t: i for i, t in enumerate(targets)}
        own = self._own_columns(targets)

        tmpl_num = np.zeros(G)                                  # Σ_s w_s T_s(g)   over sources defining T_s(g)
        tmpl_den = np.zeros(G)                                  # Σ_s w_s          over the same sources
        spec_num = np.zeros((T, G), dtype=np.float32)           # Σ_s w_s D_s(t,g)
        w_ts = np.zeros((T, S), dtype=np.float32)               # w_s if source s measured target t
        m_sg = np.zeros((S, G), dtype=np.float32)               # 1 if source s measured gene g

        for k, s in enumerate(self.sources):
            w = self._weight(s)
            present = [t for t in targets if t in self._target_sets[k]]
            if w == 0 or not present:
                continue
            sub = s.subset(present)                             # ≤ T rows: never the whole source matrix
            rows = np.array([pos[str(t)] for t in sub.targets], dtype=np.int64)
            m = np.asarray(s.measured, dtype=bool)

            # Source log2FC rows (our own float32 copy: modified in place below).
            L = np.array(sub.lfc(), dtype=np.float32)
            np.nan_to_num(L, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            L[:, ~m] = 0.0

            # Template of this source, then its contribution to the combined template. Sources are averaged with
            # their weights only (not by how many panel targets they cover): each line's template is one estimate
            # of "the shared response", and a line covering more targets does not make it a different line.
            # tmpl_s is always computed (D_s below needs it); it enters the combined template only for the sources
            # in params.template_sources. Example: the submission loads K562 and H1, but gamma was fitted for the
            # K562 template, so template_sources = ["k562_gw"] keeps T exactly as validated, while H1 still
            # contributes its own specific responses (H1 response minus H1's own template).
            tmpl_s, defined = self._source_template(L, m, own[rows])
            if self._in_template(s):
                tmpl_num += w * tmpl_s
                tmpl_den += w * defined

            # Specific response D_s = soft(z_dev) · (lfc_s − T_s). The template is removed *before* transfer because
            # a line's template is a property of that line (its machinery/stress state), not of the target; the
            # new line's template is predicted separately (gamma · T).
            # The gate looks at the z-score of the quantity it lets through, the deviation from the template:
            # z_dev = (lfc_s − T_s) / se_s. (Gating on z(lfc_s) instead would pass pure noise at full weight on
            # genes where the template itself is large, e.g. T = −0.75, se = 0.2 and a deviation of −0.25 has
            # z(lfc) = −5 but z_dev = −1.25; and it would zero a real +0.75 deviation whose lfc_s happens to be 0.)
            L -= tmpl_s.astype(np.float32)[None, :]             # L is now the deviation lfc_s − T_s
            absz = np.abs(L)                                    # new array: safe to overwrite
            with np.errstate(divide="ignore", invalid="ignore"):
                np.divide(absz, np.asarray(sub.se_lfc), out=absz, casting="same_kind")
            absz[~np.isfinite(absz)] = 0.0                      # se = 0, nan or 0/0 → no evidence (as in .z())
            soft = _soft_inplace(absz, p.z0, p.z_soft)
            del absz
            L *= soft
            del soft
            L[:, ~m] = 0.0                                      # z0 < 0 would otherwise let unmeasured genes in
            r_own = np.flatnonzero(own[rows] >= 0)
            L[r_own, own[rows][r_own]] = 0.0                    # own gene: handled by kd_lfc
            L[self._phenotype_gate(sub, p.null_p)] = 0.0        # "no detectable response" → no specific change

            # A gated or sub-threshold source still counts in the denominator: it measured t and says "no change",
            # which should pull the average towards 0 rather than be ignored.
            L *= np.float32(w)
            spec_num[rows] += L
            del L
            w_ts[rows, k] = w
            m_sg[k] = m

        template = np.divide(tmpl_num, tmpl_den, out=np.zeros(G), where=tmpl_den > 0)
        spec_den = w_ts @ m_sg                                  # (T,G): Σ w_s over sources measuring t and g
        # In place: where no source measured (t, g) the numerator is already 0, which is the answer there.
        specific = np.divide(spec_num, spec_den, out=spec_num, where=spec_den > 0)
        del spec_den
        has_source = w_ts.sum(axis=1) > 0
        return {"template": template, "specific": specific, "has_source": has_source,
                "cis": self._cis_prior(targets)}

    def predict_lfc(self, targets, ctrl_cpm) -> np.ndarray:
        """Predicted log2 fold change vs the new context's control mean CPM, (T, G) float32.

        Steps (module docstring has the formula):
          1. gamma·gamma_scale·T + beta·beta_scale·D   (each term only if its use_* switch is on)
          2. mask_C: 0 on genes the new context barely expresses (control CPM < expr_floor_cpm)
          3. min_called_lfc: on "called" entries (D != 0, control CPM > 5) the size is raised to at least
             min_called_lfc, keeping the sign of the step-2 value (an exact 0 stays 0)
          4. + cis prior for targets no source measured (their D is 0 everywhere, so step 3 never touches them)
          5. own gene = kd_lfc
        """
        p = self.params
        ctrl = self._check_ctrl(ctrl_cpm)
        targets = _as_target_list(targets)
        c = self.components(targets)
        T, G = len(targets), self.genes.size

        # Effective gains: the fitted gain times the scorer-tuned scale (both 1.0 = plain fitted model).
        beta_eff = p.beta * p.beta_scale
        gamma_eff = p.gamma * p.gamma_scale
        specific_on = bool(p.use_specific) and beta_eff != 0

        lfc = np.zeros((T, G), dtype=np.float32)
        if specific_on:
            np.multiply(c["specific"], np.float32(beta_eff), out=lfc)
        if p.use_template and gamma_eff != 0:
            lfc += (np.float32(gamma_eff) * c["template"].astype(np.float32))[None, :]
        lfc[:, ctrl < p.expr_floor_cpm] = 0.0                   # mask_C: no change on genes C barely expresses

        # Minimum size of a called change. The scorer's Wilcoxon test on 400 cells rarely calls shifts of ~0.2
        # log2 (simulation: 0% at 100 CPM, 20% at 500 CPM), so a gene the model believes in (D != 0) but predicts
        # at a tiny size is effectively not called; this lifts it to a size the test can see. Only DE-testable
        # genes (control CPM > 5) are touched: the others are never scored as DE, and a floor there would only
        # add noise to the generated cells. With the specific term switched off (use_specific False or an
        # effective beta of 0) the model calls nothing, so the floor is not applied either.
        if p.min_called_lfc > 0 and specific_on:
            called = c["specific"] != 0                         # (T,G) bool: the confidence gate let D(t,g) in
            called &= (ctrl > DE_GATE_CPM)[None, :]
            floored = np.abs(lfc)
            np.maximum(floored, np.float32(p.min_called_lfc), out=floored)
            floored *= np.sign(lfc)                             # sign of the combined value; sign(0) = 0 stays 0
            np.copyto(lfc, floored, where=called)
            del called, floored

        if p.use_cis:
            no_src = ~c["has_source"]
            if no_src.any():
                lfc[no_src] += c["cis"][no_src]
        own = self._own_columns(targets)
        r = np.flatnonzero(own >= 0)
        lfc[r, own[r]] = np.float32(p.kd_lfc)
        return lfc

    def predict_cpm(self, targets, ctrl_cpm) -> np.ndarray:
        """Predicted mean CPM per target and gene, (T, G): control CPM moved by the predicted log2 fold change."""
        ctrl = self._check_ctrl(ctrl_cpm)
        return apply_log2fc(ctrl, self.predict_lfc(targets, ctrl), EPS_CPM)


# ----------------------------------------------------------------------------------------------------------------
# Fitting the gains on a line with known answers
# ----------------------------------------------------------------------------------------------------------------

def fit_gains(model: EffectModel, truth, targets=None, max_weight: float = 100.0) -> tuple[float, float]:
    """Fit (beta, gamma) by weighted least squares of truth.lfc() on [specific, template]. Model is not modified.

    model    EffectModel whose sources do NOT include `truth`'s line (otherwise the fit is circular)
    truth    ResponseSet of a held-out line (H1) with measured responses
    targets  the panel to compute the template over (default: truth's targets). Only its targets that truth
             measured enter the fit; the template is still the whole panel's, exactly as at prediction time.

    Pairs used: truth measured the gene, truth's control CPM >= params.expr_floor_cpm (the same mask the model
    applies in a new context) AND > DE_GATE_CPM (5 CPM: only genes the scorer DE-tests; genes at 0.5-5 CPM are
    never DE-tested and barely weighted by the MSE's log1p(CPM/20), but their noisy fold changes would pull the
    gains), gene != the target (excluded by all metrics; set by kd_lfc), finite truth lfc.
    Weights: min(1 / se², max_weight). The cap matters: without it a few very precise pairs (highly expressed
    genes) dominate the fit, whereas the DE metrics count every responding gene equally.

    For targets with no source, predict_lfc adds the cis prior; it is subtracted from y here (a fixed offset), so
    the gains are fitted on exactly the part of the prediction they scale.

    Raw gains: the fit is the linear model y ≈ beta·D + gamma·T, i.e. as if beta_scale = gamma_scale = 1 and
    min_called_lfc = 0. Those three fields are never read here and the model is not modified, so a caller doing
    `replace(params, beta=b, gamma=g)` keeps the scorer-tuned scales and floor on top of the refitted gains.

    Degenerate systems: a component that is switched off (use_template / use_specific False) or carries no
    signal on the fitted pairs is not fitted, and its current value in model.params is returned; nearly collinear
    columns fall back to fitting the template gain alone. Each fallback issues a warning.
    """
    p = model.params
    if not np.array_equal(np.asarray(truth.genes).astype(str), model.genes):
        raise ValueError("truth is not on the model's gene axis")
    panel = _as_target_list(truth.targets if targets is None else targets)
    comps = model.components(panel)
    truth_targets = set(map(str, truth.targets))
    used = [t for t in panel if t in truth_targets]
    if not used:
        raise ValueError("none of the requested targets were measured in truth")
    sub = truth.subset(used)
    pos = {t: i for i, t in enumerate(panel)}
    sub_targets = [str(t) for t in sub.targets]
    rows = np.array([pos[t] for t in sub_targets], dtype=np.int64)
    own = model._own_columns(sub_targets)

    Y = np.asarray(sub.lfc())
    SE = np.asarray(sub.se_lfc)
    truth_ctrl = np.asarray(truth.ctrl_cpm, dtype=np.float64)
    gene_ok = (np.asarray(truth.measured, dtype=bool)
               & (truth_ctrl >= p.expr_floor_cpm)                # the model's own expression mask
               & (truth_ctrl > DE_GATE_CPM))                     # genes the scorer DE-tests (strictly > 5 CPM)
    x1 = np.asarray(comps["template"], dtype=np.float64)        # template column (same for every target)
    D, cis, has = comps["specific"], comps["cis"], comps["has_source"]

    # Normal-equation sums, accumulated in float64 over row chunks. Column 0 = specific (beta), 1 = template
    # (gamma). Because x1 does not depend on t, its sums reduce to per-gene column sums of the weights.
    A = np.zeros((2, 2))
    b = np.zeros(2)
    sum_w = 0.0
    for start in range(0, len(rows), _FIT_CHUNK_ROWS):
        sl = slice(start, start + _FIT_CHUNK_ROWS)
        r = rows[sl]
        y = Y[sl].astype(np.float64)
        if p.use_cis:
            y -= cis[r] * (~has[r])[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / np.square(SE[sl].astype(np.float64))
        w = np.minimum(w, max_weight)                           # se = 0 → inf → capped; se = inf → 0
        w[~np.isfinite(w)] = 0.0                                # nan se → unusable
        w[:, ~gene_ok] = 0.0
        oc = own[sl]
        ok = np.flatnonzero(oc >= 0)
        w[ok, oc[ok]] = 0.0
        bad = ~np.isfinite(y)
        w[bad] = 0.0
        y[bad] = 0.0
        x2 = D[r].astype(np.float64)

        wx2 = w * x2
        A[0, 0] += np.sum(wx2 * x2)
        A[0, 1] += wx2.sum(axis=0) @ x1
        col_w = w.sum(axis=0)
        A[1, 1] += col_w @ (x1 * x1)
        b[0] += np.sum(wx2 * y)
        b[1] += (w * y).sum(axis=0) @ x1
        sum_w += col_w.sum()
    A[1, 0] = A[0, 1]

    current = np.array([p.beta, p.gamma], dtype=np.float64)
    names = ("specific (beta)", "template (gamma)")
    if sum_w <= 0:
        warnings.warn("fit_gains: no usable (target, gene) pairs; returning the current beta and gamma")
        return float(current[0]), float(current[1])

    enabled = (bool(p.use_specific), bool(p.use_template))
    active = []
    for k in (0, 1):
        if not enabled[k]:
            continue
        if A[k, k] > 1e-12 * sum_w:                             # weighted mean square of the column > 1e-12
            active.append(k)
        else:
            warnings.warn(f"fit_gains: the {names[k]} column is zero on all fitted pairs; keeping its current "
                          f"value {current[k]:g}")
    if len(active) == 2:
        det = A[0, 0] * A[1, 1] - A[0, 1] ** 2
        if det <= 1e-8 * A[0, 0] * A[1, 1]:                     # |corr| of the columns > 1 - 5e-9
            warnings.warn("fit_gains: specific and template columns are collinear; fitting gamma alone")
            active = [1]

    gains = current.copy()
    if len(active) == 2:
        gains[0] = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
        gains[1] = (A[0, 0] * b[1] - A[0, 1] * b[0]) / det
    elif len(active) == 1:
        k = active[0]
        gains[k] = b[k] / A[k, k]
    return float(gains[0]), float(gains[1])
