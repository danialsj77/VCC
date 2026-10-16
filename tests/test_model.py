"""Tests for vccpipe.model (effect model v1).

Run either way:
    .venv/bin/python -m pytest tests/test_model.py
    .venv/bin/python tests/test_model.py

Strategy: small ResponseSets are built directly from a chosen log2FC matrix (target CPM = control CPM moved by that
log2FC), so every expected value is known. The model's vectorized components are compared with a slow, loop-based
reference written straight from the formulas in docs/DESIGN_v1.md. One test runs at real scale (the VCC gene axis
and the 300-target panel from controls/) to check speed, memory, and that the model only reads panel rows of a
large source. No data files under data/ are used.
"""

import json
import sys
import tempfile
import time
import tracemalloc
import traceback
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd

from fakes import make_fake_gtf, vcc_genes
from vccpipe.common import apply_log2fc
from vccpipe.model import EPS_CPM, EffectModel, ModelParams, fit_gains, soft_weight

CONTROLS = ROOT / "controls"

try:                                                       # the real dataclass, once vccpipe/sources.py exists
    from vccpipe.sources import ResponseSet
    RS_KIND = "vccpipe.sources.ResponseSet"
except ImportError:                                        # minimal stand-in with the interface of DESIGN_v1.md
    RS_KIND = "stand-in ResponseSet (vccpipe.sources not importable yet)"

    @dataclass
    class ResponseSet:
        name: str
        genes: np.ndarray
        measured: np.ndarray
        targets: list
        target_cpm: np.ndarray
        ctrl_cpm: np.ndarray
        se_lfc: np.ndarray
        meta: pd.DataFrame

        def lfc(self, eps=EPS_CPM):
            out = np.log2((self.target_cpm.astype(np.float64) + eps) / (self.ctrl_cpm[None, :] + eps))
            out[:, ~self.measured] = 0.0
            return out.astype(np.float32)

        def z(self, eps=EPS_CPM):
            with np.errstate(divide="ignore", invalid="ignore"):
                out = self.lfc(eps) / self.se_lfc
            out[:, ~self.measured] = 0.0
            return out

        def subset(self, targets):
            idx = {t: i for i, t in enumerate(self.targets)}
            keep = [t for t in targets if t in idx]
            rows = [idx[t] for t in keep]
            return ResponseSet(self.name, self.genes, self.measured, keep, self.target_cpm[rows], self.ctrl_cpm,
                               self.se_lfc[rows], self.meta.loc[keep])


# ----------------------------------------------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------------------------------------------

def make_rs(name, genes, targets, lfc, ctrl_cpm, se, measured=None, phenotype_p=None):
    """A ResponseSet whose lfc() equals `lfc` on measured genes (target CPM = (ctrl+eps)·2^lfc − eps)."""
    genes = np.asarray(genes)
    T, G = len(targets), genes.size
    lfc = np.asarray(lfc, dtype=np.float64)
    measured = np.ones(G, bool) if measured is None else np.asarray(measured, bool)
    ctrl = np.where(measured, np.asarray(ctrl_cpm, dtype=np.float64), 0.0)
    tc = (ctrl[None, :] + EPS_CPM) * np.exp2(lfc) - EPS_CPM
    assert (tc[:, measured] >= -1e-6).all(), "test setup: log2FC too negative for this control CPM"
    tc = np.clip(tc, 0.0, None)
    tc[:, ~measured] = 0.0
    se = np.array(np.broadcast_to(np.asarray(se, dtype=np.float32), (T, G)))
    se[:, ~measured] = np.inf
    pp = np.full(T, np.nan) if phenotype_p is None else np.asarray(phenotype_p, dtype=np.float64)
    meta = pd.DataFrame({"n_cells": np.full(T, 200, dtype=np.int64), "kd_remaining": np.full(T, np.nan),
                         "phenotype_p": pp}, index=pd.Index(list(targets)))
    return ResponseSet(name=name, genes=genes, measured=measured, targets=list(targets),
                       target_cpm=tc.astype(np.float32), ctrl_cpm=ctrl, se_lfc=se, meta=meta)


class StubCis:
    """Duck-typed CisModel: fixed log2FC entries {target: {gene index: value}}; records the targets it was asked."""

    def __init__(self, entries):
        self.entries = entries
        self.calls = []

    def predict(self, targets, genes, tss):
        self.calls.append(list(targets))
        out = np.zeros((len(targets), len(genes)), dtype=np.float32)
        for i, t in enumerate(targets):
            for g, v in self.entries.get(t, {}).items():
                out[i, g] = v
        return out


class SpySource:
    """Wraps a ResponseSet; fails if lfc()/z() are computed on more rows than allowed (i.e. on the whole source)."""

    def __init__(self, rs, max_rows):
        self._rs, self._max = rs, max_rows

    def __getattr__(self, k):
        return getattr(self._rs, k)

    def lfc(self, *a, **k):
        assert len(self._rs.targets) <= self._max, "model computed lfc() on the whole source"
        return self._rs.lfc(*a, **k)

    def z(self, *a, **k):
        assert len(self._rs.targets) <= self._max, "model computed z() on the whole source"
        return self._rs.z(*a, **k)


def reference_components(sources, params, targets):
    """Slow, loop-based components straight from DESIGN_v1.md (+ the leave-self-out template, see model.py).

    The confidence gate uses the z-score of the template-removed deviation, z_dev = (lfc_s − T_s) / se_s
    (non-finite → 0); only sources in params.template_sources (None = all) enter the combined template.
    """
    genes = np.asarray(sources[0].genes).astype(str)
    gidx = {g: i for i, g in enumerate(genes)}
    T, G = len(targets), genes.size
    tn, td = np.zeros(G), np.zeros(G)
    sn, sd = np.zeros((T, G)), np.zeros((T, G))
    for s in sources:
        w = params.source_weights.get(s.name, 1.0)
        srow = {t: i for i, t in enumerate(s.targets)}
        present = [t for t in targets if t in srow]
        if w == 0 or not present:
            continue
        L, SE = np.asarray(s.lfc(), np.float64), np.asarray(s.se_lfc, np.float64)
        Ts, defined = np.zeros(G), np.zeros(G, bool)
        for g in range(G):
            if s.measured[g]:
                vals = [L[srow[t], g] for t in present if gidx.get(t) != g]
                if vals:
                    Ts[g], defined[g] = np.mean(vals), True
        if params.template_sources is None or s.name in params.template_sources:
            tn += w * Ts * defined
            td += w * defined
        for t in present:
            i, j = targets.index(t), srow[t]
            gated = s.meta.loc[t, "phenotype_p"] > params.null_p
            for g in range(G):
                if not s.measured[g]:
                    continue
                sd[i, g] += w
                if gated or gidx.get(t) == g:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = abs((L[j, g] - Ts[g]) / SE[j, g])
                z = z if np.isfinite(z) else 0.0
                soft = min(max((z - params.z0) / params.z_soft, 0.0), 1.0) if params.z_soft > 0 else float(
                    z > params.z0)
                sn[i, g] += w * soft * (L[j, g] - Ts[g])
    tmpl = np.divide(tn, td, out=np.zeros(G), where=td > 0)
    spec = np.divide(sn, sd, out=np.zeros((T, G)), where=sd > 0)
    has = sd.sum(axis=1) > 0
    return tmpl, spec, has


def small_world(seed=0, G=40):
    """Three sources with partial overlap of targets and genes, for the reference comparisons."""
    rng = np.random.default_rng(seed)
    genes = vcc_genes(CONTROLS, n=G)
    ctrl = 20 + rng.lognormal(3, 1, G)

    def rand_lfc(T):
        L = rng.normal(0, 0.15, (T, G))                 # small, mostly sub-threshold changes
        mask = rng.random((T, G)) < 0.25                # plus a quarter of genes with clear changes
        L[mask] += rng.normal(0, 1.2, mask.sum())
        return L

    # panel: genes[0..5] (on the axis, so own columns exist) + one symbol not on the axis
    panel = [str(g) for g in genes[:6]] + ["NOT_ON_AXIS"]
    ta = panel[0:4]                                   # a: t0..t3
    tb = [panel[1], panel[2], panel[4], "OTHER_B"]    # b: t1,t2,t4 (+ a non-panel target)
    tc_ = [panel[5]]                                  # c: t5 only, weight 0 → t5 must have no source
    ma = np.ones(G, bool)
    ma[-3:] = False
    mb = np.ones(G, bool)
    mb[10:14] = False
    a = make_rs("a", genes, ta, rand_lfc(len(ta)), ctrl, rng.uniform(0.1, 0.6, (len(ta), G)), ma,
                phenotype_p=[1e-6, np.nan, 0.5, 0.01])
    b = make_rs("b", genes, tb, rand_lfc(len(tb)), ctrl * 1.3, rng.uniform(0.1, 0.6, (len(tb), G)), mb,
                phenotype_p=[1e-6, 0.03, np.nan, 1e-3])
    c = make_rs("c", genes, tc_, rand_lfc(1), ctrl, 0.2)
    return genes, panel, [a, b, c], ctrl


# ----------------------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------------------

def test_params_json_roundtrip():
    p = ModelParams(z0=3.5, z_soft=0.5, beta=0.7, gamma=1.3, null_p=0.01, expr_floor_cpm=1.0, kd_lfc=-3.0,
                    use_template=False, use_specific=True, use_cis=False, source_weights={"k562_gw": 0.25, "h1": 2},
                    beta_scale=2.0, gamma_scale=0.5, min_called_lfc=0.4, template_sources=["k562_gw"])
    q = ModelParams.from_json(p.to_json())
    assert q == p and isinstance(q.source_weights["h1"], float)
    assert ModelParams.from_json(ModelParams().to_json()) == ModelParams()
    assert ModelParams.from_json('{"beta": 2}').beta == 2.0                      # missing keys → defaults
    assert ModelParams.from_json({"gamma": 0.5}).gamma == 0.5                    # dict accepted (YAML configs)
    for bad in ('{"zo": 1}', '{"use_cis": "false"}'):
        try:
            ModelParams.from_json(bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"from_json accepted {bad}")


def test_params_json_new_fields_list_and_none():
    """beta_scale / gamma_scale / min_called_lfc / template_sources survive JSON; the list and None both do."""
    p = ModelParams(beta_scale=2.0, gamma_scale=0.5, min_called_lfc=0.4, template_sources=["k562_gw"])
    d = json.loads(p.to_json())
    assert d["template_sources"] == ["k562_gw"]                                  # a real JSON list, not a repr
    assert (d["beta_scale"], d["gamma_scale"], d["min_called_lfc"]) == (2.0, 0.5, 0.4)
    q = ModelParams.from_json(p.to_json())
    assert q == p and q.template_sources == ["k562_gw"] and isinstance(q.template_sources, list)
    # None (the default: every source with weight > 0) is written as null and comes back as None
    d0 = json.loads(ModelParams().to_json())
    assert "template_sources" in d0 and d0["template_sources"] is None
    assert ModelParams.from_json(ModelParams().to_json()).template_sources is None
    assert ModelParams.from_json('{"template_sources": null}').template_sources is None
    assert ModelParams.from_json(ModelParams(template_sources=[]).to_json()).template_sources == []   # zero template
    # several names; a tuple (e.g. built in code) is normalized to a list, so it also survives the round trip
    two = ModelParams.from_json({"template_sources": ("k562_gw", "h1_2025")})
    assert two.template_sources == ["k562_gw", "h1_2025"]
    assert ModelParams(template_sources=("a",)) == ModelParams(template_sources=["a"])
    t = ModelParams(template_sources=("a", "b"))
    assert ModelParams.from_json(t.to_json()) == t
    # replace() (used by 02 after every gain fit) keeps the scales and gives an independent list
    a = ModelParams(beta_scale=3.0, min_called_lfc=0.4, template_sources=["k562_gw"])
    b = replace(a, beta=0.5, gamma=0.2)
    assert (b.beta_scale, b.min_called_lfc, b.template_sources) == (3.0, 0.4, ["k562_gw"])
    b.template_sources.append("x")
    assert a.template_sources == ["k562_gw"]
    # integers become floats; a params file written before these fields existed loads with the defaults
    assert isinstance(ModelParams.from_json('{"beta_scale": 2}').beta_scale, float)
    new = {"beta_scale", "gamma_scale", "min_called_lfc", "template_sources"}
    old = {k: v for k, v in json.loads(ModelParams(beta=0.4).to_json()).items() if k not in new}
    q = ModelParams.from_json(old)
    assert q.beta == 0.4 and (q.beta_scale, q.gamma_scale, q.min_called_lfc, q.template_sources) == (1, 1, 0, None)
    # a bare string is refused (it would otherwise be read as the sources "k", "5", "6", ...)
    for bad in ('{"template_sources": "k562_gw"}', {"template_sources": "k562_gw"}):
        try:
            ModelParams.from_json(bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f"from_json accepted {bad!r}")


def test_soft_weight_values():
    z = np.array([-3.0, -2.5, -1.5, 0.0, 2.0, 2.5, 3.0, 10.0, np.nan])
    assert np.allclose(soft_weight(z, 2.0, 1.0), [1, 0.5, 0, 0, 0, 0.5, 1, 1, 0])
    assert np.allclose(soft_weight(z, 2.0, 0.0), [1, 1, 0, 0, 0, 1, 1, 1, 0])   # z_soft = 0 → hard gate


def test_template_mean_leave_self_out_and_unmeasured():
    genes = np.array(["A", "B", "C", "D", "U"])
    L = np.array([[-3.0, 0.4, 0.2, -0.6, 0.0],         # target A (own column 0)
                  [0.8, -2.5, 0.6, -0.2, 0.0]])        # target B (own column 1)
    measured = np.array([True, True, True, True, False])
    src = make_rs("s", genes, ["A", "B"], L, np.full(5, 50.0), 0.1, measured)
    m = EffectModel([src], ModelParams())
    c = m.components(["A", "B", "X"])                 # X: no source
    # gene A: only B's row (A's own knockdown left out); gene B: only A's row; C, D: mean of both; U: unmeasured
    assert np.allclose(c["template"], [0.8, 0.4, 0.4, -0.4, 0.0], atol=1e-4)
    assert c["has_source"].tolist() == [True, True, False]
    assert np.all(c["specific"][2] == 0)
    # the template is a property of the panel's measured targets: adding an unmeasured target changes nothing
    assert np.allclose(m.components(["A", "B"])["template"], c["template"])


def test_soft_threshold_zeroes_low_z():
    genes = vcc_genes(CONTROLS, n=8)
    L = np.array([[0.3, -0.3, 0.5, -0.5, 1.0, -1.0, 2.0, 0.1],
                  [-0.1, 0.2, -0.4, 0.6, 0.8, 0.2, -1.5, 0.3]])
    zt = np.array([[1.0, 1.9, 2.5, 2.5, 3.0, 4.0, 9.0, 0.5],
                   [0.5, 2.25, 2.75, 1.0, 3.5, 1.95, 5.0, 2.6]])
    # The gate reads the z of the template-removed deviation, z_dev = (lfc − T) / se. With two targets that are
    # not on the axis (no own-column special case) T is the plain mean of the two rows, so this se gives
    # |z_dev| = zt exactly (the two rows never coincide, so no deviation is 0).
    se = np.abs(L - L.mean(axis=0)) / zt
    targets = ["T1", "T2"]
    src = make_rs("s", genes, targets, L, np.full(8, 40.0), se)
    m = EffectModel([src], ModelParams(z0=2.0, z_soft=1.0))
    c = m.components(targets)
    lfc = np.asarray(src.lfc(), np.float64)
    expected = np.clip(zt - 2.0, 0, 1) * (lfc - c["template"][None, :])
    assert np.allclose(c["specific"], expected, atol=1e-4)
    assert np.all(c["specific"][zt <= 2.0] == 0)        # low-z entries are exactly zero
    assert np.allclose(c["template"], lfc.mean(axis=0), atol=1e-5)   # the template itself is not gated
    m.params.z_soft = 0.0                               # hard gate
    c2 = m.components(targets)
    assert np.allclose(c2["specific"], (zt > 2.0) * (lfc - c2["template"][None, :]), atol=1e-4)


def test_gate_uses_template_removed_deviation():
    """The confidence gate reads z_dev = (lfc_s − T_s) / se_s, the z of what it lets through (review finding).

    Gene 0 has a large template: T = −0.75 (mean of −1.1, −0.9, −1.0, 0.0) and se = 0.2. Three targets deviate
    by −0.35, −0.15, −0.25 (|z_dev| ≤ 1.75: noise) and one by +0.75 (z_dev = 3.75: real, although its lfc is 0).
    A gate on z(lfc) would pass the three noise deviations at full weight (|z(lfc)| ≥ 4.5) and zero the real one.
    """
    genes = vcc_genes(CONTROLS, n=4)
    targets = ["T1", "T2", "T3", "T4"]                  # not on the axis: no own-column special case
    L = np.zeros((4, 4))
    L[:, 0] = [-1.1, -0.9, -1.0, 0.0]
    L[:, 1] = [1.3, 0.1, 0.0, 0.0]                      # gene 1: T = 0.35; deviations +0.95, −0.25, −0.35, −0.35
    se = np.full((4, 4), 0.2)
    se[0, 2] = np.inf                                   # gene 2: inf se → z_dev 0 → no evidence
    se[1, 2] = 0.0                                      #         se 0 → non-finite z_dev → 0 as well
    L[:, 2] = [1.0, -1.0, 0.0, 0.0]
    src = make_rs("s", genes, targets, L, np.full(4, 80.0), se)
    lfc = np.asarray(src.lfc(), np.float64)             # = L up to float32 rounding of the stored CPM
    for z_soft in (1.0, 0.0):                           # soft ramp and hard gate agree here (|z_dev| far from z0)
        m = EffectModel([src], ModelParams(z0=2.5, z_soft=z_soft, null_p=1.0))
        c = m.components(targets)
        assert np.isclose(c["template"][0], -0.75, atol=1e-5)
        D = c["specific"]
        assert np.allclose(D[:3, 0], 0.0) and np.isclose(D[3, 0], 0.75, atol=1e-4), D[:, 0]
        # gene 1: only target 1's +0.95 deviation (z_dev 4.75) passes
        assert np.allclose(D[:, 1], [0.95, 0, 0, 0], atol=1e-4), D[:, 1]
        # gene 2: the two strong entries have no usable se, the others deviate by 0
        assert np.all(D[:, 2] == 0)
        # and the vectorized model equals the loop reference (which also gates on z_dev)
        tmpl, spec, _ = reference_components([src], m.params, targets)
        assert np.allclose(D, spec, atol=1e-5) and np.allclose(c["template"], tmpl, atol=1e-6)
    # General form on the soft ramp: D = clip((|lfc − T|/se − z0)/z_soft, 0, 1) · (lfc − T), non-finite z → 0
    m = EffectModel([src], ModelParams(z0=1.0, z_soft=2.0, null_p=1.0))
    c = m.components(targets)
    dev = lfc - c["template"][None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        zdev = np.abs(dev / se)
    zdev[~np.isfinite(zdev)] = 0.0
    assert np.allclose(c["specific"], np.clip((zdev - 1.0) / 2.0, 0, 1) * dev, atol=1e-4)


def test_phenotype_gate():
    rng = np.random.default_rng(1)
    genes = vcc_genes(CONTROLS, n=30)
    targets = ["P1", "P2", "P3"]
    L = rng.normal(0, 1.0, (3, 30))
    a = make_rs("a", genes, targets, L, np.full(30, 60.0), 0.05, phenotype_p=[0.5, np.nan, 0.01])
    m = EffectModel([a], ModelParams(null_p=0.05, z0=0.0, z_soft=1e-6))
    D = m.components(targets)["specific"]
    assert np.all(D[0] == 0)                            # p = 0.5 > null_p → no specific response
    assert np.abs(D[1]).sum() > 1 and np.abs(D[2]).sum() > 1   # nan p kept; significant p kept
    # A gated source still counts as a "no change" vote in the average over sources.
    b = make_rs("b", genes, targets, L, np.full(30, 60.0), 0.05, phenotype_p=[0.001, 0.001, 0.001])
    m2 = EffectModel([a, b], ModelParams(null_p=0.05, z0=0.0, z_soft=1e-6, source_weights={"a": 1, "b": 3}))
    D2 = m2.components(targets)["specific"]
    Db = EffectModel([b], replace(m2.params, source_weights={})).components(targets)["specific"]
    assert np.allclose(D2[0], 0.75 * Db[0], atol=1e-5)
    # null_p = 1 disables the gate
    m.params.null_p = 1.0
    assert np.abs(m.components(targets)["specific"][0]).sum() > 1


def test_multi_source_weighted_average_matches_reference():
    genes, panel, sources, _ = small_world(seed=2)
    params = ModelParams(z0=1.5, z_soft=1.0, null_p=0.05, source_weights={"a": 1.0, "b": 3.0, "c": 0.0})
    m = EffectModel(sources, params)
    c = m.components(panel)
    tmpl, spec, has = reference_components(sources, params, panel)
    assert np.allclose(c["template"], tmpl, atol=1e-5)
    assert np.allclose(c["specific"], spec, atol=1e-5)
    assert c["has_source"].tolist() == has.tolist() == [True, True, True, True, True, False, False]
    a, b = sources[0], sources[1]
    alone = replace(params, source_weights={})          # one source: its weight does not matter
    Da = EffectModel([a], alone).components(panel)["specific"]
    Db = EffectModel([b], alone).components(panel)["specific"]
    both_g = a.measured & b.measured
    # t1 measured by both: weighted mean 1:3 on genes both measured; only a's value where b lacks the gene
    assert np.allclose(c["specific"][3, both_g], Da[3, both_g], atol=1e-6)        # t3: only a
    assert np.allclose(c["specific"][4], Db[4], atol=1e-6)                        # t4: only b
    assert np.allclose(c["specific"][1, both_g], (Da[1, both_g] + 3 * Db[1, both_g]) / 4, atol=1e-5)
    only_a = a.measured & ~b.measured
    assert np.allclose(c["specific"][1, only_a], Da[1, only_a], atol=1e-6)
    # template: weighted 1:3 where both define it; a's template alone where b did not measure the gene
    Ta = EffectModel([a], alone).components(panel)["template"]
    Tb = EffectModel([b], alone).components(panel)["template"]
    assert np.allclose(c["template"][both_g], (Ta[both_g] + 3 * Tb[both_g]) / 4, atol=1e-5)
    assert np.allclose(c["template"][only_a], Ta[only_a], atol=1e-6)
    # equal weights are the default; the reference agrees there too
    p2 = ModelParams(z0=1.5, z_soft=1.0)
    c2 = EffectModel(sources, p2).components(panel)
    t2, s2, h2 = reference_components(sources, p2, panel)
    assert np.allclose(c2["template"], t2, atol=1e-5) and np.allclose(c2["specific"], s2, atol=1e-5)
    assert h2[5] and c2["has_source"][5]                                          # c counts at weight 1


def test_template_sources_restricts_template():
    """template_sources: only the listed sources build T; every source still gives D (minus its OWN template)."""
    genes, panel, sources, _ = small_world(seed=2)
    a, b, c = sources
    weights = {"a": 1.0, "b": 3.0, "c": 0.0}
    base = ModelParams(z0=1.5, z_soft=1.0, null_p=0.05, source_weights=weights)
    full = EffectModel(sources, base).components(panel)
    alone = replace(base, source_weights={})
    Ta = EffectModel([a], alone).components(panel)["template"]
    Tb = EffectModel([b], alone).components(panel)["template"]

    only_a = EffectModel(sources, replace(base, template_sources=["a"])).components(panel)
    # T = a's template alone, also on genes only b measured (there: 0, a has no template value)
    assert np.allclose(only_a["template"], Ta, atol=1e-7)
    assert np.all(only_a["template"][~a.measured] == 0) and np.abs(full["template"][~a.measured]).sum() > 0
    assert not np.allclose(only_a["template"], full["template"])                  # the restriction did something
    # D and has_source do not depend on which sources build T: b's deviations are still from b's own template
    assert np.array_equal(only_a["specific"], full["specific"])
    assert np.array_equal(only_a["has_source"], full["has_source"])
    tmpl, spec, has = reference_components(sources, replace(base, template_sources=["a"]), panel)
    assert np.allclose(only_a["template"], tmpl, atol=1e-5) and np.allclose(only_a["specific"], spec, atol=1e-5)

    # listing every source with weight > 0 is the same as None; a weight-0 source in the list adds nothing
    both = EffectModel(sources, replace(base, template_sources=["a", "b"])).components(panel)
    assert np.array_equal(both["template"], full["template"])
    only_b = EffectModel(sources, replace(base, template_sources=["b", "c"])).components(panel)
    assert np.allclose(only_b["template"], Tb, atol=1e-7)

    # [] → zero template (with a warning at construction); the prediction is then beta · D on sourced rows
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m0 = EffectModel(sources, replace(base, template_sources=[], beta=0.8, gamma=1.4))
    assert any("template is 0 everywhere" in str(r.message) for r in rec)
    c0 = m0.components(panel)
    assert np.all(c0["template"] == 0)
    ctrl = np.full(genes.size, 30.0)
    lfc = m0.predict_lfc(panel, ctrl)
    own = {i: int(np.flatnonzero(genes == t)[0]) for i, t in enumerate(panel) if t in set(genes)}
    exp = 0.8 * c0["specific"]
    for i, j in own.items():
        exp[i, j] = base.kd_lfc
    assert np.allclose(lfc, exp, atol=1e-6)

    # predict_lfc uses the restricted template: gamma · T_a on sourced rows
    ma = EffectModel(sources, replace(base, template_sources=["a"], beta=0.8, gamma=1.4))
    pa = ma.predict_lfc(panel, ctrl)
    exp = 1.4 * Ta[None, :] + 0.8 * only_a["specific"]
    for i, j in own.items():
        exp[i, j] = base.kd_lfc
    assert np.allclose(pa[:5], exp[:5], atol=1e-5)                                # rows 0-4 have a source

    # a misspelled name is reported (it would silently drop that source from the template)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        EffectModel(sources, replace(base, template_sources=["a", "k562"]))
    assert any("template_sources names no loaded source" in str(r.message) for r in rec)


def test_predict_lfc_mask_kd_and_cis():
    genes, panel, sources, ctrl_src = small_world(seed=3)
    G = genes.size
    # cis entries for a sourced target (t0) and an unsourced one (t5); t5's own column too (must be ignored)
    cis = StubCis({panel[0]: {7: -1.0}, panel[5]: {8: -1.5, 9: -0.7, 5: -9.0}, "NOT_ON_AXIS": {12: -1.1}})
    params = ModelParams(z0=1.0, beta=0.8, gamma=1.4, kd_lfc=-2.9, expr_floor_cpm=0.5,
                         source_weights={"c": 0.0})
    m = EffectModel(sources, params, cis=cis, tss=pd.DataFrame())
    ctrl = np.full(G, 30.0)
    low = np.array([9, 12, 20, 21, 30])                # genes the new context does not express (no targets)
    ctrl[low] = 0.1
    lfc = m.predict_lfc(panel, ctrl)
    c = m.components(panel)
    assert lfc.shape == (len(panel), G) and lfc.dtype == np.float32
    own = {i: int(np.flatnonzero(genes == t)[0]) for i, t in enumerate(panel) if t in set(genes)}
    has = c["has_source"]
    mask = ctrl >= params.expr_floor_cpm
    base = mask[None, :] * (1.4 * c["template"][None, :] + 0.8 * c["specific"])
    for i in range(len(panel)):
        exp = base[i].copy()
        if not has[i]:
            exp = exp + c["cis"][i]
        if i in own:
            exp[own[i]] = -2.9
        assert np.allclose(lfc[i], exp, atol=1e-5), f"row {i}"
    assert not has[5] and not has[6]
    assert np.isclose(lfc[5, 8], 1.4 * c["template"][8] - 1.5, atol=1e-5)      # cis on an unsourced target
    assert np.isclose(lfc[5, 9], -0.7, atol=1e-6)                                # cis is not masked; template is
    assert np.isclose(lfc[5, 5], -2.9) and c["cis"][5, 5] == 0                    # own column → kd_lfc
    assert np.isclose(lfc[6, 12], -1.1, atol=1e-6)                                # target off the axis works
    assert np.isclose(lfc[0, 7], base[0, 7], atol=1e-6)                           # no cis where a source exists
    assert np.all(lfc[np.ix_(np.flatnonzero(has), low)] == 0)                     # masked genes, sourced rows
    # component switches
    m.params.use_cis = False
    assert np.isclose(m.predict_lfc(panel, ctrl)[5, 8], 1.4 * c["template"][8], atol=1e-5)
    m.params.use_template = False
    assert np.allclose(m.predict_lfc(panel, ctrl)[5][mask & (np.arange(G) != 5)], 0)
    m.params.use_specific = False
    out = m.predict_lfc(panel, ctrl)
    out[[i for i in own], [own[i] for i in own]] = 0
    assert np.all(out == 0)


def test_beta_gamma_scales_multiply_the_gains():
    """lfc = mask · (gamma·gamma_scale·T + beta·beta_scale·D): a scale acts exactly like a bigger gain."""
    genes, panel, sources, _ = small_world(seed=3)
    G = genes.size
    cis = StubCis({panel[5]: {8: -1.5}})
    w = {"c": 0.0}
    ctrl = np.full(G, 30.0)
    ctrl[[20, 21]] = 0.1                                                          # masked genes
    scaled = EffectModel(sources, ModelParams(z0=1.0, beta=0.8, gamma=1.4, beta_scale=2.5, gamma_scale=0.5,
                                              source_weights=w), cis=cis, tss=pd.DataFrame())
    plain = EffectModel(sources, ModelParams(z0=1.0, beta=2.0, gamma=0.7, source_weights=w),
                        cis=cis, tss=pd.DataFrame())
    assert np.allclose(scaled.predict_lfc(panel, ctrl), plain.predict_lfc(panel, ctrl), atol=1e-6)
    c = scaled.components(panel)
    lfc = scaled.predict_lfc(panel, ctrl)
    exp = (ctrl >= 0.5)[None, :] * (1.4 * 0.5 * c["template"][None, :] + 0.8 * 2.5 * c["specific"])
    assert np.allclose(lfc[1, 10:], exp[1, 10:], atol=1e-5)                       # a sourced row, off own columns
    assert np.isclose(lfc[5, 8], 0.7 * c["template"][8] - 1.5, atol=1e-5)        # cis is not scaled
    assert np.isclose(lfc[0, int(np.flatnonzero(genes == panel[0])[0])], ModelParams().kd_lfc)   # nor is kd_lfc
    # a scale of 0 switches its term off, like use_* = False
    scaled.params.beta_scale = 0.0
    lfc_off = scaled.predict_lfc(panel, ctrl)
    no_spec = replace(scaled.params, beta_scale=1.0, use_specific=False)
    assert np.allclose(lfc_off, EffectModel(sources, no_spec, cis=cis, tss=pd.DataFrame()).predict_lfc(panel, ctrl))
    assert np.abs(lfc_off).sum() > 0


def test_min_called_lfc_floors_only_called_genes_above_5_cpm():
    """Where D != 0 and the new context's control CPM > 5: lfc = sign(lfc) · max(|lfc|, min_called_lfc)."""
    genes, panel, sources, _ = small_world(seed=3)
    G = genes.size
    cis = StubCis({panel[5]: {8: -0.1, 9: 0.05}})                                 # small cis values: not floored
    ctrl = np.full(G, 100.0)
    ctrl[6:14] = 3.0                                                              # expressed, not DE-tested
    ctrl[14:20] = 5.0                                                             # exactly the gate: not tested
    ctrl[20:26] = 5.01                                                            # just above the gate: tested
    ctrl[[30, 31]] = 0.1                                                          # below the expression floor
    params = ModelParams(z0=1.0, beta=0.3, gamma=0.5, min_called_lfc=0.4, source_weights={"c": 0.0})
    m = EffectModel(sources, params, cis=cis, tss=pd.DataFrame())
    c = m.components(panel)
    D = c["specific"]
    own = {i: int(np.flatnonzero(genes == t)[0]) for i, t in enumerate(panel) if t in set(genes)}

    lfc = m.predict_lfc(panel, ctrl)
    m.params.min_called_lfc = 0.0
    base = m.predict_lfc(panel, ctrl)                                             # the same model without the floor
    m.params.min_called_lfc = 0.4

    called = (D != 0) & (ctrl > 5.0)[None, :]
    for i, j in own.items():
        called[i, j] = False                                                      # own column: always kd_lfc
    exp = np.where(called, np.sign(base) * np.maximum(np.abs(base), 0.4), base)
    assert np.allclose(lfc, exp, atol=1e-6)

    # The checks below must not be vacuous: each kind of entry exists in this world.
    small = (np.abs(base) > 0) & (np.abs(base) < 0.4)
    lifted = called & small
    assert lifted.sum() >= 5, lifted.sum()
    assert np.allclose(np.abs(lfc[lifted]), 0.4) and np.all(np.sign(lfc[lifted]) == np.sign(base[lifted]))
    big = called & (np.abs(base) >= 0.4)
    assert big.sum() >= 1 and np.array_equal(lfc[big], base[big])                # already large: unchanged
    low_ctrl = (D != 0) & (ctrl <= 5.0)[None, :] & (ctrl >= 0.5)[None, :] & small
    assert low_ctrl.sum() >= 3 and np.array_equal(lfc[low_ctrl], base[low_ctrl])  # ≤ 5 CPM (incl. 5.0): untouched
    assert ((D != 0) & small)[:, 14:20].sum() >= 1                                # the exact-5.0 genes are covered
    lifted_edge = lifted[:, 20:26]
    assert lifted_edge.sum() >= 1                                                 # 5.01 CPM is floored
    tmpl_only = (D == 0) & (ctrl > 5.0)[None, :] & small
    for i, j in own.items():
        tmpl_only[i, j] = False
    assert tmpl_only.sum() >= 5 and np.array_equal(lfc[tmpl_only], base[tmpl_only])   # D == 0: never floored
    assert np.all(lfc[:, [30, 31]][:5] == 0)                                       # masked genes stay 0
    assert np.isclose(lfc[5, 8], base[5, 8]) and np.isclose(lfc[5, 9], base[5, 9])  # unsourced row: cis untouched
    for i, j in own.items():
        assert lfc[i, j] == np.float32(params.kd_lfc)

    # with the specific term switched off the model calls nothing, so nothing is floored
    m.params.use_specific = False
    off = m.predict_lfc(panel, ctrl)
    m.params.min_called_lfc = 0.0
    assert np.array_equal(off, m.predict_lfc(panel, ctrl))


def test_predict_cpm_shape_and_nonneg():
    genes, panel, sources, _ = small_world(seed=4)
    rng = np.random.default_rng(4)
    ctrl = rng.lognormal(1, 2, genes.size)
    ctrl[:5] = 0.0
    m = EffectModel(sources, ModelParams(z0=0.5, beta=3.0, gamma=3.0))   # large gains: strong negative lfc
    cpm = m.predict_cpm(panel, ctrl)
    lfc = m.predict_lfc(panel, ctrl)
    assert cpm.shape == (len(panel), genes.size)
    assert np.all(np.isfinite(cpm)) and np.all(cpm >= 0)
    assert np.allclose(cpm, apply_log2fc(ctrl, lfc, EPS_CPM))
    keep = (lfc == 0)
    assert np.allclose(cpm[keep], np.broadcast_to(ctrl, cpm.shape)[keep])         # lfc 0 → control CPM
    # where nothing was clipped, the scorer-style log2FC (same eps) recovers the prediction
    ok = cpm > 1e-6
    back = np.log2((cpm + EPS_CPM) / (ctrl[None, :] + EPS_CPM))
    assert np.allclose(back[ok], lfc[ok], atol=1e-4)


def _fit_world(seed=5, G=1200, n_src=50, n_extra=10, mid_cpm=None):
    """A source line and a truth line: truth = 0.7·specific + 1.3·template (+ cis where unsourced) + noise.

    mid_cpm: if given, 150 more genes get truth control CPM = mid_cpm (above the expression floor) and reliable
    (se 0.2), template-like garbage responses. fit_gains must ignore them when mid_cpm <= 5 (not DE-testable).
    Returns (model, truth, panel, comps) or, with mid_cpm, (model, truth, panel, comps, mid gene indices).
    """
    rng = np.random.default_rng(seed)
    genes = vcc_genes(CONTROLS, n=G)
    perm = rng.permutation(G)
    src_targets = [str(genes[i]) for i in perm[:n_src]]
    extra = [str(genes[i]) for i in perm[n_src:n_src + n_extra]]          # panel targets no source measured
    panel = src_targets + extra

    tmpl_src = np.zeros(G)
    tg = rng.choice(G, 200, replace=False)
    tmpl_src[tg] = rng.normal(0, 0.3, 200)
    L = tmpl_src[None, :] + rng.normal(0, 0.03, (n_src, G))
    for i in range(n_src):
        idx = rng.choice(G, 25, replace=False)
        L[i, idx] += rng.normal(0, 1.0, 25)
        L[i, perm[i]] = -3.0                                                 # the knockdown itself
    src_measured = np.ones(G, bool)
    src_measured[-50:] = False
    pp = np.full(n_src, 1e-5)
    pp[:5] = 0.5                                                             # 5 targets with no phenotype
    src = make_rs("src", genes, src_targets, L, 20 + rng.lognormal(3, 1, G), 0.1, src_measured, pp)

    # cis: entries for sourced targets (must be ignored) and unsourced ones (must be added, then removed in fit)
    entries = {t: {int(g): -1.0 for g in rng.choice(G, 3, replace=False)} for t in src_targets[:10]}
    model0 = EffectModel([src], ModelParams())
    T0 = model0.components(panel)["template"]
    top = np.argsort(-np.abs(T0))[:20]                                       # cis anti-correlated with template
    for t in extra:
        entries[t] = {int(g): float(-2.0 * np.sign(T0[g])) for g in top}
    cis = StubCis(entries)
    model = EffectModel([src], ModelParams(), cis=cis, tss=pd.DataFrame())
    comps = model.components(panel)

    beta, gamma = 0.7, 1.3
    Y = gamma * comps["template"][None, :] + beta * comps["specific"] + rng.normal(0, 0.05, (len(panel), G))
    Y += comps["cis"] * (~comps["has_source"])[:, None]
    ctrl_t = 20 + rng.lognormal(3, 1, G)
    own = np.array([int(np.flatnonzero(genes == t)[0]) for t in panel])
    low = rng.choice(np.setdiff1d(np.arange(G), own), 100, replace=False)
    ctrl_t[low] = 0.2                                                        # below the expression floor
    Y[:, low] = 2.5                                                          # garbage there: must be excluded
    Y[np.arange(len(panel)), own] = -4.0                                     # own column: must be excluded
    se = np.where(rng.random((len(panel), G)) < 0.5, 0.2, 1e-3)              # w = 25, or 1e6 → capped at 100
    junk = rng.random((len(panel), G)) < 0.05
    junk[:, low] = False
    Y[junk] = np.broadcast_to(5.0 * comps["template"][None, :] + 3.0, Y.shape)[junk]   # wrong, template-like
    se[junk] = 50.0                                                         # unreliable pairs, w = 4e-4
    t_measured = np.ones(G, bool)
    t_measured[rng.choice(G, 30, replace=False)] = False
    mid = None
    if mid_cpm is not None:                     # extra draws only here: the default world is unchanged
        free = np.setdiff1d(np.arange(G), np.concatenate([own, low]))
        free = free[t_measured[free]]
        # prefer genes where the template is non-zero, so the garbage would really pull gamma if it were fitted
        tg_free = free[np.abs(comps["template"][free]) > 0.05]
        mid = rng.choice(tg_free, min(150, tg_free.size), replace=False)
        ctrl_t[mid] = mid_cpm
        Y[:, mid] = np.clip(4.0 * comps["template"][mid] + 1.0, -1.0, None)[None, :]   # > log2(0.5/1.5)
        se[:, mid] = 0.2
    truth = make_rs("truth", genes, panel, Y, ctrl_t, se, t_measured)
    if mid_cpm is not None:
        return model, truth, panel, comps, mid
    return model, truth, panel, comps


def test_fit_gains_recovers_beta_gamma():
    model, truth, panel, comps = _fit_world()
    before = asdict(model.params)
    beta, gamma = fit_gains(model, truth, targets=panel)
    # 0.008: seeds 5-14 recover both gains within 0.0037; dropping the own-column exclusion moves gamma by 0.012,
    # dropping the cis offset by 0.29, dropping both gene masks (expression floor and > 5 CPM) by 0.30. (Either
    # mask alone removes this world's 0.2-CPM garbage genes; the > 5 CPM rule on its own is tested below.)
    assert abs(beta - 0.7) < 0.008 and abs(gamma - 1.3) < 0.008, (beta, gamma)
    assert asdict(model.params) == before                                    # the model is not modified
    b2, g2 = fit_gains(model, truth)                                          # default panel = truth's targets
    assert np.isclose(b2, beta) and np.isclose(g2, gamma)
    # Without the 1/se² weights (all weights equal) the unreliable pairs pull the template gain away.
    bu, gu = fit_gains(model, truth, targets=panel, max_weight=1e-9)
    assert abs(gu - 1.3) > 0.1, (bu, gu)
    # Specific-only and template-only fits (the other gain stays at its current value).
    model.params.use_template = False
    b3, g3 = fit_gains(model, truth, targets=panel)
    assert g3 == model.params.gamma and np.isfinite(b3)
    model.params.use_template, model.params.use_specific = True, False
    b4, g4 = fit_gains(model, truth, targets=panel)
    assert b4 == model.params.beta and np.isfinite(g4)


def test_fit_gains_degenerate_systems():
    model, truth, panel, comps = _fit_world(seed=6)
    # No specific signal (gate never opens): beta keeps its current value, gamma is still recovered.
    model.params.z0 = 1e9
    model.params.beta = 0.123
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        beta, gamma = fit_gains(model, truth, targets=panel)
    assert beta == 0.123 and np.isfinite(gamma) and any("zero on all fitted pairs" in str(r.message) for r in rec)
    # Nothing to fit at all: an all-zero source → both gains returned unchanged, with warnings.
    genes = np.asarray(truth.genes)
    zero = make_rs("zero", genes, panel[:5], np.zeros((5, genes.size)), np.full(genes.size, 50.0), 0.1)
    m0 = EffectModel([zero], ModelParams(beta=0.4, gamma=0.6))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert fit_gains(m0, truth, targets=panel) == (0.4, 0.6)
    assert len(rec) >= 2
    # Collinear columns: source rows 2u and 0 → template u, and the fitted row's specific response is u too.
    u = (np.arange(genes.size) % 7 == 0).astype(np.float64)
    pair = make_rs("pair", genes, ["S1", "S2"], np.vstack([2 * u, 0 * u]), np.full(genes.size, 50.0), 0.01)
    m1 = EffectModel([pair], ModelParams(z0=0.0, z_soft=1e-6, beta=0.9))
    c1 = m1.components(["S1", "S2"])
    assert np.allclose(c1["template"], u, atol=1e-4) and np.allclose(c1["specific"][0], u, atol=1e-4)
    tr = make_rs("tr", genes, ["S1"], 2.0 * u[None, :], np.full(genes.size, 50.0), 0.1)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        b, g = fit_gains(m1, tr, targets=["S1", "S2"])
    assert b == 0.9 and abs(g - 2.0) < 1e-3 and any("collinear" in str(r.message) for r in rec), (b, g)


def test_scales_and_floor_survive_fit_gains():
    """fit_gains estimates the RAW gains (as if beta_scale = gamma_scale = 1, min_called_lfc = 0) and leaves the
    scales alone, so 02's `replace(params, beta=b, gamma=g)` keeps the scorer-tuned stage-2 settings."""
    model, truth, panel, comps = _fit_world()
    raw = fit_gains(model, truth, targets=panel)
    tuned = replace(model.params, beta_scale=3.0, gamma_scale=0.25, min_called_lfc=0.4)
    model.params = tuned
    before = asdict(model.params)
    got = fit_gains(model, truth, targets=panel)
    assert got == raw, (got, raw)                                                # bit-identical raw gains
    assert abs(got[0] - 0.7) < 0.008 and abs(got[1] - 1.3) < 0.008, got        # = the true raw gains
    assert asdict(model.params) == before                                        # model untouched
    refit = replace(tuned, beta=got[0], gamma=got[1])                            # what 02_validate.py does
    assert (refit.beta_scale, refit.gamma_scale, refit.min_called_lfc) == (3.0, 0.25, 0.4)
    # The degenerate fallback returns the current RAW gain, not the scaled one.
    model.params = replace(tuned, z0=1e9, beta=0.123)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        b, _ = fit_gains(model, truth, targets=panel)
    assert b == 0.123


def test_fit_gains_ignores_genes_at_or_below_5_cpm():
    """Only DE-testable genes of the truth line (control CPM > 5) enter the fit; 5.0 itself is excluded."""
    for cpm in (1.0, 3.0, 5.0):                         # above the 0.5 expression floor, but never DE-tested
        model, truth, panel, comps, mid = _fit_world(mid_cpm=cpm)
        assert mid.size >= 100 and np.all(np.asarray(truth.ctrl_cpm)[mid] == cpm)
        beta, gamma = fit_gains(model, truth, targets=panel)
        assert abs(beta - 0.7) < 0.008 and abs(gamma - 1.3) < 0.008, (cpm, beta, gamma)
    # Control: the same garbage on genes just above the gate is fitted and pulls the gains (non-vacuous test).
    model, truth, panel, comps, mid = _fit_world(mid_cpm=5.5)
    beta, gamma = fit_gains(model, truth, targets=panel)
    assert abs(gamma - 1.3) > 0.1, (beta, gamma)


def test_with_real_cis_model_and_fake_gtf():
    """Integration with vccpipe.cis: a fitted-looking CisModel and a TSS table from a GENCODE-like fake GTF."""
    try:
        from vccpipe.cis import CisModel, load_tss
    except ImportError:
        print("  SKIP: vccpipe.cis not importable yet")
        return
    rng = np.random.default_rng(9)
    genes = vcc_genes(CONTROLS, n=300)
    with tempfile.TemporaryDirectory() as tmp:
        gtf = Path(tmp) / "fake.gtf.gz"
        make_fake_gtf(gtf, genes, rng)                 # genes 2k and 2k+1 (k < 20) share a bidirectional promoter
        tss = load_tss(gtf)
    cis = CisModel(effect=[-1.5, -0.5, 0.0, 0.0, 0.0, 0.0])
    sourced = [str(genes[0]), str(genes[2])]
    panel = sourced + [str(genes[4]), str(genes[6])]   # genes[4], genes[6]: no source → cis prior used
    L = rng.normal(0, 0.5, (2, genes.size))
    src = make_rs("src", genes, sourced, L, np.full(genes.size, 40.0), 0.1)
    model = EffectModel([src], ModelParams(gamma=1.2, beta=0.9), cis=cis, tss=tss)
    ctrl = np.full(genes.size, 25.0)
    lfc = model.predict_lfc(panel, ctrl)
    c = model.components(panel)
    prior = cis.predict(panel, genes, tss)
    assert prior[2, 5] == -1.5 and prior[3, 7] == -1.5 and prior[0, 1] == -1.5     # bidirectional partners
    base = 1.2 * c["template"][None, :] + 0.9 * c["specific"]
    exp = base + prior * (~c["has_source"])[:, None]
    exp[np.arange(4), [0, 2, 4, 6]] = ModelParams().kd_lfc
    assert np.allclose(lfc, exp, atol=1e-5)
    assert np.isclose(lfc[2, 5], 1.2 * c["template"][5] - 1.5, atol=1e-5)       # unsourced: prior added
    assert np.isclose(lfc[0, 1], base[0, 1], atol=1e-6)                           # sourced: prior not added


def _with(model, **kw):
    """A model sharing `model`'s sources with params attributes set AFTER construction (bypassing __post_init__)."""
    p = replace(model.params)
    for k, v in kw.items():
        setattr(p, k, v)
    out = EffectModel(model.sources, ModelParams(), cis=model.cis, tss=model.tss)
    out.params = p
    return out


def test_input_validation():
    genes, panel, sources, ctrl = small_world(seed=7)
    m = EffectModel(sources, ModelParams())
    other = make_rs("other", np.array([f"Z{i}" for i in range(genes.size)]), ["Q"], np.zeros((1, genes.size)),
                    ctrl, 0.1)
    checks = [
        (lambda: EffectModel([], ModelParams()), ValueError),
        (lambda: EffectModel([sources[0], other], ModelParams()), ValueError),     # different gene axes
        (lambda: EffectModel([sources[0], sources[0]], ModelParams()), ValueError),  # duplicate names
        (lambda: m.components(["A", "A"]), ValueError),
        (lambda: m.components("ABC"), TypeError),
        (lambda: m.predict_lfc(panel, ctrl[:-1]), ValueError),
        (lambda: m.predict_lfc(panel, -ctrl), ValueError),
        (lambda: fit_gains(m, other), ValueError),
        (lambda: fit_gains(m, sources[0], targets=["NOPE"]), ValueError),
        (lambda: EffectModel(sources, ModelParams(source_weights={"a": -1})).components(panel), ValueError),
        (lambda: EffectModel(sources, ModelParams(beta_scale=np.nan)).components(panel), ValueError),
        (lambda: EffectModel(sources, ModelParams(gamma_scale=np.inf)).predict_lfc(panel, ctrl), ValueError),
        (lambda: EffectModel(sources, ModelParams(min_called_lfc=-0.1)).predict_lfc(panel, ctrl), ValueError),
        (lambda: ModelParams(template_sources="a"), TypeError),                    # a bare string, not a list
        (lambda: _with(m, template_sources="a").components(panel), TypeError),     # assigned after construction
    ]
    for fn, exc in checks:
        try:
            fn()
        except exc:
            continue
        raise AssertionError(f"expected {exc.__name__}")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        EffectModel(sources, ModelParams(source_weights={"k562": 1.0}))             # misspelled source name
    assert rec


def test_real_scale_speed_memory_and_subset_only():
    """VCC axis (18,533 genes) and the real 300-target panel; a 1,500-target source must only be read by row."""
    if not (CONTROLS / "gene_names.csv").exists() or not (CONTROLS / "pert_counts.csv").exists():
        print("  SKIP: controls bundle not found")
        return
    rng = np.random.default_rng(8)
    genes = vcc_genes(CONTROLS)
    G = genes.size
    panel = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].astype(str).tolist()
    on_axis, in_panel = set(genes), set(panel)
    others = [g for g in genes[rng.permutation(G)] if g not in in_panel][:1228]
    big_targets = panel[:272] + others                                         # like K562-GW: 272/300 covered
    ctrl = (20 + rng.lognormal(2, 1.5, G)).astype(np.float64)

    def big_rs(name, targets, sd):
        T = len(targets)
        L = rng.standard_normal((T, G), dtype=np.float32) * np.float32(sd)
        tc = (ctrl.astype(np.float32)[None, :] + np.float32(EPS_CPM)) * np.exp2(L) - np.float32(EPS_CPM)
        del L
        np.clip(tc, 0, None, out=tc)
        meas = np.ones(G, bool)
        meas[rng.choice(G, G // 3, replace=False)] = False                     # bulk screens miss many genes
        tc[:, ~meas] = 0
        se = np.full((T, G), 0.1, dtype=np.float32)
        se[:, ~meas] = np.inf
        meta = pd.DataFrame({"n_cells": np.full(T, 150, dtype=np.int64), "kd_remaining": np.full(T, 0.2),
                             "phenotype_p": rng.uniform(0, 0.1, T)}, index=pd.Index(targets))
        return ResponseSet(name=name, genes=genes, measured=meas, targets=list(targets), target_cpm=tc,
                           ctrl_cpm=np.where(meas, ctrl, 0.0), se_lfc=se, meta=meta)

    big = big_rs("k562_like", big_targets, 0.3)
    h1 = big_rs("h1_like", panel[250:300], 0.3)                                # 22 overlap with big
    sources = [SpySource(big, 300), SpySource(h1, 300)]
    cis = StubCis({t: {0: -1.0} for t in panel})
    model = EffectModel(sources, ModelParams(source_weights={"k562_like": 1.0, "h1_like": 2.0}),
                        cis=cis, tss=pd.DataFrame())
    ctx_ctrl = rng.lognormal(1, 2, G)

    tracemalloc.start()
    t0 = time.time()
    lfc = model.predict_lfc(panel, ctx_ctrl)
    cpm = model.predict_cpm(panel, ctx_ctrl)
    dt = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  ResponseSet: {RS_KIND}")
    print(f"  predict_lfc + predict_cpm, 300 targets × {G} genes: {dt:.1f} s, peak traced memory "
          f"{peak / 2**20:.0f} MB (one 300×G float32 = {300 * G * 4 / 2**20:.0f} MB)")
    assert lfc.shape == cpm.shape == (300, G) and np.all(cpm >= 0) and np.all(np.isfinite(cpm))
    assert sum(t in on_axis for t in panel) > 250
    assert dt < 60 and peak < 700 * 2**20, (dt, peak)

    truth = big_rs("truth_like", panel[:120], 0.3)
    t0 = time.time()
    tracemalloc.start()
    beta, gamma = fit_gains(model, truth, targets=panel)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  fit_gains, 120 truth targets: {time.time() - t0:.1f} s, peak {peak / 2**20:.0f} MB "
          f"(beta={beta:.3f}, gamma={gamma:.3f}; noise-only data, values not meaningful)")
    assert np.isfinite(beta) and np.isfinite(gamma) and peak < 700 * 2**20


# ----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    print(f"vccpipe.model tests ({RS_KIND})")
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f} s)")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
