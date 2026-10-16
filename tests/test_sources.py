"""Tests for vccpipe.sources (ResponseSet, load_replogle_bulk, h1_pseudobulk).

All data are small fakes with the real files' layout (tests/fakes.py), written to a temporary folder; the gene axis
is the first 3,000 symbols of the real VCC axis (controls/gene_names.csv). Nothing under data/ is read.

Run with pytest, or directly:  .venv/bin/python tests/test_sources.py
"""

import atexit
import contextlib
import functools
import io
import re
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fakes  # noqa: E402
import vccpipe.sources as S  # noqa: E402
from vccpipe.common import CONTROL, DE_GATE_CPM  # noqa: E402
from vccpipe.sources import EPS_CPM, ResponseSet, h1_pseudobulk, load_replogle_bulk  # noqa: E402

N_GENES = 3000           # gene axis used by the tests
N_OFF = 5                # the last N_OFF axis genes are absent from the fake files (-> measured = False)
N_TARGETS = 20
META = ["n_cells", "kd_remaining", "phenotype_p"]

warnings.filterwarnings("ignore", message=".*not unique.*")          # duplicated symbols are deliberate here


@functools.lru_cache(maxsize=None)
def _tmpdir():
    d = Path(tempfile.mkdtemp(prefix="test_sources_"))
    atexit.register(shutil.rmtree, d, True)
    return d


@functools.lru_cache(maxsize=None)
def _axis():
    return fakes.vcc_genes(ROOT / "controls", n=N_GENES)


def _own_gene_mask(targets, cols):
    """(T, len(cols)) True at each target's own gene (excluded from response checks: it is the knockdown)."""
    where = {g: j for j, g in enumerate(cols)}
    m = np.zeros((len(targets), len(cols)), dtype=bool)
    for i, t in enumerate(targets):
        if t in where:
            m[i, where[t]] = True
    return m


def _effective_lfc(base_cpm, planted):
    """The fold change the fakes really produce: they renormalize each profile to 1e6, which shifts every gene a
    little when some genes change. log2 with the model's pseudocount, as the estimator uses."""
    mu = base_cpm[None, :] * np.exp2(planted)
    mu = mu / mu.sum(axis=1, keepdims=True) * 1e6
    return np.log2((mu + EPS_CPM) / (base_cpm[None, :] + EPS_CPM))


# ---------------------------------------------------------------------------------------------------------------
# Reference implementations: deliberately naive (loops, whole files in memory), written from the docstrings
# ---------------------------------------------------------------------------------------------------------------

# Reference label parser, written independently of vccpipe.sources (a different method: one regex that lists the
# transcript forms seen in the real files instead of a generic "RefSeq or one token" rule). The gene group is lazy,
# so "ZNF735_NM_001159524" splits as ZNF735 + NM_001159524, not ZNF735_NM + 001159524.
_REF_LABEL = re.compile(r"\d+_(.+?)_(?:P1P2|P1|P2|non-targeting|ENST\d+(?:\.\d+)?(?:,ENST\d+(?:\.\d+)?)*"
                        r"|[NX][MR]_\d+(?:\.\d+)?(?:,[NX][MR]_\d+(?:\.\d+)?)*)_(?:ENSG\d+|non-targeting|nan)")


def _ref_label(s):
    m = _REF_LABEL.fullmatch(s)
    assert m, f"reference parser does not know the row name {s!r}"
    return m.group(1)


def _direct_replogle(path, axis):
    a = ad.read_h5ad(path)
    X = np.asarray(a.X.toarray() if sp.issparse(a.X) else a.X, dtype=np.float64)
    ai = pd.Index(axis)
    names = a.var["gene_name"].astype(str).to_numpy()
    lab = [_ref_label(s) for s in a.obs_names]
    n = pd.to_numeric(a.obs["num_cells_filtered"], errors="coerce").to_numpy(float)
    fold = a.obs["fold_expr"].to_numpy(float)
    pv = a.obs["energy_test_p_value"].to_numpy(float)
    lib = X.sum(axis=1)

    def on_axis(row):
        out = np.zeros(axis.size)
        for j, g in enumerate(names):
            if g in ai:
                out[ai.get_loc(g)] += row[j] / row.sum() * 1e6
        return out

    ok = np.isfinite(n) & (n > 0)
    crow = [j for j in range(len(lab)) if lab[j] == CONTROL and ok[j]]
    C = np.array([on_axis(X[j]) for j in crow])
    nc = n[crow]
    N = nc.sum()
    c = (nc[:, None] * C).sum(0) / N
    sigma2 = np.mean([nc[k] * np.log2((C[k] + EPS_CPM) / (c + EPS_CPM)) ** 2 / (1 - nc[k] / N)
                      for k in range(len(crow))], axis=0)
    # Poisson floor: Var(cpm) of a count of (observed + 1) molecules out of mol, through the log (delta method)
    pois = lambda cpm, mol: (cpm + 1e6 / mol) * 1e6 / mol / ((cpm + EPS_CPM) ** 2 * np.log(2) ** 2)  # noqa: E731
    var_c = np.maximum(sigma2 / N, pois(c, (nc * lib[crow]).sum()))
    res = {}
    for t in dict.fromkeys(lab[j] for j in range(len(lab)) if lab[j] != CONTROL and ok[j]):
        rows = [j for j in range(len(lab)) if lab[j] == t and ok[j]]
        w = n[rows]
        x = (w[:, None] * np.array([on_axis(X[j]) for j in rows])).sum(0) / w.sum()
        var_t = np.maximum(sigma2 / w.sum(), pois(x, (w * lib[rows]).sum()))
        f = np.isfinite(fold[rows])
        kd = (w[f] * fold[rows][f]).sum() / w[f].sum() if f.any() else np.nan
        p = np.nanmin(pv[rows]) if np.isfinite(pv[rows]).any() else np.nan
        res[t] = dict(x=x, se=np.sqrt(var_t + var_c), n=w.sum(), kd=kd, p=p)
    return c, res


def _direct_h1(paths, axis, ctrl_files=(0,)):
    """Whole files in memory, dense per-cell CPM (total over all the file's genes), numpy mean / var(ddof=1)."""
    ai = pd.Index(axis)
    ctrl, tgt = [], {}
    for k, p in enumerate(paths):
        a = ad.read_h5ad(p)
        X = np.asarray(a.X.toarray() if sp.issparse(a.X) else a.X, dtype=np.float64)
        cpm = X / X.sum(axis=1, keepdims=True) * 1e6
        A = np.zeros((X.shape[0], axis.size))
        for j, g in enumerate(a.var_names.astype(str)):
            if g in ai:
                A[:, ai.get_loc(g)] += cpm[:, j]
        tg = a.obs["target_gene"].astype(str).to_numpy()
        if k in ctrl_files:
            ctrl.append(A[tg == CONTROL])
        for t in pd.unique(tg[tg != CONTROL]):
            tgt.setdefault(t, []).append(A[tg == t])
    C = np.vstack(ctrl)
    c, vc, nc = C.mean(0), C.var(0, ddof=1), C.shape[0]
    res = {}
    for t, blocks in tgt.items():
        B = np.vstack(blocks)
        x, v, n = B.mean(0), B.var(0, ddof=1), B.shape[0]
        se = np.sqrt(v / (n * (x + EPS_CPM) ** 2) + vc / (nc * (c + EPS_CPM) ** 2)) / np.log(2)
        res[t] = dict(x=x, se=se, n=n)
    return c, res


def _check_against_direct(rs, c_ref, ref, targets=None, ctrl_rtol=1e-10):
    """ctrl_rtol: the Replogle loader scales the 0.35 GB count matrix to CPM in float32 (memory), so its control
    means agree to ~1e-7 only; h1_pseudobulk accumulates in float64 and agrees to rounding."""
    targets = rs.targets if targets is None else targets
    assert rs.targets == list(targets)
    m = rs.measured
    np.testing.assert_allclose(rs.ctrl_cpm[m], c_ref[m], rtol=ctrl_rtol, atol=1e-9)
    assert np.all(rs.ctrl_cpm[~m] == 0)
    for i, t in enumerate(targets):
        r = ref[t]
        np.testing.assert_allclose(rs.target_cpm[i, m], r["x"][m], rtol=2e-6, atol=1e-5, err_msg=t)
        np.testing.assert_allclose(rs.se_lfc[i, m], r["se"][m], rtol=2e-5, atol=1e-7, err_msg=t)
        assert np.all(rs.target_cpm[i, ~m] == 0) and np.all(np.isinf(rs.se_lfc[i, ~m]))
        assert rs.meta.loc[t, "n_cells"] == r["n"]


# ---------------------------------------------------------------------------------------------------------------
# Fake datasets (built once per run)
# ---------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _replogle_case():
    """Fake K562 bulk file. File genes = the axis minus its last N_OFF genes, plus N_OFF symbols that are not on the
    axis; the fake renames the last of those to the first symbol (a duplicated symbol, as in the real file)."""
    rng = np.random.default_rng(1)
    axis = _axis()
    file_genes = np.concatenate([axis[:N_GENES - N_OFF], [f"NOTONAXIS{i}" for i in range(N_OFF)]])
    targets = [str(t) for t in rng.choice(axis[1:N_GENES - N_OFF], N_TARGETS - 2, replace=False)]
    targets += ["OFFTARGET1", "OFFTARGET2"]                       # targets whose gene is not on the axis
    path = _tmpdir() / "K562_fake_raw_bulk_01.h5ad"
    truth = fakes.make_fake_replogle_bulk(path, file_genes, targets, rng)
    rs = load_replogle_bulk(path, axis, "k562_fake")
    return dict(path=path, axis=axis, targets=targets, truth=truth, rs=rs)


@functools.lru_cache(maxsize=None)
def _h1_case():
    """Three fake H1 files (targets split across them, the same 600 control cells in each). File genes = the axis
    minus its last N_OFF genes, plus 3 symbols that are not on the axis (as in the real H1 files)."""
    rng = np.random.default_rng(2)
    axis = _axis()
    file_genes = np.concatenate([axis[:N_GENES - N_OFF], ["OFFAXIS_A", "OFFAXIS_B", "OFFAXIS_C"]])
    targets = [str(t) for t in rng.choice(axis[:N_GENES - N_OFF], N_TARGETS - 1, replace=False)]
    targets += ["OFFTARGET_H1"]
    planted, null = fakes.planted_effects(targets, file_genes, rng)
    paths = [_tmpdir() / f"h1_fake_{k}.h5ad" for k in range(3)]
    truth = fakes.make_fake_h1(paths, file_genes, targets, rng, lfc=planted)
    rs = h1_pseudobulk(paths, axis, chunk_cells=250, verbose=False)     # small chunks: exercise the chunking
    return dict(paths=paths, axis=axis, targets=targets, planted=planted, null=null,
                base_cpm=truth["base_cpm"], rs=rs)


def _response_checks(rs, targets, planted, base_cpm, cols):
    """(b) + (c): recovered lfc vs the planted one, and z calibration. cols = axis columns whose file column is
    the same index (clean 1:1 mapping)."""
    L, Z, SE = rs.lfc()[:, cols], rs.z()[:, cols], rs.se_lfc[:, cols]
    P = planted[:, cols]
    own = _own_gene_mask(targets, rs.genes[cols])
    base = np.broadcast_to(base_cpm[cols], P.shape)
    affected = (P != 0) & ~own
    assert affected.sum() > 300

    # (b) the recovered fold changes track the planted ones (all affected genes; tighter for testable genes)
    r_all = np.corrcoef(L[affected], P[affected])[0, 1]
    gated = affected & (base > DE_GATE_CPM)
    r_gated = np.corrcoef(L[gated], P[gated])[0, 1]
    assert r_all > 0.9, r_all
    assert r_gated > 0.95, r_gated

    # (c) strong changes have large z ...
    strong = affected & (np.abs(P) >= 1) & (base > DE_GATE_CPM)
    assert np.median(np.abs(Z[strong])) > 5, np.median(np.abs(Z[strong]))
    assert np.mean(np.abs(Z[strong]) > 3) > 0.8
    # ... and unaffected genes look like N(0, 1) noise. Raw z: rarely beyond 4. Calibration (mean 0, sd 1) is
    # checked around the effective truth, because the fakes renormalize each profile to 1e6 after planting the
    # effects, which shifts every gene a little (deep H1 data resolves that shift: raw z has mean ~ -0.1).
    null = (P == 0) & ~own
    assert np.mean(np.abs(Z[null]) > 4) < 0.01, np.mean(np.abs(Z[null]) > 4)
    z_res = ((L - _effective_lfc(base_cpm, planted)[:, cols]) / SE)[null]
    assert np.mean(np.abs(z_res) > 4) < 0.01, np.mean(np.abs(z_res) > 4)
    assert 0.8 < np.std(z_res) < 1.2, np.std(z_res)
    assert abs(np.mean(z_res)) < 0.1, np.mean(z_res)


# ---------------------------------------------------------------------------------------------------------------
# load_replogle_bulk
# ---------------------------------------------------------------------------------------------------------------

def test_replogle_shapes_and_meta():
    c = _replogle_case()
    rs, axis = c["rs"], c["axis"]
    T, G = N_TARGETS, N_GENES
    assert rs.name == "k562_fake"
    assert rs.genes.tolist() == axis.tolist()
    assert rs.targets == c["targets"]                                  # order of first appearance in the file
    assert rs.target_cpm.shape == (T, G) and rs.target_cpm.dtype == np.float32
    assert rs.se_lfc.shape == (T, G) and rs.se_lfc.dtype == np.float32
    assert rs.ctrl_cpm.shape == (G,) and rs.ctrl_cpm.dtype == np.float64
    assert rs.measured.tolist() == [True] * (G - N_OFF) + [False] * N_OFF
    assert np.all(np.isinf(rs.se_lfc[:, ~rs.measured])) and np.all(np.isfinite(rs.se_lfc[:, rs.measured]))
    assert np.all(rs.target_cpm[:, ~rs.measured] == 0) and np.all(rs.ctrl_cpm[~rs.measured] == 0)
    assert list(rs.meta.columns[:3]) == META and rs.meta.index.tolist() == rs.targets
    assert rs.meta["n_cells"].dtype == np.int64 and (rs.meta["n_cells"] >= 60).all()
    np.testing.assert_allclose(rs.meta["kd_remaining"], 0.15)          # the fake writes fold_expr = 0.15
    null = c["truth"]["null"]
    assert np.all(rs.meta["phenotype_p"].to_numpy()[~null] == 1e-6)
    assert np.all(rs.meta["phenotype_p"].to_numpy()[null] >= 0.2)
    # CPM rows sum to ~1e6 over the axis (minus the off-axis genes' small share)
    assert np.all(rs.target_cpm.sum(1) < 1e6 + 1) and np.all(rs.target_cpm.sum(1) > 9e5)


def test_replogle_matches_direct_computation():
    c = _replogle_case()
    c_ref, ref = _direct_replogle(c["path"], c["axis"])
    _check_against_direct(c["rs"], c_ref, ref, c["targets"], ctrl_rtol=1e-6)
    # the duplicated symbol (axis gene 0 = file columns 0 and last) is summed
    assert c["rs"].ctrl_cpm[0] > 0


def test_replogle_recovers_planted_effects():
    c = _replogle_case()
    cols = np.arange(1, N_GENES - N_OFF)          # axis gene 0 also holds the duplicated column: skip it
    _response_checks(c["rs"], c["targets"], c["truth"]["lfc"], c["truth"]["base_cpm"], cols)
    # the knockdown itself is recovered (planted log2(0.15) = -2.74 on the target's own gene), within its error
    rs = c["rs"]
    gi = pd.Index(rs.genes).get_indexer(rs.targets)
    rows = np.flatnonzero(gi >= 0)
    kd = rs.lfc()[rows, gi[rows]]
    truth = _effective_lfc(c["truth"]["base_cpm"], c["truth"]["lfc"])[rows, gi[rows]]
    expressed = c["truth"]["base_cpm"][gi[rows]] > 20
    assert np.all(kd[expressed] < -1.5), kd[expressed]
    assert np.all(np.abs(kd - truth) < 4 * rs.se_lfc[rows, gi[rows]]), (kd, truth)


def test_replogle_pooling_by_hand():
    """Several rows per target, rows without a cell count, a duplicated and an off-axis symbol, sparse X, a
    target named 'NA' (must stay a string)."""
    rng = np.random.default_rng(5)
    axis = np.array(["A", "B", "C", "NA"])
    obs_rows = [  # name, num_cells_filtered, fold_expr, energy_test_p_value
        ("1_non-targeting_non-targeting_non-targeting", np.nan, np.nan, np.nan),
        ("2_non-targeting_non-targeting_non-targeting", 100, np.nan, 0.5),
        ("3_non-targeting_non-targeting_non-targeting", 50, np.nan, 0.6),
        ("4_non-targeting_non-targeting_non-targeting", 80, np.nan, 0.7),
        ("10_A_P1_ENSG0000000001", 10, 0.2, 0.01),
        ("11_A_P2_ENSG0000000001", 30, 0.4, 0.001),
        ("12_C_P1P2_ENSG0000000003", 20, np.nan, np.nan),
        ("13_NA_P1P2_ENSG0000000004", 25, 0.5, 0.2),
        ("14_B_P1_ENSG0000000002", np.nan, 0.3, 0.02),                 # no cell count: skipped, B has no row
    ]
    obs = pd.DataFrame([r[1:] for r in obs_rows], index=[r[0] for r in obs_rows],
                       columns=["num_cells_filtered", "fold_expr", "energy_test_p_value"])
    var = pd.DataFrame({"gene_name": ["A", "B", "B", "Z", "NA"]}, index=[f"ENSG9{i}" for i in range(5)])
    X = rng.gamma(2.0, 5.0, (len(obs), 5)).astype(np.float32)
    path = _tmpdir() / "hand_raw_bulk_01.h5ad"
    ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var).write_h5ad(path)

    rs = load_replogle_bulk(path, axis, "hand")
    assert rs.targets == ["A", "C", "NA"]
    assert rs.measured.tolist() == [True, True, False, True]
    m = rs.meta
    assert m.loc["A", "n_cells"] == 40 and m.loc["C", "n_cells"] == 20 and m.loc["NA", "n_cells"] == 25
    np.testing.assert_allclose(m.loc["A", "kd_remaining"], (10 * 0.2 + 30 * 0.4) / 40)
    assert np.isnan(m.loc["C", "kd_remaining"]) and m.loc["NA", "kd_remaining"] == 0.5
    assert m.loc["A", "phenotype_p"] == 0.001 and np.isnan(m.loc["C", "phenotype_p"])

    # by hand: CPM per row over all 5 file genes, B = columns 1 + 2, Z dropped; weighted by cells
    cpm = X.astype(np.float64) / X.sum(1, keepdims=True) * 1e6
    ax = np.stack([cpm[:, 0], cpm[:, 1] + cpm[:, 2], np.zeros(len(obs)), cpm[:, 4]], axis=1)
    ctrl = (100 * ax[1] + 50 * ax[2] + 80 * ax[3]) / 230
    np.testing.assert_allclose(rs.ctrl_cpm, ctrl, rtol=1e-6)
    np.testing.assert_allclose(rs.target_cpm[0], (10 * ax[4] + 30 * ax[5]) / 40, rtol=1e-6)
    c_ref, ref = _direct_replogle(path, axis)
    _check_against_direct(rs, c_ref, ref, ["A", "C", "NA"], ctrl_rtol=1e-6)


# The one real K562 genome-wide row whose transcript field is a RefSeq id (it contains an underscore itself).
# The old parser read its target as "ZNF735_NM" (not on the VCC axis), so ZNF735 silently had no K562 response.
REAL_REFSEQ_ROW = "10598_ZNF735_NM_001159524_ENSG00000223614"


def test_replogle_label_parsing():
    """Every transcript form found in the real Replogle files (K562 GW, K562 essential, RPE1), plus the edge cases
    the parser is designed for; and malformed names fail loudly."""
    cases = {
        REAL_REFSEQ_ROW: "ZNF735",                                         # the exact real example
        "1_non-targeting_non-targeting_non-targeting": CONTROL,            # control rows
        "5_A1BG_P1P2_ENSG00000121410": "A1BG",                             # promoter-level constructs
        "6_TP53_P1_ENSG00000141510": "TP53",
        "7_TP53_P2_ENSG00000141510": "TP53",
        "8_HLA-A_ENST00000376809.5_ENSG00000206503": "HLA-A",              # one Ensembl transcript
        "9_GENE1_ENST00000375793.2,ENST00000375794.1_ENSG00000000001": "GENE1",   # a comma-separated list
        "10_GENE2_P1P2_nan": "GENE2",                                      # a missing Ensembl gene id ("nan")
        "11_XYZ_NR_024540.1_ENSG00000000002": "XYZ",                       # versioned non-coding RefSeq id
        "12_XYZ_XM_17,XM_18_ENSG00000000002": "XYZ",                       # several RefSeq ids
        "13_MY_GENE_P1_ENSG00000000003": "MY_GENE",                        # (hypothetical) symbol with "_"
        "14_MY_GENE_NM_5_ENSG00000000003": "MY_GENE",                      # ... and a RefSeq transcript
        "15_NA_P1P2_ENSG00000000004": "NA",                                # a symbol that looks like a missing value
    }
    got = S._replogle_target_labels(list(cases))
    assert got.dtype == object and got.tolist() == list(cases.values()), got
    # the reference parser used by the direct computations agrees on every form it knows
    assert [_ref_label(s) for s in cases if not s.startswith(("13_", "14_"))] == \
        [v for s, v in cases.items() if not s.startswith(("13_", "14_"))]
    # the old rule, kept here to document the bug this guards against
    assert REAL_REFSEQ_ROW.split("_", 1)[1].rsplit("_", 2)[0] == "ZNF735_NM"
    for bad in ("abc", "1_A_ENSG1", "1__P1_ENSG1", "1_A__", ""):
        try:
            S._replogle_target_labels([bad])
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass


def test_replogle_refseq_row_is_loaded_under_its_symbol():
    """End to end through load_replogle_bulk: the RefSeq row becomes target ZNF735 (an axis gene) and is pooled with
    another ZNF735 row, weighted by cells, exactly like any other target."""
    rng = np.random.default_rng(11)
    axis = np.array(["ZNF735", "B", "C", "D"])
    obs_rows = [  # name, num_cells_filtered, fold_expr, energy_test_p_value
        ("1_non-targeting_non-targeting_non-targeting", 120, np.nan, 0.5),
        ("2_non-targeting_non-targeting_non-targeting", 90, np.nan, 0.6),
        ("3_non-targeting_non-targeting_non-targeting", 150, np.nan, 0.7),
        (REAL_REFSEQ_ROW, 40, 0.1, 0.002),
        ("10599_ZNF735_P1P2_ENSG00000223614", 60, 0.3, 0.02),              # made-up second guide row
        ("20_B_ENST00000000020.1,ENST00000000021.4_ENSG00000000020", 70, 0.2, 0.3),
    ]
    obs = pd.DataFrame([r[1:] for r in obs_rows], index=[r[0] for r in obs_rows],
                       columns=["num_cells_filtered", "fold_expr", "energy_test_p_value"])
    var = pd.DataFrame({"gene_name": list(axis)}, index=[f"ENSG8{i}" for i in range(4)])
    X = rng.gamma(2.0, 5.0, (len(obs), 4)).astype(np.float32)
    path = _tmpdir() / "refseq_raw_bulk_01.h5ad"
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(path)

    rs = load_replogle_bulk(path, axis, "refseq")
    assert rs.targets == ["ZNF735", "B"]                       # no "ZNF735_NM"
    assert rs.meta.loc["ZNF735", "n_cells"] == 100
    np.testing.assert_allclose(rs.meta.loc["ZNF735", "kd_remaining"], (40 * 0.1 + 60 * 0.3) / 100)
    assert rs.meta.loc["ZNF735", "phenotype_p"] == 0.002
    cpm = X.astype(np.float64) / X.sum(1, keepdims=True) * 1e6
    np.testing.assert_allclose(rs.target_cpm[0], (40 * cpm[3] + 60 * cpm[4]) / 100, rtol=1e-6)
    c_ref, ref = _direct_replogle(path, axis)
    _check_against_direct(rs, c_ref, ref, ["ZNF735", "B"], ctrl_rtol=1e-6)


# ---------------------------------------------------------------------------------------------------------------
# h1_pseudobulk
# ---------------------------------------------------------------------------------------------------------------

def test_h1_shapes_and_meta():
    c = _h1_case()
    rs, axis = c["rs"], c["axis"]
    T, G = N_TARGETS, N_GENES
    assert rs.name == "h1_2025"
    assert rs.genes.tolist() == axis.tolist()
    assert rs.targets == c["targets"]                       # file order, then first appearance within a file
    assert rs.target_cpm.shape == (T, G) and rs.se_lfc.shape == (T, G) and rs.ctrl_cpm.shape == (G,)
    assert rs.target_cpm.dtype == np.float32 and rs.se_lfc.dtype == np.float32 and rs.ctrl_cpm.dtype == np.float64
    assert rs.measured.tolist() == [True] * (G - N_OFF) + [False] * N_OFF
    assert list(rs.meta.columns[:3]) == META and rs.meta.index.tolist() == rs.targets
    assert (rs.meta["n_cells"] == 420).all() and rs.meta["n_cells"].dtype == np.int64
    assert rs.meta["phenotype_p"].isna().all()
    kd = rs.meta["kd_remaining"]
    assert np.isnan(kd["OFFTARGET_H1"])                     # its gene is not on the axis
    expressed = pd.Index(axis).get_indexer(rs.targets[:-1])
    ok = c["base_cpm"][expressed] > 20                      # planted 0.15 left; noisy for barely expressed genes
    np.testing.assert_allclose(kd.to_numpy()[:-1][ok], 0.15, atol=0.06)


def test_h1_matches_direct_computation():
    """Multi-file input with the same control cells in every file: controls used once, every number equal to a
    plain in-memory computation, and only the needed rows are read (each control cell once)."""
    c = _h1_case()
    c_ref, ref = _direct_h1(c["paths"], c["axis"], ctrl_files=(0,))
    _check_against_direct(c["rs"], c_ref, ref, c["targets"])

    read = []
    orig = S._read_rows
    S._read_rows = lambda X, rows: (read.append(len(rows)), orig(X, rows))[1]
    try:
        rs2 = h1_pseudobulk(c["paths"], c["axis"], verbose=False)              # default chunk size
    finally:
        S._read_rows = orig
    assert sum(read) == 600 + N_TARGETS * 420, sum(read)
    np.testing.assert_allclose(rs2.target_cpm, c["rs"].target_cpm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rs2.se_lfc, c["rs"].se_lfc, rtol=1e-6)


def test_h1_recovers_planted_effects():
    c = _h1_case()
    cols = np.arange(N_GENES - N_OFF)
    _response_checks(c["rs"], c["targets"], c["planted"], c["base_cpm"], cols)


def test_h1_distinct_control_barcodes_are_pooled():
    """Control cells are de-duplicated by barcode: if a file's controls have new barcodes they are added."""
    c = _h1_case()
    a = ad.read_h5ad(c["paths"][1])
    is_c = (a.obs["target_gene"] == CONTROL).to_numpy()
    a.obs_names = [f"other_{b}" if k else b for b, k in zip(a.obs_names, is_c)]
    p1 = _tmpdir() / "h1_fake_1_newbarcodes.h5ad"
    a.write_h5ad(p1)
    paths = [c["paths"][0], p1]
    rs = h1_pseudobulk(paths, c["axis"], verbose=False)
    c_ref, ref = _direct_h1(paths, c["axis"], ctrl_files=(0, 1))          # 1,200 control cells
    _check_against_direct(rs, c_ref, ref)


def test_h1_min_and_max_cells():
    c = _h1_case()
    # min_cells: shrink the first target of file 0 to 30 cells -> dropped at min_cells=50
    a = ad.read_h5ad(c["paths"][0])
    tg = a.obs["target_gene"].astype(str).to_numpy()
    first = c["targets"][0]
    drop = np.flatnonzero(tg == first)[30:]
    keep = np.setdiff1d(np.arange(a.n_obs), drop)
    p0 = _tmpdir() / "h1_fake_0_small_target.h5ad"
    a[keep].copy().write_h5ad(p0)
    paths = [p0, *c["paths"][1:]]
    rs = h1_pseudobulk(paths, c["axis"], min_cells=50, verbose=False)
    assert rs.targets == c["targets"][1:]
    rs_all = h1_pseudobulk(paths, c["axis"], min_cells=20, verbose=False)
    assert rs_all.targets == c["targets"] and rs_all.meta.loc[first, "n_cells"] == 30
    try:
        h1_pseudobulk(paths, c["axis"], min_cells=10_000, verbose=False)
        raise AssertionError("expected ValueError when no target has enough cells")
    except ValueError:
        pass

    # max_cells_per_target: a seeded random subset, reproducible, only those rows read
    read = []
    orig = S._read_rows
    S._read_rows = lambda X, rows: (read.append(len(rows)), orig(X, rows))[1]
    try:
        r1 = h1_pseudobulk(c["paths"], c["axis"], max_cells_per_target=100, seed=3, verbose=False)
    finally:
        S._read_rows = orig
    assert sum(read) == 600 + N_TARGETS * 100
    assert (r1.meta["n_cells"] == 100).all()
    r2 = h1_pseudobulk(c["paths"], c["axis"], max_cells_per_target=100, seed=3, verbose=False)
    r3 = h1_pseudobulk(c["paths"], c["axis"], max_cells_per_target=100, seed=4, verbose=False)
    assert np.array_equal(r1.target_cpm, r2.target_cpm) and not np.array_equal(r1.target_cpm, r3.target_cpm)
    np.testing.assert_allclose(r1.ctrl_cpm, c["rs"].ctrl_cpm)               # controls are never subsampled
    # fewer cells -> larger standard errors, same responses
    m = r1.measured
    assert np.median(r1.se_lfc[:, m] / c["rs"].se_lfc[:, m]) > 1.5
    big = np.abs(c["planted"][:, :N_GENES - N_OFF]) >= 1
    assert np.corrcoef(r1.lfc()[:, :N_GENES - N_OFF][big], c["rs"].lfc()[:, :N_GENES - N_OFF][big])[0, 1] > 0.9


def test_h1_dense_matrix_and_duplicated_symbols():
    """Fallbacks: X stored dense on disk; a symbol appearing twice in var (summed within each cell)."""
    c = _h1_case()
    a = ad.read_h5ad(c["paths"][0])
    pd_ = _tmpdir() / "h1_fake_0_dense.h5ad"
    ad.AnnData(X=a.X.toarray(), obs=a.obs, var=a.var).write_h5ad(pd_)
    paths = [pd_, *c["paths"][1:]]
    rs = h1_pseudobulk(paths, c["axis"], verbose=False)
    np.testing.assert_allclose(rs.target_cpm, c["rs"].target_cpm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rs.se_lfc, c["rs"].se_lfc, rtol=1e-6)

    rng = np.random.default_rng(9)
    axis = _axis()
    genes_small = np.concatenate([axis[:300], [axis[3], "OFFX"]])           # axis[3] twice, one off-axis gene
    targets = [str(t) for t in axis[10:14]]
    paths = [_tmpdir() / f"h1_dup_{k}.h5ad" for k in range(2)]
    fakes.make_fake_h1(paths, genes_small, targets, rng, cells_per_target=120, n_ctrl=200)
    rs = h1_pseudobulk(paths, axis[:310], min_cells=50, chunk_cells=64, verbose=False)
    assert rs.measured.tolist() == [True] * 300 + [False] * 10
    c_ref, ref = _direct_h1(paths, axis[:310])
    _check_against_direct(rs, c_ref, ref, targets)


# ---------------------------------------------------------------------------------------------------------------
# ResponseSet
# ---------------------------------------------------------------------------------------------------------------

def _assert_same(a, b):
    assert a.name == b.name and a.targets == b.targets
    assert np.array_equal(a.genes, b.genes) and np.array_equal(a.measured, b.measured)
    assert np.array_equal(a.target_cpm, b.target_cpm) and a.target_cpm.dtype == b.target_cpm.dtype
    assert np.array_equal(a.se_lfc, b.se_lfc) and a.se_lfc.dtype == b.se_lfc.dtype     # inf compares equal
    assert np.array_equal(a.ctrl_cpm, b.ctrl_cpm)
    pd.testing.assert_frame_equal(a.meta, b.meta)


def _tiny_set(targets=("NA", "NULL", "A")):
    rng = np.random.default_rng(0)
    genes = np.array(["A", "B", "C", "D"])
    T = len(targets)
    measured = np.array([True, True, False, True])
    x = rng.gamma(1.0, 10.0, (T, 4)) * measured
    se = np.where(measured, rng.uniform(0.1, 1.0, (T, 4)), np.inf)
    meta = pd.DataFrame({"n_cells": np.arange(T) + 50, "kd_remaining": [0.2, np.nan, 0.5][:T],
                         "phenotype_p": [np.nan, 1e-300, 0.123456789012345][:T]},
                        index=pd.Index(list(targets), name="target"))
    return ResponseSet("tiny", genes, measured, list(targets), x, rng.gamma(1.0, 10.0, 4) * measured, se, meta)


def test_save_load_round_trip():
    d = _tmpdir()
    for rs in (_replogle_case()["rs"], _h1_case()["rs"], _tiny_set(), _tiny_set().subset([])):
        path = d / f"rt_{rs.name}_{len(rs.targets)}.npz"
        rs.save(path)
        assert path.exists() and not Path(str(path) + ".tmp").exists()
        back = ResponseSet.load(path)
        _assert_same(rs, back)
        if rs.targets:
            assert np.array_equal(rs.lfc(), back.lfc()) and np.array_equal(rs.z(), back.z())
    # the exact path is used even without a .npz suffix; symbols like "NA"/"NULL" stay strings
    rs = _tiny_set()
    p = d / "sub" / "tiny_response_set.bin"
    rs.save(p)
    assert p.exists()
    back = ResponseSet.load(p)
    assert back.targets == ["NA", "NULL", "A"] and back.meta.index.tolist() == ["NA", "NULL", "A"]
    _assert_same(rs, back)


def test_subset():
    rs = _replogle_case()["rs"]
    t = rs.targets
    sub = rs.subset([t[3], "NOT_A_TARGET", t[1], t[3]])
    assert sub.targets == [t[3], t[1]]
    assert np.array_equal(sub.target_cpm, rs.target_cpm[[3, 1]])
    assert np.array_equal(sub.se_lfc, rs.se_lfc[[3, 1]])
    assert np.array_equal(sub.ctrl_cpm, rs.ctrl_cpm) and np.array_equal(sub.genes, rs.genes)
    assert np.array_equal(sub.measured, rs.measured)
    pd.testing.assert_frame_equal(sub.meta, rs.meta.loc[[t[3], t[1]]])
    assert np.array_equal(sub.lfc(), rs.lfc()[[3, 1]])
    sub.target_cpm[:] = -1                                   # a subset is a copy: the parent is unchanged
    assert (rs.target_cpm >= 0).all()
    empty = rs.subset(["NOPE"])
    assert empty.targets == [] and empty.target_cpm.shape == (0, N_GENES) and empty.lfc().shape == (0, N_GENES)


def _reindex_by_hand(rs, new_genes):
    """Reference for reindex_genes: one gene at a time, by name, with the documented defaults."""
    T, G = len(rs.targets), len(new_genes)
    x, se = np.zeros((T, G), np.float32), np.full((T, G), np.inf, np.float32)
    c, m = np.zeros(G), np.zeros(G, bool)
    old = list(rs.genes)
    for j, g in enumerate(new_genes):
        if g in old and rs.measured[old.index(g)]:
            k = old.index(g)
            x[:, j], se[:, j], c[j], m[j] = rs.target_cpm[:, k], rs.se_lfc[:, k], rs.ctrl_cpm[k], True
    return x, se, c, m


def _silently(f, *a):
    """f(*a), asserting it printed nothing (reindex_genes is called in loops by 03 and must stay quiet)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        r = f(*a)
    assert out.getvalue() == "" and err.getvalue() == "", (out.getvalue(), err.getvalue())
    return r


def test_reindex_genes_small_by_hand():
    rs = _tiny_set()                                  # genes A B C D, C not measured by the source
    r = _silently(rs.reindex_genes, ["D", "NEW", "A", "C"])
    assert r.genes.tolist() == ["D", "NEW", "A", "C"]
    assert r.measured.tolist() == [True, False, True, False]   # NEW: absent; C: present but never measured
    np.testing.assert_array_equal(r.target_cpm[:, 0], rs.target_cpm[:, 3])
    np.testing.assert_array_equal(r.target_cpm[:, 2], rs.target_cpm[:, 0])
    np.testing.assert_array_equal(r.se_lfc[:, [0, 2]], rs.se_lfc[:, [3, 0]])
    assert np.all(r.target_cpm[:, [1, 3]] == 0) and np.all(np.isinf(r.se_lfc[:, [1, 3]]))
    assert r.ctrl_cpm.tolist() == [rs.ctrl_cpm[3], 0.0, rs.ctrl_cpm[0], 0.0]
    assert r.name == rs.name and r.targets == rs.targets
    pd.testing.assert_frame_equal(r.meta, rs.meta)
    assert r.target_cpm.dtype == np.float32 and r.se_lfc.dtype == np.float32 and r.ctrl_cpm.dtype == np.float64
    # derived quantities follow the columns; the new / unmeasured columns are exactly 0 (no evidence)
    np.testing.assert_array_equal(r.lfc()[:, [0, 2]], rs.lfc()[:, [3, 0]])
    np.testing.assert_array_equal(r.z()[:, [0, 2]], rs.z()[:, [3, 0]])
    assert np.all(r.lfc()[:, [1, 3]] == 0) and np.all(r.z()[:, [1, 3]] == 0)
    # dropping genes: B is gone, nothing else changes
    r2 = rs.reindex_genes(["A", "D"])
    assert r2.genes.tolist() == ["A", "D"] and r2.measured.all()
    np.testing.assert_array_equal(r2.target_cpm, rs.target_cpm[:, [0, 3]])
    # an empty set and a set reindexed onto itself
    e = rs.subset([]).reindex_genes(["X", "A"])
    assert e.targets == [] and e.target_cpm.shape == (0, 2) and e.se_lfc.shape == (0, 2)
    assert e.measured.tolist() == [False, True]
    _assert_same(rs.reindex_genes(rs.genes), rs)


def test_reindex_genes_on_a_loaded_screen():
    """A shuffled axis that drops some genes and adds new ones, on the fake K562 screen; small row blocks to exercise
    the block-wise copy. Also: superset round trip, copy semantics, save/load, and bad axes."""
    rs = _replogle_case()["rs"]
    rng = np.random.default_rng(21)
    keep = rng.permutation(rs.genes)[: N_GENES - 300]                     # 300 axis genes dropped, order shuffled
    new = np.concatenate([keep[:1000], [f"NEWGENE{i}" for i in range(50)], keep[1000:]])
    orig_block = S._BLOCK_TARGETS
    S._BLOCK_TARGETS = 3                                                  # 20 targets -> 7 blocks
    try:
        r = _silently(rs.reindex_genes, new)
    finally:
        S._BLOCK_TARGETS = orig_block
    x, se, c, m = _reindex_by_hand(rs, list(new))
    assert r.genes.tolist() == new.tolist()
    np.testing.assert_array_equal(r.measured, m)
    np.testing.assert_array_equal(r.target_cpm, x)
    np.testing.assert_array_equal(r.se_lfc, se)
    np.testing.assert_array_equal(r.ctrl_cpm, c)
    assert r.targets == rs.targets
    pd.testing.assert_frame_equal(r.meta, rs.meta)
    assert m.sum() == np.isin(new, rs.genes[rs.measured]).sum()       # every measured kept gene stays measured

    # superset (old axis shuffled + new genes) and back: exactly the original set
    sup = np.concatenate([[f"EXTRA{i}" for i in range(7)], rng.permutation(rs.genes)])
    _assert_same(rs.reindex_genes(sup).reindex_genes(rs.genes), rs)

    # a copy: changing the result leaves the source untouched, even when the axis is unchanged
    same = rs.reindex_genes(rs.genes)
    assert not np.shares_memory(same.target_cpm, rs.target_cpm) and not np.shares_memory(same.se_lfc, rs.se_lfc)
    same.target_cpm[:] = -1
    same.meta.loc[same.targets[0], "n_cells"] = -1
    assert (rs.target_cpm >= 0).all() and (rs.meta["n_cells"] > 0).all()

    # a reindexed set saves and loads like any other
    p = _tmpdir() / "reindexed.npz"
    r.save(p)
    _assert_same(ResponseSet.load(p), r)

    # bad axes: a repeated symbol in the new axis, or in the set's own axis, is refused
    for bad_call in (lambda: rs.reindex_genes(np.concatenate([new, new[:1]])),
                     lambda: ResponseSet("dup", ["A", "A"], [True, True], ["T1"], np.ones((1, 2)), np.ones(2),
                                         np.ones((1, 2)), pd.DataFrame({"n_cells": [50], "kd_remaining": [np.nan],
                                                                        "phenotype_p": [np.nan]}, index=["T1"])
                                         ).reindex_genes(["A"])):
        try:
            bad_call()
            raise AssertionError("expected ValueError for a repeated gene symbol")
        except ValueError:
            pass


def test_lfc_and_z_definitions():
    rs = _tiny_set()
    lfc = rs.lfc()
    expected = np.log2((rs.target_cpm.astype(np.float64) + EPS_CPM) / (rs.ctrl_cpm + EPS_CPM))
    expected[:, ~rs.measured] = 0
    np.testing.assert_allclose(lfc, expected, rtol=1e-5, atol=1e-6)
    assert lfc.dtype == np.float32
    z = rs.z()
    np.testing.assert_allclose(z[:, rs.measured], (expected / rs.se_lfc)[:, rs.measured], rtol=1e-5)
    assert np.all(z[:, ~rs.measured] == 0)
    lfc1 = rs.lfc(eps=1.0)
    np.testing.assert_allclose(lfc1[:, rs.measured], np.log2((rs.target_cpm + 1.0) / (rs.ctrl_cpm + 1.0))
                               [:, rs.measured], rtol=1e-5)
    # a zero standard error (no evidence either way) gives z = 0, not inf/nan
    rs.se_lfc[0, 0] = 0
    assert rs.z()[0, 0] == 0
    # a broadcastable se row (one value per gene for every target) is expanded to a full (T, G) array
    one_row = np.where(rs.measured, 0.2, np.inf)[None, :]
    rb = ResponseSet("b", rs.genes, rs.measured, rs.targets, rs.target_cpm, rs.ctrl_cpm, one_row, rs.meta)
    assert rb.se_lfc.shape == rs.target_cpm.shape and rb.se_lfc.flags.writeable and rb.se_lfc[2, 0] == np.float32(0.2)
    # construction checks catch shape and index mistakes
    for bad in (dict(targets=["NA", "NA", "A"]), dict(ctrl_cpm=np.zeros(3)), dict(se_lfc=np.ones((2, 4))),
                dict(meta=rs.meta.iloc[::-1])):
        args = dict(name="x", genes=rs.genes, measured=rs.measured, targets=rs.targets, target_cpm=rs.target_cpm,
                    ctrl_cpm=rs.ctrl_cpm, se_lfc=rs.se_lfc, meta=rs.meta)
        args.update(bad)
        try:
            ResponseSet(**args)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass


# ---------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    t_all = time.time()
    for n, f in tests:
        t0 = time.time()
        try:
            f()
            print(f"PASS  {n}  ({time.time() - t0:.1f} s)", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL  {n}  ({time.time() - t0:.1f} s)", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed in {time.time() - t_all:.0f} s")
    sys.exit(1 if failed else 0)
