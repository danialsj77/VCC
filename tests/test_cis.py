"""Tests for vccpipe.cis: GTF parsing, the binned cis-effect fit, prediction, and JSON persistence.

Run with pytest, or directly:  .venv/bin/python tests/test_cis.py

Only fake data is used: a GENCODE-like GTF from tests/fakes.make_fake_gtf (or written by hand) in a temporary
directory, and a synthetic ResponseSet with a planted cis effect. The real gene axis (controls/gene_names.csv)
supplies realistic symbols and the full 18,533-gene size for the scale test; no large data file is read.

Every check on the vectorized neighbour search is made against an independent brute-force implementation
(`_brute_pairs`), which loops over targets and compares each one with every gene directly.
"""

import gzip
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):           # import vccpipe from the project and fakes.py from tests/
    if p not in sys.path:
        sys.path.insert(0, p)

from fakes import make_fake_gtf, vcc_genes  # noqa: E402
from vccpipe.cis import CisModel, cis_pairs, load_tss  # noqa: E402

CONTROLS = ROOT / "controls"
PLANT = {0: -1.5, 1: -0.5}                 # planted cis log2FC: bin [0,1) kb → -1.5, [1,2) kb → -0.5, else 0
NOISE_SD = 0.2


# --------------------------------------------------------------------------------------------------------------
# Synthetic ResponseSet (the real dataclass if vccpipe.sources exists, else a stand-in with the same attributes)
# --------------------------------------------------------------------------------------------------------------

class _StandInRS:
    """Minimal object with the ResponseSet attributes CisModel uses: genes, measured, targets, lfc(), meta,
    ctrl_cpm. Used when vccpipe.sources is not available, and for the scale test (no dense CPM matrices)."""

    def __init__(self, genes, measured, targets, lfc, n_cells, ctrl_cpm=None):
        self.name = "standin"
        self.genes = np.asarray(genes)
        self.measured = np.asarray(measured, dtype=bool)
        self.targets = list(targets)
        self._lfc = lfc
        self.ctrl_cpm = ctrl_cpm
        self.meta = pd.DataFrame({"n_cells": np.asarray(n_cells, dtype=np.int64)}, index=self.targets)

    def lfc(self, eps=None):
        return self._lfc


def make_rs(genes, measured, targets, lfc, n_cells, ctrl_cpm):
    """A ResponseSet whose lfc() equals `lfc` on measured genes and 0 elsewhere.

    With the real dataclass the planted log2FC is encoded in target_cpm: target = (ctrl + eps) * 2^lfc - eps,
    which lfc() inverts exactly (up to float32 rounding)."""
    lfc = np.where(measured[None, :], lfc, 0.0)
    try:
        from vccpipe.sources import EPS_CPM, ResponseSet
    except ImportError:
        return _StandInRS(genes, measured, targets, lfc, n_cells, ctrl_cpm), "stand-in"
    ctrl = np.where(measured, ctrl_cpm, 0.0).astype(np.float64)
    tgt = np.where(measured[None, :], (ctrl[None, :] + EPS_CPM) * np.exp2(lfc) - EPS_CPM, 0.0)
    se = np.where(measured[None, :], 0.1, np.inf).astype(np.float32)
    meta = pd.DataFrame({"n_cells": np.asarray(n_cells, dtype=np.int64), "kd_remaining": np.nan,
                         "phenotype_p": np.nan}, index=list(targets))
    rs = ResponseSet(name="fake", genes=np.asarray(genes), measured=np.asarray(measured, dtype=bool),
                     targets=list(targets), target_cpm=tgt.astype(np.float32), ctrl_cpm=ctrl, se_lfc=se, meta=meta)
    return rs, "vccpipe.sources.ResponseSet"


# --------------------------------------------------------------------------------------------------------------
# Brute-force reference
# --------------------------------------------------------------------------------------------------------------

def _tss_lookup(tss):
    first = tss.drop_duplicates("gene_name", keep="first")
    return {g: (c, int(p)) for g, c, p in zip(first["gene_name"], first["chrom"], first["tss"])}


def _brute_pairs(targets, genes, tss, max_kb, target_ok=None, gene_ok=None):
    """(t_idx, g_idx, distance_kb) by direct comparison of each target with every gene (independent reference)."""
    look = _tss_lookup(tss)
    genes = np.asarray(genes).astype(str)
    g_chrom = np.array([look[g][0] if g in look else "" for g in genes], dtype=object)
    g_pos = np.array([look[g][1] if g in look else 0 for g in genes], dtype=np.int64)
    g_known = np.array([g in look for g in genes])
    if gene_ok is not None:
        g_known &= np.asarray(gene_ok, dtype=bool)
    T, G, D = [], [], []
    for i, t in enumerate(targets):
        if (target_ok is not None and not target_ok[i]) or t not in look:
            continue
        c, p = look[t]
        d = np.abs(g_pos - p) / 1000.0
        hit = np.flatnonzero(g_known & (g_chrom == c) & (d < max_kb) & (genes != t))
        T += [i] * hit.size
        G += hit.tolist()
        D += d[hit].tolist()
    return np.array(T, dtype=np.int64), np.array(G, dtype=np.int64), np.array(D)


def _bin_of(d_kb, edges):
    b = np.searchsorted(np.asarray(edges, dtype=float), d_kb, side="right") - 1
    return np.where((b >= 0) & (b < len(edges) - 1), b, -1)


# --------------------------------------------------------------------------------------------------------------
# A small genome with a planted cis effect
# --------------------------------------------------------------------------------------------------------------

def _scenario(tmp, seed=0):
    """600 real VCC symbols on 3 short chromosomes (~1.5 kb between neighbouring TSSs, so every distance bin is
    well populated), a source that measured ~85% of the axis, and 150 targets with a planted cis effect.

    Extra cases built in:
      * two axis genes with no TSS (never paired);
      * a target that is not on the axis but has a TSS (still usable: only its TSS matters);
      * a target with no TSS (ignored);
      * five low-cell targets (n_cells < 50) whose neighbours are planted at +2.0: they would drag the medians up
        if they were not excluded;
      * every target's own gene knocked down to -3.0 at distance 0: it would land in bin 0 if not excluded;
      * ~10% of genes with low control expression (for the optional min_ctrl_cpm filter).
    """
    rng = np.random.default_rng(seed)
    sym = vcc_genes(CONTROLS, n=600)
    gtf = Path(tmp) / "fake.gtf.gz"
    truth = make_fake_gtf(gtf, list(sym) + ["OFFAXIS_T"], rng, chrom_len=300_000)
    tss = load_tss(gtf)
    axis = np.array(list(sym[:590]) + ["NOTSS_A", "NOTSS_B"])
    G = axis.size
    measured = rng.random(G) < 0.85
    ctrl_cpm = np.where(rng.random(G) < 0.1, 1.0, 100.0)

    good = list(rng.choice(sym[:590], 150, replace=False))
    low = list(rng.choice(np.setdiff1d(sym[:590], good), 5, replace=False))
    targets = good + ["OFFAXIS_T", "NOTSS_T"] + low
    n_cells = np.array([200] * len(good) + [120, 300] + [10] * len(low))

    # Planted log2FC: noise everywhere, the cis step function near each target, the knockdown on its own gene.
    lfc = rng.normal(0.0, NOISE_SD, (len(targets), G))
    ti, gi, d = _brute_pairs(targets, axis, tss, 50.0)       # all axis genes near each target (any measurement)
    b = _bin_of(d, [0, 1, 2, 5, 10, 20, 50])
    is_low = np.isin(np.asarray(targets)[ti], low)
    for k, v in PLANT.items():
        sel = (b == k) & ~is_low
        lfc[ti[sel], gi[sel]] += v
    lfc[ti[is_low], gi[is_low]] += 2.0
    col = {g: j for j, g in enumerate(axis)}
    for i, t in enumerate(targets):
        if t in col:
            lfc[i, col[t]] = -3.0
    rs, kind = make_rs(axis, measured, targets, lfc, n_cells, ctrl_cpm)
    return dict(rs=rs, kind=kind, tss=tss, truth=truth, axis=axis, measured=measured, targets=targets,
                n_cells=n_cells, lfc=np.where(measured[None, :], lfc, 0.0), ctrl_cpm=ctrl_cpm, low=low)


# --------------------------------------------------------------------------------------------------------------
# GTF parsing
# --------------------------------------------------------------------------------------------------------------

def test_load_tss_matches_fake_gtf():
    """The fake GENCODE file (header line, gene + exon records, both strands) parses to the planted table."""
    rng = np.random.default_rng(1)
    genes = vcc_genes(CONTROLS, n=300)
    with tempfile.TemporaryDirectory() as tmp:
        truth = make_fake_gtf(Path(tmp) / "g.gtf.gz", genes, rng)
        got = load_tss(Path(tmp) / "g.gtf.gz")
    assert list(got.columns) == ["gene_name", "chrom", "tss", "strand"]
    assert got["tss"].dtype == np.int64
    assert got["gene_name"].is_unique and len(got) == len(genes)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), truth.astype({"tss": np.int64}),
                                  check_dtype=False)
    # Strand awareness: the fake's bidirectional pairs have TSSs exactly 300 bp apart.
    first = got.set_index("gene_name")
    for k in range(0, 40, 2):
        a, b = genes[k], genes[k + 1]
        assert first.loc[a, "strand"] != first.loc[b, "strand"]
        assert abs(int(first.loc[a, "tss"]) - int(first.loc[b, "tss"])) == 300


def _write_lines(path, lines, gz=True):
    opener = gzip.open if gz else open
    with opener(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")


def _rec(chrom, feature, start, end, strand, attrs):
    return f"{chrom}\tHAVANA\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t{attrs}"


def test_load_tss_rules():
    """Strand-aware TSS, gene_type filter, first record per symbol wins, non-gene records and comments ignored."""
    a = lambda name, gtype, extra="": (f'gene_id "ENSG_{name}"; gene_type "{gtype}"; gene_name "{name}"; '
                                       f'level 2;{extra}')
    lines = [
        "#!genome-build GRCh38.p14",
        "##provider: GENCODE",
        _rec("chr1", "gene", 1000, 5000, "+", a("GENEA", "protein_coding", ' tag "basic";')),
        _rec("chr1", "transcript", 1500, 5000, "+", a("GENEA", "protein_coding")),   # not a gene record
        _rec("chr1", "gene", 2000, 9000, "-", a("GENEB", "lncRNA")),                  # − strand: TSS = end
        _rec("chr2", "gene", 100, 180, "+", a("MIR1", "miRNA")),                      # filtered type
        _rec("chr2", "gene", 400, 900, "-", a("PSG1", "processed_pseudogene")),       # filtered type
        _rec("chr2", "gene", 50, 60, "+", a("DUPT", "misc_RNA")),                     # filtered, then ...
        _rec("chr2", "gene", 7000, 8000, "-", a("DUPT", "protein_coding")),           # ... first KEPT record wins
        _rec("chrX", "gene", 100, 200, "+", a("PARG", "protein_coding")),             # PAR gene: chrX first ...
        _rec("chrY", "gene", 300, 400, "+", a("PARG", "protein_coding")),             # ... chrY copy ignored
        _rec("chr3", "exon", 10, 20, "+", a("ONLYEXON", "protein_coding")),           # no gene record → absent
        _rec("chr3", "gene", 10, 20, "+", 'gene_id "ENSG_NONAME"; gene_type "protein_coding"; level 2;'),
        _rec("chr4", "gene", 500, 600, "-",                                            # Ensembl-style attribute
             'gene_id "ENSG_E"; gene_name "ENSEMBLSTYLE"; gene_biotype "protein_coding";'),
        _rec("chr4", "gene", 700, 800, "+",                                            # "havana_gene_type" must
             'gene_id "ENSG_H"; havana_gene_type "protein_coding"; gene_type "miRNA"; gene_name "HAV";'),
    ]
    expected = pd.DataFrame({"gene_name": ["GENEA", "GENEB", "DUPT", "PARG", "ENSEMBLSTYLE"],
                             "chrom": ["chr1", "chr1", "chr2", "chrX", "chr4"],
                             "tss": np.array([1000, 9000, 8000, 100, 600], dtype=np.int64),
                             "strand": ["+", "-", "-", "+", "-"]})
    with tempfile.TemporaryDirectory() as tmp:
        for gz in (True, False):                                 # gzipped and plain text give the same table
            path = Path(tmp) / ("x.gtf.gz" if gz else "x.gtf")
            _write_lines(path, lines, gz=gz)
            got = load_tss(path)
            pd.testing.assert_frame_equal(got, expected, check_dtype=False)
            assert got["tss"].dtype == np.int64
        only_mirna = load_tss(Path(tmp) / "x.gtf.gz", gene_types=("miRNA",))
        assert only_mirna["gene_name"].tolist() == ["MIR1", "HAV"]


# --------------------------------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------------------------------

def test_fit_recovers_planted_bins():
    with tempfile.TemporaryDirectory() as tmp:
        s = _scenario(tmp)
    rs, tss, axis = s["rs"], s["tss"], s["axis"]
    print(f"    (ResponseSet used: {s['kind']})")
    lfc_rs = rs.lfc()
    assert np.allclose(lfc_rs, s["lfc"], atol=1e-4), "synthetic ResponseSet does not reproduce the planted lfc"

    m = CisModel().fit(rs, tss)
    edges = m.bins_kb
    assert edges == [0, 1, 2, 5, 10, 20, 50] and len(m.effect) == 6 and len(m.n_pairs) == 6

    # Exactly the pairs of the brute-force reference: good targets only, measured genes only, own gene excluded.
    target_ok = s["n_cells"] >= 50
    bt, bg, bd = _brute_pairs(s["targets"], axis, tss, 50.0, target_ok, s["measured"])
    bb = _bin_of(bd, edges)
    assert m.n_pairs == np.bincount(bb, minlength=6).tolist()
    for k in range(6):
        ref = np.median(lfc_rs[bt[bb == k], bg[bb == k]]) if (bb == k).sum() >= 20 else 0.0
        assert abs(m.effect[k] - ref) < 1e-9, (k, m.effect[k], ref)

    # ... and those medians recover the planted effect.
    planted = [PLANT.get(k, 0.0) for k in range(6)]
    assert all(n >= 20 for n in m.n_pairs), m.n_pairs
    assert np.allclose(m.effect, planted, atol=0.1), (m.effect, planted)
    print("    fitted:\n" + "\n".join("    " + line for line in m.summary().to_string(index=False).splitlines()))

    # The pair table itself matches the reference (same pairs, distances and values).
    pt = cis_pairs(rs, tss)
    got = pt.sort_values(["target", "gene"]).reset_index(drop=True)
    ref = pd.DataFrame({"target": np.asarray(s["targets"])[bt], "gene": axis[bg], "distance_kb": bd,
                        "lfc": lfc_rs[bt, bg].astype(np.float64)}).sort_values(["target", "gene"])
    pd.testing.assert_frame_equal(got, ref.reset_index(drop=True), check_dtype=False)
    assert not np.isin(pt["target"], s["low"] + ["NOTSS_T"]).any()
    assert (pt["target"] != pt["gene"]).all()
    assert "OFFAXIS_T" in set(pt["target"])                      # off-axis target with a TSS is used


def test_fit_min_pairs_and_filters():
    with tempfile.TemporaryDirectory() as tmp:
        s = _scenario(tmp, seed=3)
    rs, tss = s["rs"], s["tss"]
    m = CisModel().fit(rs, tss)
    n1 = m.n_pairs[1]
    assert CisModel().fit(rs, tss, min_pairs=n1).effect[1] == m.effect[1] != 0.0
    assert CisModel().fit(rs, tss, min_pairs=n1 + 1).effect[1] == 0.0      # too few pairs → 0
    assert CisModel().fit(rs, tss, min_pairs=10 ** 9).effect == [0.0] * 6

    # min_cells: lowering it lets the five low-cell targets (planted +2.0) in, which shifts the far bins up.
    loose = CisModel().fit(rs, tss, min_cells=1)
    assert sum(loose.n_pairs) > sum(m.n_pairs)
    assert loose.effect[-1] > m.effect[-1]

    # Optional extension: min_ctrl_cpm drops lowly expressed neighbour genes.
    hi = CisModel().fit(rs, tss, min_ctrl_cpm=5.0)
    bt, bg, bd = _brute_pairs(s["targets"], s["axis"], tss, 50.0, s["n_cells"] >= 50,
                              s["measured"] & (s["ctrl_cpm"] >= 5.0))
    assert hi.n_pairs == np.bincount(_bin_of(bd, hi.bins_kb), minlength=6).tolist()

    # Custom bins: distances below the first edge belong to no bin; the window is the last edge.
    c = CisModel(bins_kb=[0.5, 1.5, 3.0]).fit(rs, tss)
    bt, bg, bd = _brute_pairs(s["targets"], s["axis"], tss, 3.0, s["n_cells"] >= 50, s["measured"])
    bb = _bin_of(bd, [0.5, 1.5, 3.0])
    assert c.n_pairs == [int((bb == 0).sum()), int((bb == 1).sum())]

    # A source with no usable pairs: a valid, all-zero model (not an error).
    empty = CisModel().fit(rs, tss.iloc[:0])
    assert empty.effect == [0.0] * 6 and empty.n_pairs == [0] * 6


# --------------------------------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------------------------------

def test_predict_places_effect():
    with tempfile.TemporaryDirectory() as tmp:
        s = _scenario(tmp, seed=5)
    tss, axis = s["tss"], s["axis"]
    m = CisModel().fit(s["rs"], tss)
    fitted = s["targets"][:20]
    unseen = [g for g in axis[:590] if g not in s["targets"]][:5]      # targets no source measured
    targets = fitted + unseen + ["OFFAXIS_T", "NOTSS_T", "NOT_A_GENE"]
    out = m.predict(targets, axis, tss)
    assert out.shape == (len(targets), axis.size) and out.dtype == np.float32

    # Exact expected matrix from the brute-force pairs (every axis gene, measured or not).
    expected = np.zeros_like(out)
    ti, gi, d = _brute_pairs(targets, axis, tss, 50.0)
    b = _bin_of(d, m.bins_kb)
    expected[ti, gi] = np.asarray(m.effect, dtype=np.float32)[b]
    assert np.array_equal(out, expected)

    # The planted effect sits on the right entries: genes < 1 kb from the target ≈ -1.5, 1-2 kb ≈ -0.5.
    assert np.allclose(out[ti[b == 0], gi[b == 0]], -1.5, atol=0.1)
    assert np.allclose(out[ti[b == 1], gi[b == 1]], -0.5, atol=0.1)
    # Unmeasured genes near a target still get the effect; far genes and other chromosomes get 0.
    near_unmeasured = (b == 0) & ~s["measured"][gi]
    assert near_unmeasured.any() and np.all(out[ti[near_unmeasured], gi[near_unmeasured]] < -1.0)
    assert (out != 0).sum() == np.count_nonzero(np.asarray(m.effect)[b])
    # Own column 0 (the knockdown is modelled elsewhere); rows with no TSS all 0.
    col = {g: j for j, g in enumerate(axis)}
    for i, t in enumerate(targets):
        if t in col:
            assert out[i, col[t]] == 0.0
    assert not out[targets.index("NOTSS_T")].any() and not out[targets.index("NOT_A_GENE")].any()
    assert out[targets.index("OFFAXIS_T")].any()
    # Genes without a TSS never get an effect.
    assert not out[:, np.isin(axis, ["NOTSS_A", "NOTSS_B"])].any()

    # The fake's bidirectional partners (TSSs 300 bp apart) are predicted to be silenced.
    sym = list(axis)
    a, bb = sym[0], sym[1]
    pair = m.predict([a], axis, tss)
    assert pair[0, col[bb]] == np.float32(m.effect[0]) and pair[0, col[a]] == 0.0

    # Not fitted → clear error.
    try:
        CisModel().predict(targets, axis, tss)
    except ValueError:
        pass
    else:
        raise AssertionError("an unfitted model should refuse to predict")


# --------------------------------------------------------------------------------------------------------------
# Persistence and validation
# --------------------------------------------------------------------------------------------------------------

def test_json_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        s = _scenario(tmp, seed=7)
    m = CisModel().fit(s["rs"], s["tss"])
    m2 = CisModel.from_json(m.to_json())
    assert m2 == m
    assert np.array_equal(m2.predict(s["targets"], s["axis"], s["tss"]),
                          m.predict(s["targets"], s["axis"], s["tss"]))
    with tempfile.TemporaryDirectory() as tmp:                      # through a file, as the scripts do
        p = Path(tmp) / "cis.json"
        p.write_text(m.to_json())
        assert CisModel.from_json(p.read_text()) == m
    u = CisModel(bins_kb=[0, 2, 10])
    assert CisModel.from_json(u.to_json()) == u and u.effect is None
    # Unknown keys (e.g. added by a later version) are ignored.
    assert CisModel.from_json('{"bins_kb": [0, 1, 50], "effect": [-1, 0], "n_pairs": [30, 40], "note": "x"}') \
        == CisModel(bins_kb=[0, 1, 50], effect=[-1.0, 0.0], n_pairs=[30, 40])
    for bad in (dict(bins_kb=[0, 5, 5]), dict(bins_kb=[1]), dict(bins_kb=[0, 1, 2], effect=[0.0])):
        try:
            CisModel(**bad)
        except ValueError:
            continue
        raise AssertionError(f"invalid parameters accepted: {bad}")


# --------------------------------------------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------------------------------------------

def test_scale_k562_size():
    """K562-sized fit (~9,866 targets x the 18,533-gene VCC axis) and a 300-target panel prediction run in seconds.

    TSSs are random on 23 chromosomes of realistic length. The stand-in's lfc() returns a lazily allocated
    zero matrix (calloc'd pages are not touched), so this test measures the neighbour search, not the memory of a
    dense ResponseSet."""
    rng = np.random.default_rng(11)
    axis = vcc_genes(CONTROLS)
    G = axis.size
    lengths_mb = [248, 242, 198, 190, 181, 171, 159, 145, 138, 134, 135, 133, 114, 107, 102, 90, 83, 80, 59, 64,
                  47, 51, 156]
    chrom_of = rng.choice(len(lengths_mb), G, p=np.asarray(lengths_mb) / sum(lengths_mb))
    pos = (rng.random(G) * np.asarray(lengths_mb)[chrom_of] * 1e6).astype(np.int64) + 1
    tss = pd.DataFrame({"gene_name": axis, "chrom": [f"chr{c + 1}" for c in chrom_of], "tss": pos,
                        "strand": np.where(rng.random(G) < 0.5, "+", "-")})
    T = 9866
    targets = list(rng.choice(axis, T, replace=False))
    measured = rng.random(G) < 0.6
    lfc = np.zeros((T, G), dtype=np.float32)
    rs = _StandInRS(axis, measured, targets, lfc, n_cells=rng.integers(20, 300, T))

    t0 = time.time()
    m = CisModel().fit(rs, tss)
    t_fit = time.time() - t0
    t0 = time.time()
    out = m.predict(targets[:300], axis, tss)
    t_pred = time.time() - t0
    print(f"    fit {T} targets x {G} genes: {t_fit:.2f} s ({sum(m.n_pairs)} pairs); "
          f"predict 300 targets: {t_pred:.2f} s")
    assert t_fit < 20 and t_pred < 5
    assert out.shape == (300, G)

    # Pair counts agree with the brute-force reference on a random sample of 200 targets.
    sample = sorted(rng.choice(T, 200, replace=False))
    sub = [targets[i] for i in sample]
    ok = rs.meta["n_cells"].to_numpy()[sample] >= 50
    bt, bg, bd = _brute_pairs(sub, axis, tss, 50.0, ok, measured)
    pt = cis_pairs(rs, tss)
    got = pt[pt["target"].isin(sub)].sort_values(["target", "gene"])
    ref = pd.DataFrame({"target": np.asarray(sub)[bt], "gene": axis[bg]}).sort_values(["target", "gene"])
    assert got[["target", "gene"]].to_numpy().tolist() == ref.to_numpy().tolist()


# --------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f} s)", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL  {name}  ({time.time() - t0:.1f} s)", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
