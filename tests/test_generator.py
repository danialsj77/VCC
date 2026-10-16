"""Tests for vccpipe.generator: both generators must reproduce the predicted mean expression and obey the upload rules.

Also checks the depth-aware NB generator's pseudobulk realism (the scorer's PDS/MSE space) and the per-target seeding
of vccpipe.harness.write_prediction.

Uses 3,000 real control cells of context A (light enough for a laptop). Run: .venv/bin/python tests/test_generator.py
(~1-2 min).
"""

import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vccpipe.common import load_bundle_info, load_controls, mean_cpm  # noqa: E402
from vccpipe.generator import NBGenerator, TemplateGenerator  # noqa: E402
from vccpipe.harness import write_prediction  # noqa: E402

_cache = {}


def _controls():
    if "X" not in _cache:
        genes, _, _ = load_bundle_info(ROOT / "controls")
        X = load_controls(ROOT / "controls", "A", genes)
        _cache["genes"] = genes
        _cache["X"] = X[np.random.default_rng(0).choice(X.shape[0], 3000, replace=False)]
    return _cache["X"]


def _genes():
    _controls()
    return _cache["genes"]


def _pseudobulk(X):
    """A group of cells in the scorer's PDS/MSE space: log1p(5e4 × summed counts / total counts) per gene
    (cell_eval2's vcc2026 profile, bulk_target_sum = 5e4)."""
    s = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    return np.log1p(5e4 * s / s.sum())


def _planted(cm, rng):
    """A prediction: 100 genes at 85% knockdown, 100 up 2×, 100 down 2× (all well expressed), rest unchanged."""
    idx = rng.choice(np.flatnonzero(cm > 20), 300, replace=False)
    lfc = np.zeros(cm.size)
    lfc[idx[:100]], lfc[idx[100:200]], lfc[idx[200:]] = np.log2(0.15), 1.0, -1.0
    return idx, lfc, cm * np.exp2(lfc)


def _check_means(gen, cm, n_cells=3000, tol=0.08):
    rng = np.random.default_rng(1)
    idx, lfc, target = _planted(cm, rng)
    out = gen(target, n_cells, rng)
    assert out.data.min() >= 0 and not np.modf(out.data)[0].any(), "counts must be non-negative integers"
    assert out.sum(axis=1).max() <= 1_000_000
    realized = np.log2((mean_cpm(out)[idx] + 1e-9) / (cm[idx] + 1e-9))
    for name, sl in (("knockdown", slice(0, 100)), ("up", slice(100, 200)), ("down", slice(200, 300))):
        err = abs(realized[sl].mean() - lfc[idx][sl].mean())
        assert err < tol, f"{name}: mean realized log2FC off by {err:.3f}"
    return out


def test_template_generator_preserves_means():
    X = _controls()
    gen = TemplateGenerator(X)
    _check_means(gen, gen.ctrl_mean)


def test_template_generator_spread_preserves_means():
    X = _controls()
    gen = TemplateGenerator(X, spread=0.5)
    _check_means(gen, gen.ctrl_mean)


def test_nb_generator_preserves_means_and_depth():
    X = _controls()
    gen = NBGenerator(X)
    out = _check_means(gen, gen.ctrl_mean)
    ratio = np.median(np.asarray(out.sum(axis=1)).ravel()) / np.median(gen.lib)
    assert 0.9 < ratio < 1.1, f"median depth ratio {ratio:.2f}"
    assert np.all(gen.phi >= 0) and np.isfinite(gen.phi).all()


def test_null_prediction_creates_few_false_calls():
    """Predicting 'no change' must not shift unaffected genes: compare generated vs real control means."""
    X = _controls()
    for gen in (TemplateGenerator(X), NBGenerator(X)):
        out = gen(gen.ctrl_mean, 3000, np.random.default_rng(2))
        m = mean_cpm(out)
        well = gen.ctrl_mean > 50
        rel = np.abs(m[well] / gen.ctrl_mean[well] - 1)
        assert np.median(rel) < 0.05, f"{type(gen).__name__}: median relative shift {np.median(rel):.3f}"


def test_nb_no_change_pseudobulk_matches_real_subsets():
    """A 'no change' NB group must sit as far from the control pseudobulk as a real 400-cell group does.

    Statistic: squared distance ||pb(group) − pb(all 3,000 controls)||² in the scorer's space (sum over all genes).
    It is pure sampling noise for a real group, and noise + any systematic bias for a generated one. The depth-blind
    NB (one share per gene) added a bias of ~25% here; the depth-aware one must be within 15% of real groups.

    The yardstick resamples the 3,000 cells WITH replacement (bootstrap): a generator fitted on these cells draws
    independent cells, which is exactly what a bootstrap does. Drawing 400 of 3,000 cells without replacement would
    shrink the noise by the finite-population factor (1 − 400/3000 ≈ 0.87) and make even a perfect generator look
    ~15% too noisy (with all 18,400 context A controls the factor is 0.98, so the two yardsticks agree there).
    Measured here: real 14.4, depth-aware NB 14.1 (ratio 0.98), depth-blind NB 18.1 (ratio 1.26).
    """
    X = _controls()
    ref = _pseudobulk(X)

    def dist(M):
        return float(((_pseudobulk(M) - ref) ** 2).sum())

    rng = np.random.default_rng(4)
    d_real = np.mean([dist(X[rng.integers(0, X.shape[0], 400)]) for _ in range(20)])   # cheap: indexing only
    gen = NBGenerator(X)
    d_nb = np.mean([dist(gen(gen.ctrl_mean, 400, np.random.default_rng([4, r]))) for r in range(6)])
    assert abs(d_nb / d_real - 1) < 0.15, f"NB {d_nb:.2f} vs real {d_real:.2f} (ratio {d_nb / d_real:.3f})"

    # Sensitivity: the same generator with the depth effect switched off (H = 1, the old depth-blind mean) must FAIL
    # this tolerance, otherwise the check above could not tell the two apart.
    gen.H[:] = 1.0
    d_blind = np.mean([dist(gen(gen.ctrl_mean, 400, np.random.default_rng([4, r]))) for r in range(4)])
    assert d_blind / d_real > 1.15, f"depth-blind NB {d_blind:.2f} vs real {d_real:.2f}: the check lost its power"


def test_nb_depth_bins():
    """Deciles have equal size, H averages to 1 per gene over control cells, and generated cells copy a control
    library size (so generated depths follow the control distribution)."""
    X = _controls()
    gen = NBGenerator(X)
    sizes = np.bincount(gen.dec)
    assert sizes.size == 10 and sizes.max() - sizes.min() <= 1
    assert np.allclose((sizes[:, None] * gen.H).sum(axis=0) / sizes.sum(), 1.0)
    # deep cells are in the top decile, shallow ones in the bottom
    assert gen.lib[gen.dec == 9].min() >= gen.lib[gen.dec == 0].max()
    out = gen(gen.ctrl_mean, 2000, np.random.default_rng(6))
    q_gen = np.quantile(np.asarray(out.sum(axis=1)).ravel(), [0.1, 0.5, 0.9])
    q_ctrl = np.quantile(gen.lib, [0.1, 0.5, 0.9])
    assert np.all(np.abs(q_gen / q_ctrl - 1) < 0.08), f"depth quantiles {q_gen} vs controls {q_ctrl}"


def test_per_target_seeding_isolates_targets():
    """write_prediction seeds every target separately: two predictions that differ only in one target's profile
    produce identical cells for every other target (so variants are compared on the same cells), and two targets
    with the same profile still get different cells (each target has its own stream)."""
    X = _controls()
    genes = _genes()
    targets = ["T0", "T1", "T2", "T3"]
    cpp = 60
    for gen in (TemplateGenerator(X), TemplateGenerator(X, spread=0.5), NBGenerator(X)):
        base = np.tile(gen.ctrl_mean, (len(targets), 1))           # 'no change' for every target
        alt = base.copy()
        alt[1] = _planted(gen.ctrl_mean, np.random.default_rng(5))[2]   # only T1 changes
        with tempfile.TemporaryDirectory() as d:
            # Only 10 control rows go into the files (they are just appended; the generator was fitted above).
            pa = write_prediction(Path(d) / "a.h5ad", targets, base, gen, X[:10], genes, seed=7, cells_per_pert=cpp)
            pb = write_prediction(Path(d) / "b.h5ad", targets, alt, gen, X[:10], genes, seed=7, cells_per_pert=cpp)
            a, b = ad.read_h5ad(pa), ad.read_h5ad(pb)
        name = f"{type(gen).__name__}(spread={getattr(gen, 'spread', '-')})"

        def block(adata, t):
            return adata.X[np.flatnonzero(adata.obs["target_gene"].to_numpy() == t)]

        for t in targets:
            n_diff = (block(a, t) != block(b, t)).nnz
            if t == "T1":
                assert n_diff > 0, f"{name}: T1 changed its prediction but not its cells"
            else:
                assert n_diff == 0, f"{name}: {t} has {n_diff} different counts although only T1 changed"
        assert (block(a, "T0") != block(a, "T2")).nnz > 0, f"{name}: T0 and T2 share a random stream"


def test_gated_genes_never_all_zero():
    X = _controls()
    gen = TemplateGenerator(X)
    target = gen.ctrl_mean.copy()
    target[gen.gated[:50]] = 0.0                       # a prediction that silences 50 testable genes completely
    out = gen(target, 400, np.random.default_rng(3))
    assert (np.asarray(out[:, gen.gated].sum(axis=0)).ravel() > 0).all()


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
