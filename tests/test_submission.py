"""Test vccpipe.submission.

* A chunked build on the real controls bundle (3 cells per target) passes vcc prep (needs the controls bundle and
  the vcc CLI).
* Per-target seeding, on a tiny synthetic bundle: a target's cells depend only on its own prediction, not on the
  other targets' predictions and not on how the targets are split into chunks.

Run: .venv/bin/python tests/test_submission.py   (~1 min)
"""

import json
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vccpipe.common import CONTROL, load_bundle_info, mean_cpm  # noqa: E402
from vccpipe.generator import NBGenerator, TemplateGenerator  # noqa: E402
from vccpipe.submission import build_submission, vcc_prep  # noqa: E402


def _tiny_bundle(d, rng, n_genes=300, n_targets=6, n_ctrl=200, contexts=("A", "B")):
    """A controls bundle with the real layout (gene_names.csv, pert_counts.csv, manifest.json, context_X.h5ad)
    but tiny: NB-like control counts with varying depth."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    genes = np.array([f"G{i}" for i in range(n_genes)])
    targets = [f"G{i}" for i in range(n_targets)]
    pd.DataFrame({"gene_name": genes}).to_csv(d / "gene_names.csv", index=False)
    pd.DataFrame({"target_gene": targets}).to_csv(d / "pert_counts.csv", index=False)
    (d / "manifest.json").write_text(json.dumps({"contexts": list(contexts), "pert_col": "target_gene",
                                                 "context_col": "context", "control_label": CONTROL}))
    base = rng.gamma(0.5, 1.0, n_genes) + 0.01
    for ctx in contexts:
        lib = rng.lognormal(np.log(3000), 0.4, n_ctrl)[:, None]
        X = rng.poisson(rng.gamma(2.0, base / base.sum() * lib / 2.0)).astype(np.float32)
        obs = pd.DataFrame({"target_gene": CONTROL, "context": ctx}, index=[f"{ctx}_c{i}" for i in range(n_ctrl)])
        ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(d / f"context_{ctx}.h5ad")
    return d, genes, targets, list(contexts)


def test_chunked_submission_passes_vcc_prep():
    controls = ROOT / "controls"
    genes, targets, contexts = load_bundle_info(controls)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "sub.h5ad"
        path, ctxs = build_submission(
            predict_cpm=lambda ctx, t, X: np.tile(mean_cpm(X), (len(t), 1)),
            make_generator=lambda X: TemplateGenerator(X),
            controls_dir=controls, out_h5ad=out, workdir=Path(d), cells_per_pert=3, chunk_targets=128)
        a = ad.read_h5ad(path, backed="r")
        assert a.shape == (len(contexts) * len(targets) * 3, genes.size)
        assert list(a.var_names) == list(genes)
        assert set(a.obs["context"].astype(str)) == set(contexts)
        assert (a.obs.groupby(["context", "target_gene"], observed=True).size() == 3).all()
        a.file.close()
        vcc_prep(path, controls, ctxs, dry_run=True, cells_per_pert=3)


def test_per_target_seeding_in_submission():
    """build_submission seeds every (context, target) separately: (1) changing one target's prediction in one context
    changes only that block of cells; (2) the file does not depend on chunk_targets; (3) the same target in two
    contexts, or two targets with the same prediction, get different streams."""
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        controls, genes, targets, contexts = _tiny_bundle(d / "controls", rng)
        changed = ("B", 2)                                         # the one (context, target index) that differs

        def predict(alt):
            def predict_cpm(ctx, t, X):
                p = np.tile(mean_cpm(X), (len(t), 1))
                if alt and ctx == changed[0]:
                    p[changed[1], :50] *= 3.0                       # a strong change on 50 genes
                return p
            return predict_cpm

        for name, make_gen in (("template", lambda X: TemplateGenerator(X)), ("nb", lambda X: NBGenerator(X))):
            files = {}
            for key, alt, chunk in (("base_c4", False, 4), ("alt_c4", True, 4), ("base_c6", False, 6)):
                out = d / f"{key}.h5ad"
                build_submission(predict(alt), make_gen, controls, out, d / key, cells_per_pert=20,
                                 chunk_targets=chunk, seed=3)
                files[key] = ad.read_h5ad(out)

            def block(a, ctx, j):
                o = a.obs
                return a.X[np.flatnonzero((o["context"].astype(str) == ctx).to_numpy()
                                          & (o["target_gene"].astype(str) == targets[j]).to_numpy())]

            for ctx in contexts:
                for j in range(len(targets)):
                    b0, b1, b2 = (block(files[k], ctx, j) for k in ("base_c4", "alt_c4", "base_c6"))
                    assert b0.shape[0] == 20
                    assert (b0 != b2).nnz == 0, f"{name}: {ctx}/{targets[j]} depends on chunk_targets"
                    n_diff = (b0 != b1).nnz
                    if (ctx, j) == changed:
                        assert n_diff > 0, f"{name}: the changed target's cells did not change"
                    else:
                        assert n_diff == 0, f"{name}: {ctx}/{targets[j]} changed although its prediction did not"
            a = files["base_c4"]
            assert (block(a, "A", 0) != block(a, "B", 0)).nnz > 0, f"{name}: contexts share a random stream"
            assert (block(a, "A", 0) != block(a, "A", 1)).nnz > 0, f"{name}: targets share a random stream"


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
