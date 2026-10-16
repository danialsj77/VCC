"""Shared constants, paths and expression helpers."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

CELLS_PER_PERT = 400                 # fixed by the challenge
CONTROL = "non-targeting"            # control label in obs['target_gene']
DE_GATE_CPM = 5.0                    # the scorer tests a gene only if its control mean CPM is > 5


def project_root(start=None):
    """The project folder: the first parent (or the folder itself) that contains `vccpipe/`."""
    p = Path(start or Path.cwd()).resolve()
    for q in [p, *p.parents]:
        if (q / "vccpipe").is_dir():
            return q
    raise FileNotFoundError("could not find the project folder (no vccpipe/ above the current directory)")


def load_bundle_info(controls_dir):
    """Read a controls bundle: gene axis, target list and context labels (from manifest.json)."""
    controls_dir = Path(controls_dir)
    genes = pd.read_csv(controls_dir / "gene_names.csv")["gene_name"].astype(str).to_numpy()
    targets = pd.read_csv(controls_dir / "pert_counts.csv")["target_gene"].astype(str).tolist()
    contexts = json.loads((controls_dir / "manifest.json").read_text())["contexts"]
    return genes, targets, contexts


def load_controls(controls_dir, ctx, genes=None):
    """Raw-count control cells of one context as CSR float32, with checks on gene order and labels."""
    a = ad.read_h5ad(Path(controls_dir) / f"context_{ctx}.h5ad")
    if genes is not None and list(a.var_names) != list(genes):
        raise ValueError(f"context_{ctx}.h5ad genes are not in gene_names.csv order")
    if "context" in a.obs and not (a.obs["context"].astype(str) == ctx).all():
        raise ValueError(f"context_{ctx}.h5ad holds cells of another context")
    return sp.csr_matrix(a.X, dtype=np.float32)


def per_cell_cpm(X):
    """Counts → counts-per-million cell by cell, exactly as the scorer normalizes before its DE test."""
    X = sp.csr_matrix(X)
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    lib[lib == 0] = 1.0
    return sp.diags(1e6 / lib) @ X


def mean_cpm(X):
    """Arithmetic mean over cells of per-cell CPM (normalize first, then average): the scorer's group mean."""
    return np.asarray(per_cell_cpm(X).mean(axis=0)).ravel()


def cpm_mean_var(X):
    """Per-gene mean and variance of per-cell CPM within a group of cells."""
    c = per_cell_cpm(X)
    m = np.asarray(c.mean(axis=0)).ravel()
    m2 = np.asarray(c.multiply(c).mean(axis=0)).ravel()
    return m, np.maximum(m2 - m * m, 0.0)


def log2fc(cpm, ctrl_cpm, eps=0.1):
    """log2 fold change of mean CPM vs control, with a small CPM pseudocount for stability."""
    return np.log2((np.asarray(cpm) + eps) / (np.asarray(ctrl_cpm) + eps))


def apply_log2fc(ctrl_cpm, lfc, eps=0.1):
    """Inverse of `log2fc`: predicted mean CPM from control CPM and a log2 fold change (clipped at 0)."""
    return np.clip((np.asarray(ctrl_cpm) + eps) * np.exp2(lfc) - eps, 0.0, None)
