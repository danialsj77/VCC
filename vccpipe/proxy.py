"""Build the H1 "proxy context": Arc's 2025 H1 hESC CRISPRi data formatted like one VCC 2026 context.

Why a proxy? The final phase is blind (new cell lines, new targets, only the last entry counts), so we need a
practice exam we can take as often as we like, graded by the OFFICIAL scorer. H1 is a real cell line we never
train on during model selection, measured with the same 10x Flex chemistry, so it is the closest stand-in for an
unseen VCC context. Notebook 01 §4b explains the idea; this is the production version.

What "formatted like a VCC context" means
-----------------------------------------
* the VCC gene axis (18,533 symbols, `controls/gene_names.csv` order); H1 genes not on it are dropped, axis genes H1
  lacks are all-zero columns;
* 400 cells per kept target (`CELLS_PER_PERT`) and `n_ctrl` non-targeting control cells (18,400 in VCC);
* only targets with a real knockdown (the target's own mean CPM drops by >= `kd_min` versus controls, and the target
  is expressed in controls, > 5 CPM, so the drop is measurable). Arc keeps constructs with >= 80% median on-target
  knockdown; our mean-based rule is a slightly stricter stand-in;
* sequencing depth thinned to ~20k UMI per cell (binomial thinning of molecules, which is exactly what shallower
  sequencing of the same library looks like). H1 cells have ~54k UMI; VCC cells have ~20k.

Memory and I/O at real scale
----------------------------
The three H1 files are 7-15 GB each (~490k cells in total). We never load a whole file:
1. Rows are chosen from `obs` alone (cheap).
2. Rows are read in contiguous windows of the file (`block_rows` file rows per read). Fancy row indexing on a
   backed sparse matrix issues one HDF5 read per row (slow on compressed files), while one contiguous slice is a
   single sequential read; we then pick the wanted rows in memory.
3. Each window is mapped to the VCC axis, scored for the knockdown filter, thinned and written to a small chunk
   file on disk right away. Only the knockdown statistics stay in memory.
4. After all files are read, the chunks are filtered to the kept targets and concatenated on disk
   (`anndata.experimental.concat_on_disk`), so peak memory is about one window, not the whole proxy.
"""

import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.experimental import concat_on_disk

from .common import CELLS_PER_PERT, CONTROL, DE_GATE_CPM


# --------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------
def _gene_map(src_genes, pos, n_axis):
    """0/1 matrix (source genes x VCC genes) that moves each source column to its VCC position.

    `counts @ M` re-orders the columns onto the VCC axis in one sparse product: source genes not on the axis
    have no 1 in their row and vanish; axis genes the source lacks get an all-zero column. (If a source symbol
    occurs twice, both columns land on the same VCC gene and are summed: the molecules are still counted once each.)
    """
    src = [i for i, g in enumerate(src_genes) if g in pos]
    dst = [pos[src_genes[i]] for i in src]
    return sp.csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(len(src_genes), n_axis))


def _row_windows(rows, block_rows):
    """Split sorted file-row numbers into groups that lie in the same aligned window of `block_rows` file rows."""
    if rows.size == 0:
        return []
    return np.split(rows, np.flatnonzero(np.diff(rows // block_rows)) + 1)


def _read_rows(Xb, rows, M, block_rows):
    """Yield (row numbers, CSR counts on the VCC axis) for the wanted rows of a backed matrix, window by window.

    One contiguous slice per window (a single sequential HDF5 read), then the wanted rows are picked in memory.
    Works for backed sparse (CSR) and dense HDF5 matrices.
    """
    if getattr(Xb, "format", "csr") == "csc":
        # A CSC matrix is stored column by column: any row window would read the whole multi-GB file.
        raise ValueError("the H1 matrix is stored as CSC; convert the file to CSR once before building the proxy")
    for grp in _row_windows(np.asarray(rows, dtype=np.int64), block_rows):
        lo, hi = int(grp[0]), int(grp[-1]) + 1
        blk = sp.csr_matrix(Xb[lo:hi], dtype=np.float32)[grp - lo]
        X = (blk @ M).tocsr()
        X.sort_indices()
        if X.data.size and np.modf(X.data)[0].any():
            raise ValueError("the H1 matrix must hold raw integer counts (found fractional values)")
        yield grp, X


def _thin(X, p, rng):
    """Binomial thinning: keep each molecule with probability p (what sequencing a fraction p of reads gives)."""
    if p < 1.0 and X.nnz:
        X.data = rng.binomial(X.data.astype(np.int64), p).astype(np.float32)
        X.eliminate_zeros()
    return X


def _write_chunk(path, X, labels, names, genes):
    obs = pd.DataFrame({"target_gene": np.asarray(labels, dtype=object)}, index=pd.Index(names, dtype=object))
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes, dtype=object))).write_h5ad(path)


# --------------------------------------------------------------------------------------------------------------
# The proxy builder
# --------------------------------------------------------------------------------------------------------------
def build_h1_proxy(h5ad_paths, genes, out_h5ad, n_ctrl=18400, kd_min=0.8, median_umi=20000, seed=0,
                   targets=None, max_targets=None, block_rows=8192):
    """Write a VCC-format proxy context built from the Arc VCC-2025 H1 single-cell files; return the kept targets.

    Parameters
    ----------
    h5ad_paths : list of paths to the H1 files (obs 'target_gene' with control "non-targeting"; var index = gene
        symbols; X = raw counts). The same control cells are in every file: they are taken once, from the first
        file that has them.
    genes : the VCC gene axis (output column order).
    out_h5ad : output path. Also written: `<out>.kd.csv` with the knockdown statistics of every candidate target.
    n_ctrl : control cells to keep (min(n_ctrl, available controls)).
    kd_min : minimum knockdown, 1 - mean CPM(target in its cells) / mean CPM(target in controls).
    median_umi : thin all cells with one probability so the control median UMI becomes this (never thickens).
    seed : seeds every random choice (cells, thinning, target subsample), so the proxy is reproducible.
    targets : optional list restricting the candidate targets.
    max_targets : optional cap; a random subset of the kept targets is used if more pass the filter.
    block_rows : file rows per contiguous read (memory per read ~ block_rows x ~10k stored values x 8 bytes).

    Returns
    -------
    list[str] : the kept targets (sorted). The output has CELLS_PER_PERT cells for each of them plus the controls,
    obs['target_gene'] (control label "non-targeting") and var index = `genes`.
    """
    rng = np.random.default_rng(seed)
    genes = np.asarray(genes).astype(str)
    pos = {g: j for j, g in enumerate(genes)}
    paths = [Path(p) for p in h5ad_paths]
    out_h5ad = Path(out_h5ad)
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    allowed = None if targets is None else set(map(str, targets))

    # ---- 1) Plan from the cell labels only: which rows of which file we need ----------------------------------
    # Targets must be on the VCC axis (so the knockdown is measurable and scorable) and have >= 400 cells.
    # A target present in several files is taken from the first file where it qualifies (H1 files are disjoint
    # in targets, so this is only a safeguard).
    plans, taken, ctrl_plan = [], set(), None
    for k, path in enumerate(paths):
        a = ad.read_h5ad(path, backed="r")
        tg = a.obs["target_gene"].astype(str).to_numpy()
        src_genes = a.var_names.astype(str).to_numpy()
        a.file.close()
        if ctrl_plan is None and (tg == CONTROL).any():
            avail = np.flatnonzero(tg == CONTROL)
            n = min(int(n_ctrl), avail.size)
            if n < n_ctrl:
                print(f"  note: only {avail.size:,} control cells available (asked for {n_ctrl:,})")
            ctrl_plan = (k, np.sort(rng.choice(avail, n, replace=False)))
        rows, labels = [], []
        uniq, counts = np.unique(tg[tg != CONTROL], return_counts=True)
        for t, c in zip(uniq, counts):                     # np.unique is sorted: deterministic order
            if t in taken or t not in pos or c < CELLS_PER_PERT or (allowed is not None and t not in allowed):
                continue
            taken.add(t)
            rows.append(rng.choice(np.flatnonzero(tg == t), CELLS_PER_PERT, replace=False))
            labels += [t] * CELLS_PER_PERT
        if rows:
            rows = np.concatenate(rows)
            order = np.argsort(rows)
            plans.append((k, rows[order], np.asarray(labels, dtype=object)[order], src_genes))
        else:
            plans.append((k, np.zeros(0, np.int64), np.zeros(0, dtype=object), src_genes))
        print(f"  {path.name}: {len(set(labels))} candidate targets ({len(labels):,} cells)", flush=True)
    if ctrl_plan is None:
        raise ValueError(f"no '{CONTROL}' cells in any of {[p.name for p in paths]}")
    if not taken:
        raise ValueError("no target is on the VCC axis with >= 400 cells")

    chunk_dir = out_h5ad.parent / f"{out_h5ad.stem}_chunks"
    shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(parents=True)
    chunks = []

    # ---- 2) Controls: read, measure depth, fix the thinning probability, write ---------------------------------
    # The controls set (a) the thinning probability (their median UMI on the VCC axis) and (b) the baseline CPM of
    # every target gene for the knockdown filter. They are ~18k cells, so we hold them in memory once.
    k0, ctrl_rows = ctrl_plan
    a = ad.read_h5ad(paths[k0], backed="r")
    M = _gene_map(a.var_names.astype(str).to_numpy(), pos, genes.size)
    Xc = sp.vstack([X for _, X in _read_rows(a.X, ctrl_rows, M, block_rows)]).tocsr()
    a.file.close()
    ctrl_lib = np.asarray(Xc.sum(axis=1)).ravel()
    p = min(1.0, median_umi / max(float(np.median(ctrl_lib)), 1.0))
    # Baseline for the knockdown filter: mean per-cell CPM in controls (the scorer's definition), before thinning,
    # like the target side below (thinning does not change expected CPM, only adds noise).
    safe_lib = np.where(ctrl_lib > 0, ctrl_lib, 1.0)
    ctrl_cpm = np.asarray((sp.diags(1e6 / safe_lib) @ Xc).mean(axis=0)).ravel()
    path = chunk_dir / "c0000_controls.h5ad"
    _write_chunk(path, _thin(Xc, p, rng), [CONTROL] * ctrl_rows.size, [f"h1f{k0}_{r}" for r in ctrl_rows], genes)
    chunks.append(path)
    print(f"  controls: {ctrl_rows.size:,} cells, median UMI {np.median(ctrl_lib):,.0f} on the VCC axis "
          f"→ thinning probability {p:.3f}", flush=True)
    del Xc

    # ---- 3) Target cells: window by window, knockdown statistics + thinning + chunk files -------------------
    # For the filter we only need, per cell, the CPM of that cell's own target gene: accumulate sum and count.
    kd_sum, kd_n = {}, {}
    for k, rows, labels, src_genes in plans:
        if rows.size == 0:
            continue
        a = ad.read_h5ad(paths[k], backed="r")
        M = _gene_map(src_genes, pos, genes.size)
        lab_of = dict(zip(rows.tolist(), labels))
        for w, (grp, X) in enumerate(_read_rows(a.X, rows, M, block_rows)):
            lab = np.array([lab_of[r] for r in grp.tolist()], dtype=object)
            lib = np.asarray(X.sum(axis=1)).ravel()
            own = np.asarray(X[np.arange(grp.size), [pos[t] for t in lab]]).ravel()
            own_cpm = own / np.where(lib > 0, lib, 1.0) * 1e6
            uniq, inv = np.unique(lab.astype(str), return_inverse=True)       # targets in this window
            for t, s, c in zip(uniq, np.bincount(inv, weights=own_cpm), np.bincount(inv)):
                kd_sum[t] = kd_sum.get(t, 0.0) + float(s)
                kd_n[t] = kd_n.get(t, 0) + int(c)
            path = chunk_dir / f"f{k}_{w:04d}.h5ad"
            _write_chunk(path, _thin(X, p, rng), lab, [f"h1f{k}_{r}" for r in grp], genes)
            chunks.append(path)
        a.file.close()
        print(f"  {paths[k].name}: read and thinned {rows.size:,} target cells", flush=True)

    # ---- 4) Knockdown filter -----------------------------------------------------------------------------------
    stats = pd.DataFrame({"target": sorted(kd_sum)})
    stats["ctrl_cpm"] = [ctrl_cpm[pos[t]] for t in stats["target"]]
    stats["target_cpm"] = [kd_sum[t] / kd_n[t] for t in stats["target"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        stats["kd_frac"] = 1.0 - stats["target_cpm"] / stats["ctrl_cpm"]
    # The target must be expressed in controls (> 5 CPM, the scorer's DE gate) for a knockdown to be measurable.
    stats["passes"] = (stats["ctrl_cpm"] > DE_GATE_CPM) & (stats["kd_frac"] >= kd_min)
    keep = sorted(stats.loc[stats["passes"], "target"])
    print(f"  {len(keep)} of {len(stats)} targets pass the {kd_min:.0%} knockdown filter", flush=True)
    if max_targets is not None and len(keep) > max_targets:
        keep = sorted(rng.choice(keep, int(max_targets), replace=False).tolist())
        print(f"  random subset of {len(keep)} targets kept (max_targets)")
    if not keep:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        raise ValueError("no target passes the knockdown filter")
    stats["kept"] = stats["target"].isin(keep)
    stats.to_csv(out_h5ad.with_suffix(".kd.csv"), index=False)

    # ---- 5) Drop the cells of filtered-out targets, then concatenate the chunks on disk ------------------------
    wanted = set(keep) | {CONTROL}
    final = []
    for path in chunks:
        a = ad.read_h5ad(path)
        mask = a.obs["target_gene"].astype(str).isin(wanted).to_numpy()
        if mask.all():
            final.append(path)
        elif mask.any():
            a[mask].copy().write_h5ad(path)
            final.append(path)
        else:
            path.unlink()
        del a
    out_h5ad.unlink(missing_ok=True)
    concat_on_disk([str(p) for p in final], str(out_h5ad), axis=0, join="inner")
    shutil.rmtree(chunk_dir, ignore_errors=True)
    n_cells = ctrl_rows.size + len(keep) * CELLS_PER_PERT
    print(f"  wrote {out_h5ad.name}: {len(keep)} targets x {CELLS_PER_PERT} cells + {ctrl_rows.size:,} controls "
          f"= {n_cells:,} cells on {genes.size:,} genes", flush=True)
    return keep
