"""Public perturbation screens as per-target response matrices on the VCC gene axis.

A *source* is one public CRISPRi screen summarized per knocked-down target gene ("target"):

* the mean expression (CPM) of the target's cells and of the screen's non-targeting control cells, gene by gene;
* the standard error of the target's log2 fold change, so the model can tell a real response from sampling noise
  (z = lfc / se; the effect model only trusts genes with a large |z|, see METHOD.md 3b/3d).

Two loaders build them (interfaces fixed in docs/DESIGN_v1.md):

* `load_replogle_bulk`: Replogle et al. 2022 "bulk" files. Each row is already a pseudobulk (mean counts per cell of
  one sgRNA's cells), so the cell-to-cell spread is gone. The noise is instead estimated from how much the many
  non-targeting guide rows scatter around their pool: they are pseudobulks of cells that received *no* effective
  perturbation, so their scatter is pure noise.
* `h1_pseudobulk`: Arc's 2025 H1 hESC single-cell files (7-15 GB each). Streamed from disk one target at a time (the
  files never sit in memory); the noise comes from the cell-to-cell spread (delta method).

Conventions shared by everything in the model:

* **One gene axis.** Every matrix has the VCC gene axis as columns (controls/gene_names.csv, 18,533 symbols). A
  source gene that is not on the axis is dropped; an axis gene the source did not measure has `measured = False`,
  CPM 0 and an infinite standard error, so it can never look like a confident change. `ResponseSet.reindex_genes`
  moves a built set onto another axis (e.g. a final-phase bundle with a reordered or slightly different gene list)
  by symbol, with the same convention for genes the source lacks.
* **CPM means, like the scorer.** The scorer compares arithmetic means of per-cell CPM (normalize each cell, then
  average; notes/VCC2026_brief.md §4.1). `h1_pseudobulk` computes exactly that. Replogle's bulk rows only carry
  mean counts, so there CPM is computed per row (a ratio of means); fold changes are nearly identical.
* **log2 fold changes with a pseudocount.** lfc = log2((target_cpm + EPS_CPM) / (ctrl_cpm + EPS_CPM)). EPS_CPM keeps
  genes near zero from producing huge, meaningless ratios (a gene going from 0.01 to 0.05 CPM is not a 5x change
  anyone can measure).
"""

import io
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import anndata.abc
import anndata.io
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import CONTROL

EPS_CPM = 0.5            # pseudocount (CPM) for log2 fold changes everywhere in the model
CHUNK_CELLS = 2_000      # at most this many single cells are held in memory at once by h1_pseudobulk. Measured on
                         # an H1-sized synthetic file (~9.3k stored counts per cell, rows shuffled): peak RSS
                         # ~1.5-2 GB at 2,000-2,500 cells vs ~3.2 GB at 5,000, and no slower.
_BLOCK_TARGETS = 1_024   # Replogle targets are finished in blocks of this many rows (bounds the temporaries)
_LN2 = np.log(2.0)
_META_COLUMNS = ["n_cells", "kd_remaining", "phenotype_p"]
_FORMAT_VERSION = 1      # bump if the .npz layout changes


# ---------------------------------------------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class ResponseSet:
    """One screen's per-target responses on the VCC gene axis (see the module docstring for conventions).

    Rows are targets, columns are the VCC genes. `meta` holds one row per target:
    n_cells (cells pooled for the target), kd_remaining (fraction of the target's own expression left, nan if
    unknown) and phenotype_p (the screen's "does this knockdown change anything" p-value, nan if unknown).
    """

    name: str
    genes: np.ndarray          # (G,) str, the VCC gene axis
    measured: np.ndarray       # (G,) bool
    targets: list              # (T,) unique target symbols, one per row
    target_cpm: np.ndarray     # (T, G) float32 mean CPM per target (0 where not measured)
    ctrl_cpm: np.ndarray       # (G,) float64 control mean CPM (0 where not measured)
    se_lfc: np.ndarray         # (T, G) float32 standard error of the log2 fold change (inf where not measured)
    meta: pd.DataFrame         # index = targets; columns n_cells, kd_remaining, phenotype_p

    def __post_init__(self):
        # Normalize dtypes once here so every consumer can rely on them (and catch shape mistakes early, at
        # construction, instead of as a confusing broadcast error deep inside the model).
        self.name = str(self.name)
        self.genes = np.asarray(self.genes).astype(str)
        self.measured = np.asarray(self.measured, dtype=bool)
        self.targets = [str(t) for t in self.targets]
        G, T = self.genes.size, len(self.targets)
        self.target_cpm = self._full(self.target_cpm, np.float32, (T, G), "target_cpm")
        self.se_lfc = self._full(self.se_lfc, np.float32, (T, G), "se_lfc")
        self.ctrl_cpm = np.asarray(self.ctrl_cpm, dtype=np.float64)
        if self.measured.shape != (G,) or self.ctrl_cpm.shape != (G,):
            raise ValueError(f"{self.name}: measured/ctrl_cpm must have shape ({G},)")
        if len(set(self.targets)) != T:
            raise ValueError(f"{self.name}: targets must be unique")
        if [str(i) for i in self.meta.index] != self.targets:
            raise ValueError(f"{self.name}: meta must be indexed by the targets, in row order")
        missing = [c for c in _META_COLUMNS if c not in self.meta.columns]
        if missing:
            raise ValueError(f"{self.name}: meta lacks columns {missing}")
        # One index type and name whatever the caller built (also for an empty set), so saved and loaded sets
        # compare equal. set_axis returns a new frame: the caller's DataFrame is not modified.
        self.meta = self.meta.set_axis(pd.Index(self.targets, dtype=str, name="target"), axis=0)
        n = pd.to_numeric(self.meta["n_cells"], errors="coerce")
        self.meta = self.meta.astype({"kd_remaining": np.float64, "phenotype_p": np.float64})
        self.meta["n_cells"] = n.astype(np.int64) if np.isfinite(n.to_numpy(dtype=np.float64)).all() else n

    def _full(self, a, dtype, shape, what):
        """a as a (T, G) array of dtype. No copy when it already is one; a broadcastable array (e.g. one se row
        (1, G) shared by all targets, handy in synthetic sets) is expanded to a full writable (T, G) array; an empty
        set may arrive as shape (0,)."""
        a = np.asarray(a, dtype=dtype)
        if a.shape == shape:
            return a
        if a.size == 0 and shape[0] == 0:
            return a.reshape(shape)
        try:
            return np.array(np.broadcast_to(a, shape))
        except ValueError:
            raise ValueError(f"{self.name}: {what} must have shape {shape}, got {a.shape}") from None

    # -- derived quantities --------------------------------------------------------------------------------------

    def lfc(self, eps=EPS_CPM):
        """(T, G) float32 log2((target_cpm + eps) / (ctrl_cpm + eps)); 0 where the gene was not measured.

        Computed in place on one (T, G) float32 array: for the K562 genome-wide screen that array alone is ~0.7 GB,
        so no float64 or extra temporaries of that size are created.
        """
        out = self.target_cpm + np.float32(eps)                    # new float32 array (T, G)
        np.log2(out, out=out)
        out -= np.log2(self.ctrl_cpm + eps).astype(np.float32)     # broadcast over rows
        out[:, ~self.measured] = 0.0
        return out

    def z(self, eps=EPS_CPM):
        """(T, G) float32 lfc / se_lfc: how many standard errors the change is from 0 (0 where not measured).

        se_lfc was computed with EPS_CPM; for another eps this is an approximation (fine for eps of similar size).
        """
        out = self.lfc(eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            out /= self.se_lfc                                     # se = inf (not measured) gives exactly 0
        out[~np.isfinite(out)] = 0.0                               # se = 0 would be a 0/0: no evidence, so 0
        return out

    # -- selection -------------------------------------------------------------------------------------------------

    def subset(self, targets):
        """A new ResponseSet with the rows of the given targets that this set has, in the order given.

        Targets the source did not measure are skipped silently (callers check coverage with `meta.index`).
        """
        row = {t: i for i, t in enumerate(self.targets)}
        keep = list(dict.fromkeys(str(t) for t in targets if str(t) in row))     # present, de-duplicated, ordered
        idx = np.array([row[t] for t in keep], dtype=np.int64)
        return ResponseSet(name=self.name, genes=self.genes.copy(), measured=self.measured.copy(), targets=keep,
                           target_cpm=self.target_cpm[idx], ctrl_cpm=self.ctrl_cpm.copy(),
                           se_lfc=self.se_lfc[idx], meta=self.meta.iloc[idx].copy())

    def reindex_genes(self, genes):
        """A new ResponseSet on another gene axis `genes`, with the columns matched by gene symbol.

        Why: the sources are built once (01_build_sources.py) on the A/B/C bundle's axis. If a later controls bundle
        (the final D/E/F phase) lists its genes in another order, or adds or drops a few, the sources can be moved
        onto the new axis in seconds instead of being rebuilt (which takes hours and would overwrite the files the
        validation used). The responses themselves do not depend on the axis: every value belongs to one gene.

        For each gene of the new axis:
          * the source has the gene  -> its column is copied unchanged (measured flag, CPMs, standard errors);
          * the source lacks the gene -> measured = False, target and control CPM 0, standard error inf: exactly
            how the loaders mark an axis gene a screen did not measure, so it can never look like a confident change.
        Genes of the old axis that are not on the new one are dropped. Targets, their order, `meta` and `name` are
        unchanged (a target whose own gene left the axis keeps its row; the model looks targets up by symbol).

        Returns a copy even when the axis is identical, so the caller may modify the result freely. Prints nothing:
        callers that want to report how many genes were added or dropped compare the two axes themselves.
        Memory: the two (T, G_new) result arrays plus one block of rows at a time (no full-size temporaries),
        i.e. ~1.5 GB for the K562 genome-wide screen on an 18.5k-gene axis.
        """
        new = _check_axis(genes)                                  # 1-D, unique symbols (else ValueError)
        old = pd.Index(self.genes)
        if not old.is_unique:
            # Every loader builds sets on a unique axis; a hand-built set with a repeated symbol has no well-defined
            # "column of gene g", so refuse instead of silently picking one of the copies.
            raise ValueError(f"{self.name}: cannot reindex a set whose own gene axis repeats a symbol")
        pos = old.get_indexer(pd.Index(new))                      # old column of each new gene, -1 if absent
        # A new column is measured only if the gene exists in the source AND the source measured it there.
        measured = np.zeros(new.size, dtype=bool)
        has = pos >= 0
        measured[has] = self.measured[pos[has]]
        dst = np.flatnonzero(measured)                            # new columns that receive data
        src = pos[dst]                                            # ... and the old columns they come from

        T, G = len(self.targets), new.size
        target_cpm = np.zeros((T, G), dtype=np.float32)          # "not measured" defaults: CPM 0 ...
        se = np.full((T, G), np.inf, dtype=np.float32)            # ... and an infinite standard error
        # Copy the measured columns block by block: `self.target_cpm[:, src]` on all rows at once would build a
        # third (T, n_measured) array (~0.3 GB for K562) before scattering it into place.
        for b0 in range(0, T, _BLOCK_TARGETS):
            b = slice(b0, min(T, b0 + _BLOCK_TARGETS))
            target_cpm[b, dst] = self.target_cpm[b, src]
            se[b, dst] = self.se_lfc[b, src]
        ctrl_cpm = np.zeros(G, dtype=np.float64)
        ctrl_cpm[dst] = self.ctrl_cpm[src]
        return ResponseSet(name=self.name, genes=new, measured=measured, targets=list(self.targets),
                           target_cpm=target_cpm, ctrl_cpm=ctrl_cpm, se_lfc=se, meta=self.meta.copy())

    # -- persistence -------------------------------------------------------------------------------------------------

    def save(self, path):
        """Write one .npz holding everything (arrays, gene axis, targets, and meta as a CSV string).

        Only the measured columns of the (T, G) matrices are stored (the rest is 0 / inf by definition), which
        halves the K562 file without the slow zlib compression of np.savez_compressed. The file is written to a
        temporary name and renamed, so an interrupted save never leaves a truncated file behind.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        m = self.measured
        arrays = {
            "format_version": np.array(_FORMAT_VERSION),
            "name": np.array(self.name),
            "genes": self.genes.astype(str),
            "measured": m,
            "targets": np.array(self.targets, dtype=str),
            "target_cpm_measured": np.ascontiguousarray(self.target_cpm[:, m]),
            "se_lfc_measured": np.ascontiguousarray(self.se_lfc[:, m]),
            "ctrl_cpm": self.ctrl_cpm,
            "meta_csv": np.array(self.meta.to_csv(index_label="target")),
        }
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:                    # a file handle stops numpy from appending ".npz" to the name
            np.savez(fh, **arrays)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path):
        """Read a file written by `save`."""
        with np.load(path, allow_pickle=False) as f:
            version = int(f["format_version"])
            if version != _FORMAT_VERSION:
                raise ValueError(f"{path}: ResponseSet format {version}, this code reads {_FORMAT_VERSION}")
            genes = f["genes"].astype(str)
            measured = f["measured"].astype(bool)
            targets = [str(t) for t in f["targets"]]
            T, G = len(targets), genes.size
            target_cpm = np.zeros((T, G), dtype=np.float32)
            target_cpm[:, measured] = f["target_cpm_measured"]
            se = np.full((T, G), np.inf, dtype=np.float32)
            se[:, measured] = f["se_lfc_measured"]
            ctrl_cpm = f["ctrl_cpm"].astype(np.float64)
            name = str(f["name"])
            meta_csv = str(f["meta_csv"])
        # keep_default_na=False: a gene symbol such as "NA" must stay a string. pandas writes NaN as an empty
        # field, so only empty fields are read back as NaN.
        meta = pd.read_csv(io.StringIO(meta_csv), index_col=0, keep_default_na=False, na_values=[""])
        meta.index = targets                                   # the exact strings (__post_init__ sets the type)
        return cls(name=name, genes=genes, measured=measured, targets=targets, target_cpm=target_cpm,
                   ctrl_cpm=ctrl_cpm, se_lfc=se, meta=meta)

    def __repr__(self):
        return (f"ResponseSet({self.name!r}: {len(self.targets)} targets x {self.genes.size} genes, "
                f"{int(self.measured.sum())} measured)")


# ---------------------------------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------------------------------

def _check_axis(genes):
    genes = np.asarray(genes).astype(str)
    if genes.ndim != 1 or not pd.Index(genes).is_unique:
        raise ValueError("the gene axis must be a 1-D list of unique symbols")
    return genes


def _axis_positions(file_genes, genes):
    """Position of each of a file's genes on the axis (-1 if the symbol is not on it). Duplicated file symbols map
    to the same axis column; callers sum them (the duplicates are the same gene under two ids)."""
    return pd.Index(genes).get_indexer(pd.Index(np.asarray(file_genes).astype(str)))


def _to_axis(A, pos, G, dtype):
    """Columns of A (last axis = the file's genes) moved onto the axis; duplicated symbols summed, others dropped."""
    src = np.flatnonzero(pos >= 0)
    dst = pos[src]
    out = np.zeros(A.shape[:-1] + (G,), dtype=dtype)
    _, first = np.unique(dst, return_index=True)
    out[..., dst[first]] = A[..., src[first]]
    for k in np.setdiff1d(np.arange(src.size), first):        # the few duplicated symbols (2 in Replogle K562)
        out[..., dst[k]] += A[..., src[k]]
    return out


def _poisson_var_log2(cpm, molecules, eps):
    """Variance of log2(cpm + eps) if cpm were measured from `molecules` counted molecules with Poisson noise only.

    k molecules out of M have variance k, and cpm = 1e6 k / M, so Var(cpm) = cpm * 1e6 / M; the delta method turns
    that into the variance of the log. This is a lower bound on the real noise (biology only adds to it).
    One molecule is added to the observed count (cpm + 1e6/M): when a strongly knocked-down gene happens to show
    0 molecules, the plain formula would claim zero variance and a huge |z|, although 0 is just what a level of
    ~1 molecule gives a third of the time. With many molecules the extra one changes nothing.
    """
    unit = 1e6 / molecules                                     # CPM of a single molecule
    return (cpm + unit) * unit / ((cpm + eps) ** 2 * _LN2 ** 2)


# ---------------------------------------------------------------------------------------------------------------
# Replogle 2022 bulk files
# ---------------------------------------------------------------------------------------------------------------

# A RefSeq accession: two capital letters, an underscore, digits and an optional version, e.g. NM_001159524 or
# NR_024540.1. Several may be listed with commas, as the files do for Ensembl transcripts ("ENST...,ENST...").
_REFSEQ = r"[A-Z]{2}_\d+(?:\.\d+)?"
# "<GENE>_<RefSeq accession(s)>": group 1 is the gene. `fullmatch` with a greedy (.+) still finds the right split,
# because the tail must be a complete accession list (the regex backtracks to the last split where it is).
_GENE_THEN_REFSEQ = re.compile(rf"(.+)_{_REFSEQ}(?:,{_REFSEQ})*")


def _replogle_target_label(s):
    """Target symbol of one obs-index entry "<id>_<GENE>_<transcript>_<ENSG>" (see `_replogle_target_labels`)."""
    fields = s.split("_")
    if len(fields) < 4:
        raise ValueError(f"unexpected Replogle row name {s!r} (want '<id>_<GENE>_<transcript>_<ENSG>')")
    # The first field (row id, digits) and the last (an Ensembl gene id, "non-targeting", or "nan" in a few rows)
    # never contain an underscore, so dropping one field at each end leaves "<GENE>_<transcript>".
    body = "_".join(fields[1:-1])
    m = _GENE_THEN_REFSEQ.fullmatch(body)
    if m:
        gene = m.group(1)                  # the transcript is a RefSeq id, which itself contains one underscore
    else:
        gene, _, transcript = body.rpartition("_")   # every other transcript form is one token without "_"
        if not transcript:
            raise ValueError(f"unexpected Replogle row name {s!r}: empty transcript field")
    if not gene:
        raise ValueError(f"unexpected Replogle row name {s!r}: empty gene symbol")
    return gene


def _replogle_target_labels(index):
    """Target symbol of each row from the obs index "<id>_<GENE>_<transcript>_<ENSG>".

    Controls read "<id>_non-targeting_non-targeting_non-targeting" and come out as "non-targeting".

    The transcript field is what the guide was designed against. In the real files (K562 genome-wide, K562
    essential, RPE1; checked Oct 2026) it takes these forms:
      * P1, P2, P1P2                    (promoter-level constructs, by far the most common)
      * ENST00000375793.2, or several comma-separated ENST ids
      * non-targeting                   (control rows)
      * NM_001159524                    (a RefSeq id: one row of K562 genome-wide, "10598_ZNF735_NM_001159524_
                                          ENSG00000223614"). It contains an underscore itself.
    The old rule "the symbol sits between the first underscore and the last two" read that row as the target
    "ZNF735_NM", a symbol not on the VCC axis, so ZNF735 (which is on the axis) silently lost its K562 response.
    Rule now: drop the id (first field) and the Ensembl gene id (last field); if what is left ends with a RefSeq
    accession, the symbol is what precedes it, otherwise the transcript is the last underscore-free token. A symbol
    that itself contained an underscore would still parse correctly (HGNC symbols do not, but nothing breaks).
    """
    return np.array([_replogle_target_label(s) for s in map(str, index)], dtype=object)


def _numeric_obs(obs, col):
    if col not in obs.columns:
        return np.full(len(obs), np.nan)
    return pd.to_numeric(obs[col], errors="coerce").to_numpy(dtype=np.float64)


def load_replogle_bulk(path, genes, name):
    """Load a Replogle 2022 `*_raw_bulk_01.h5ad` as a ResponseSet on the axis `genes`.

    What the file holds: X (dense) = mean raw counts per cell, one row per sgRNA construct (a "guide row"), over
    ~8,000 expressed genes; obs num_cells_filtered (cells behind each row), fold_expr (fraction of the target's
    expression left), energy_test_p_value (is the row's transcriptome different from controls at all).

    Steps:
    1. CPM per row over all the row's genes; rows without a usable cell count (NaN for some control rows) are
       skipped, since they cannot be weighted.
    2. Targets: all rows of one symbol (e.g. separate P1/P2 promoter constructs) are pooled, weighted by cells.
    3. Controls: the non-targeting rows pooled the same way.
    4. Noise. A guide row j of n_j cells is a pseudobulk, so its log2FC vs the control pool has variance
       ~ sigma2_g / n_j. The non-targeting rows have no true effect, so their scatter estimates sigma2_g:
           sigma2_g = mean_j( n_j * l_jg^2 / (1 - n_j/N) ),  l_jg = log2FC of control row j vs the control pool.
       (1 - n_j/N) corrects for row j being part of the pool it is compared with (it pulls the pool towards
       itself, which shrinks l_jg by exactly that factor in variance.) Then
           se(t,g)^2 = max(sigma2_g / n_t, Poisson_t) + max(sigma2_g / N, Poisson_ctrl),
       where Poisson_* is the variance from counting molecules alone (`_poisson_var_log2`, at the target's own
       level). sigma2_g is measured at the *control* level; this floor matters where the target sits far below or
       above it: a gene at ~0 CPM in every control but with a few counts in a target (the controls' scatter is ~0
       there: one stray molecule would get z = infinity), or a strongly knocked-down gene (few molecules left, so
       its log is much noisier than in the controls). The (1 - n_j/N) factor and this floor are the only
       additions to the DESIGN_v1 formula; with many control rows and well-expressed genes both change nothing.
    meta: n_cells = pooled cells, kd_remaining = cell-weighted mean fold_expr, phenotype_p = min energy p-value.
    """
    t0 = time.time()
    genes = _check_axis(genes)
    G, eps = genes.size, EPS_CPM
    a = ad.read_h5ad(path)                    # ~11k x 8k dense float32 = 0.35 GB for K562 genome-wide
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    X = np.asarray(X, dtype=np.float32)
    obs, var = a.obs, a.var
    del a

    labels = _replogle_target_labels(obs.index)
    n_rows = _numeric_obs(obs, "num_cells_filtered")
    fold = _numeric_obs(obs, "fold_expr")
    p_rows = _numeric_obs(obs, "energy_test_p_value")
    lib = X.sum(axis=1, dtype=np.float64)                    # mean UMI per cell of each row (over all its genes)
    usable = np.isfinite(n_rows) & (n_rows > 0) & (lib > 0)

    # 1) CPM per row, in place (no second 0.35 GB copy).
    X *= (1e6 / np.where(lib > 0, lib, 1.0)).astype(np.float32)[:, None]
    file_genes = var["gene_name"].astype(str).to_numpy() if "gene_name" in var.columns else var.index.astype(str)
    pos = _axis_positions(file_genes, genes)
    measured = np.zeros(G, dtype=bool)
    measured[pos[pos >= 0]] = True

    # 3) + 4) Controls and the per-gene noise sigma2_g (all on the axis; there are only a few hundred control rows).
    is_ctrl = labels == CONTROL
    c_rows = np.flatnonzero(is_ctrl & usable)
    if c_rows.size < 2:
        raise ValueError(f"{path}: need >= 2 non-targeting rows with a cell count, found {c_rows.size}")
    n_c = n_rows[c_rows]
    N = n_c.sum()
    C = _to_axis(X[c_rows], pos, G, np.float64)              # (Rc, G) CPM of each control row
    ctrl_cpm = (n_c @ C) / N
    l_c = np.log2((C + eps) / (ctrl_cpm + eps))              # 0 in unmeasured columns (0 vs 0)
    sigma2 = ((n_c / (1.0 - n_c / N))[:, None] * l_c ** 2).mean(axis=0)
    del C, l_c
    molecules_c = n_c @ lib[c_rows]                          # cells x mean UMI per cell = molecules in the pool
    var_ctrl = np.maximum(sigma2 / N, _poisson_var_log2(ctrl_cpm, molecules_c, eps))

    # 2) Targets: rows grouped by symbol, in order of first appearance in the file.
    t_rows = np.flatnonzero(~is_ctrl & usable)
    t_labels = labels[t_rows]
    targets = [str(t) for t in pd.unique(t_labels)]
    T = len(targets)
    t_of_row = pd.Index(targets).get_indexer(t_labels)
    n_cells = np.bincount(t_of_row, weights=n_rows[t_rows], minlength=T)
    molecules_t = np.bincount(t_of_row, weights=n_rows[t_rows] * lib[t_rows], minlength=T)
    # Pooling matrix P (T x all rows): weight n_j / n_t on each of the target's rows, so P @ X = pooled CPM.
    P = sp.csr_matrix((n_rows[t_rows] / n_cells[t_of_row], (t_of_row, t_rows)), shape=(T, X.shape[0]),
                      dtype=np.float32)

    target_cpm = np.zeros((T, G), dtype=np.float32)
    se = np.full((T, G), np.inf, dtype=np.float32)
    for b0 in range(0, T, _BLOCK_TARGETS):
        b = slice(b0, min(T, b0 + _BLOCK_TARGETS))
        x = _to_axis(np.asarray(P[b] @ X), pos, G, np.float64)        # (block, G) pooled CPM on the axis
        target_cpm[b] = x
        var_t = np.maximum(sigma2[None, :] / n_cells[b, None],
                           _poisson_var_log2(x, molecules_t[b, None], eps))
        s = np.sqrt(var_t + var_ctrl[None, :])
        s[:, ~measured] = np.inf
        se[b] = s
    del X, P

    # meta. kd_remaining: cell-weighted mean of the rows' fold_expr (rows with a known value only).
    has_fold = np.isfinite(fold[t_rows])
    w_fold = np.bincount(t_of_row[has_fold], weights=n_rows[t_rows][has_fold], minlength=T)
    s_fold = np.bincount(t_of_row[has_fold], weights=(n_rows * fold)[t_rows][has_fold], minlength=T)
    with np.errstate(invalid="ignore", divide="ignore"):
        kd = np.where(w_fold > 0, s_fold / w_fold, np.nan)
    p_min = pd.Series(p_rows[t_rows]).groupby(t_of_row).min().reindex(range(T)).to_numpy(dtype=np.float64)
    meta = pd.DataFrame({"n_cells": np.rint(n_cells).astype(np.int64), "kd_remaining": kd, "phenotype_p": p_min},
                        index=pd.Index(targets, name="target"))

    ctrl_cpm[~measured] = 0.0
    n_skipped = int((~usable).sum())
    print(f"  {name}: {T:,} targets from {t_rows.size:,} guide rows, {c_rows.size} control rows "
          f"({N:,.0f} cells); {n_skipped} rows skipped (no cell count); {int(measured.sum()):,} of {G:,} axis genes "
          f"measured ({time.time() - t0:.0f} s)", flush=True)
    return ResponseSet(name=name, genes=genes, measured=measured, targets=targets, target_cpm=target_cpm,
                       ctrl_cpm=ctrl_cpm, se_lfc=se, meta=meta)


# ---------------------------------------------------------------------------------------------------------------
# Arc 2025 H1 single-cell files
# ---------------------------------------------------------------------------------------------------------------

def _h5ad_labels(path):
    """(target_gene per cell, barcode per cell, gene symbols) of an .h5ad, reading only its obs and var.

    Like ad.read_h5ad(path, backed="r") but lighter: backed mode still loads every other element (layers, obsm,
    obsp, uns) into memory, which for a 15 GB file with, say, a second counts layer would defeat the purpose.
    """
    with h5py.File(path, "r") as f:
        if "encoding-type" in f.attrs:                        # any h5ad written by anndata >= 0.7
            obs, var = anndata.io.read_elem(f["obs"]), anndata.io.read_elem(f["var"])
        else:                                                 # very old layout: let anndata handle it
            obs = var = None
    if obs is None:
        a = ad.read_h5ad(path, backed="r")
        obs, var = a.obs, a.var
        a.file.close()
    return (obs["target_gene"].astype(str).to_numpy(), obs.index.astype(str).to_numpy(),
            var.index.astype(str).to_numpy())


def _backed_X(f):
    """The X of an open h5py.File as a backed matrix (nothing is read yet): CSR dataset, or h5py dataset if dense."""
    g = f["X"]
    return anndata.io.sparse_dataset(g) if isinstance(g, h5py.Group) else g


def _read_rows(X, rows):
    """Rows (sorted, unique) of a backed X as an in-memory CSR matrix, reading only those rows from disk."""
    if isinstance(X, anndata.abc.CSRDataset):
        mask = np.zeros(X.shape[0], dtype=bool)               # a boolean mask lets anndata read contiguous runs
        mask[rows] = True                                     # of rows as one slice instead of row by row
        return sp.csr_matrix(X[mask])                         # no dtype argument: that would copy the data
    if isinstance(X, anndata.abc.CSCDataset):
        raise NotImplementedError("X is stored column-wise (CSC); reading cells would load the whole file")
    return sp.csr_matrix(np.asarray(X[rows, :], dtype=np.float32))    # dense on disk (h5py needs sorted rows)


def _cpm_sums(X, rows, pos_ext, G, has_dup):
    """Read rows of a backed X and return the per-gene sum and sum of squares of their per-cell CPM, on the axis.

    Each cell's CPM uses its total over *all* the file's genes (as the scorer does), including genes that are not
    on the axis. Those are kept in an extra dummy column G so they count in the total, then dropped. Works on the
    sparse entries only (never a dense cells x genes array): peak ~20 bytes per stored count, i.e. ~1 GB for
    5,000 H1 cells (~9-10k stored counts per cell).
    """
    block = _read_rows(X, rows)
    # int32 row pointers when they fit (always, for a chunk): otherwise scipy upcasts the column indices to int64,
    # doubling their memory.
    indptr = block.indptr.astype(np.int32) if block.nnz < 2**31 - 1 else block.indptr
    Y = sp.csr_matrix((block.data.astype(np.float64), pos_ext[block.indices], indptr),
                      shape=(block.shape[0], G + 1))
    del block                                                  # the float64 copy above replaces it
    lib = Y @ np.ones(G + 1)                                 # counts per cell over all the file's genes
    scale = np.zeros_like(lib)
    np.divide(1e6, lib, out=scale, where=lib > 0)              # an empty cell contributes CPM 0 (as common.py)
    if has_dup:
        Y.sum_duplicates()                                     # duplicated symbols: add them up inside each cell
    s1 = Y.T @ scale                                           # sum_i x_ig * 1e6/lib_i
    Y.data **= 2
    s2 = Y.T @ (scale * scale)                                 # sum_i (x_ig * 1e6/lib_i)^2
    return s1[:G], s2[:G]


def _chunks(rows, size):
    for i in range(0, rows.size, size):
        yield rows[i:i + size]


def h1_pseudobulk(h5ad_paths, genes, name="h1_2025", min_cells=50, max_cells_per_target=None, seed=0, *,
                  chunk_cells=CHUNK_CELLS, verbose=True):
    """Per-target pseudobulk of Arc's VCC-2025 H1 single-cell files, streamed from disk.

    Files: obs target_gene (controls "non-targeting"), var index = gene symbols, X = raw counts (CSR). The same
    control cells appear in every file; they are recognized by their barcode (obs name) and used once (a file's
    control cells with new barcodes are added; the log shows how many were shared).

    Memory: X stays on disk (anndata's backed CSR dataset; only obs, var and the needed rows of X are ever read,
    see `_h5ad_labels`) and is read one target at a time; at most `chunk_cells` cells are in memory at once
    (controls, and any target with more cells, are read in chunks). What accumulates is two (T, G) float64 arrays
    of sums (~90 MB for 300 targets).

    Per target t and gene g, over the target's n_t cells (x = per-cell CPM):
        target_cpm = mean(x)       (the scorer's definition: normalize each cell, then average)
        v_t        = sample variance of x
    and the same over the n_c control cells (ctrl_cpm = c, v_c). The delta method gives the log2FC's error:
        se = (1/ln2) * sqrt( v_t / (n_t (target_cpm + eps)^2) + v_c / (n_c (c + eps)^2) ).
    meta: n_cells; kd_remaining = target's own mean CPM / control mean CPM (nan if its gene is not measured or
    not expressed in controls); phenotype_p = nan (the files carry no such test).

    min_cells: targets with fewer cells are dropped. max_cells_per_target: if set, a random subset of that many
    cells (seeded) is used, e.g. to match the 400 cells per target of the challenge. chunk_cells and verbose are
    extensions of the DESIGN_v1 signature (keyword-only, defaults keep its behaviour).
    """
    t_start = time.time()
    genes = _check_axis(genes)
    G, eps = genes.size, EPS_CPM
    paths = [Path(h5ad_paths)] if isinstance(h5ad_paths, (str, Path)) else [Path(p) for p in h5ad_paths]
    if min_cells < 2:
        raise ValueError("min_cells must be >= 2 (a variance needs two cells)")
    rng = np.random.default_rng(seed)
    log = print if verbose else (lambda *a, **k: None)

    # ---- Pass 1: labels only. Decide which rows of which file belong to which target, and which control cells
    #      are new (not already seen in an earlier file).
    plans, cells, seen_ctrl, measured = [], {}, set(), np.ones(G, dtype=bool)
    for k, path in enumerate(paths):
        tg, barcodes, file_genes = _h5ad_labels(path)
        pos = _axis_positions(file_genes, genes)
        in_file = np.zeros(G, dtype=bool)
        in_file[pos[pos >= 0]] = True
        measured &= in_file                       # a gene counts as measured only if every file has it
        c_all = np.flatnonzero(tg == CONTROL)
        new = np.fromiter((b not in seen_ctrl for b in barcodes[c_all]), dtype=bool, count=c_all.size)
        seen_ctrl.update(barcodes[c_all[new]])
        dst = pos[pos >= 0]
        plans.append({"path": path, "ctrl_rows": c_all[new], "n_ctrl_skipped": int((~new).sum()),
                      "pos_ext": np.where(pos >= 0, pos, G).astype(np.int32),
                      "has_dup": dst.size != np.unique(dst).size, "targets": {}})
        nc = np.flatnonzero(tg != CONTROL)
        codes, uniq = pd.factorize(tg[nc])                     # order of first appearance in the file
        order = np.argsort(codes, kind="stable")
        bounds = np.cumsum(np.bincount(codes, minlength=len(uniq)))[:-1]
        for t, rows in zip(uniq, np.split(nc[order], bounds)):
            cells.setdefault(str(t), []).append((k, rows))     # a target may (rarely) span several files

    # Keep targets with enough cells; optionally subsample to max_cells_per_target (seeded, reproducible).
    targets = []
    for t, parts in cells.items():
        total = sum(r.size for _, r in parts)
        if total < min_cells:
            continue
        if max_cells_per_target is not None and total > max_cells_per_target:
            pick = np.zeros(total, dtype=bool)
            pick[rng.choice(total, int(max_cells_per_target), replace=False)] = True
            sizes = np.cumsum([0] + [r.size for _, r in parts])
            parts = [(k, r[pick[s0:s1]]) for (k, r), s0, s1 in zip(parts, sizes[:-1], sizes[1:])]
        ti = len(targets)
        targets.append(t)
        for k, r in parts:
            if r.size:
                plans[k]["targets"][ti] = np.sort(r)
    if not targets:
        raise ValueError(f"no target has >= {min_cells} cells in {[p.name for p in paths]}")
    n_ctrl = sum(p["ctrl_rows"].size for p in plans)
    if n_ctrl < 2:
        raise ValueError("need >= 2 control cells")
    T = len(targets)
    log(f"  {name}: {T} targets with >= {min_cells} cells (of {len(cells)}), {n_ctrl:,} control cells", flush=True)

    # ---- Pass 2: stream the cells and accumulate per-gene sums of CPM and CPM^2.
    S1 = np.zeros((T, G))
    S2 = np.zeros((T, G))
    n_t = np.zeros(T)
    c1, c2 = np.zeros(G), np.zeros(G)
    for plan in plans:
        t0 = time.time()
        with h5py.File(plan["path"], "r") as f:                 # X stays on disk; rows are read on demand
            X = _backed_X(f)
            for rows in _chunks(plan["ctrl_rows"], chunk_cells):
                s1, s2 = _cpm_sums(X, rows, plan["pos_ext"], G, plan["has_dup"])
                c1 += s1
                c2 += s2
            for ti, rows in plan["targets"].items():
                for chunk in _chunks(rows, chunk_cells):
                    s1, s2 = _cpm_sums(X, chunk, plan["pos_ext"], G, plan["has_dup"])
                    S1[ti] += s1
                    S2[ti] += s2
                n_t[ti] += rows.size
        log(f"    {plan['path'].name}: {len(plan['targets'])} targets, {plan['ctrl_rows'].size:,} control cells "
            f"({plan['n_ctrl_skipped']:,} shared with an earlier file, not re-read) in {time.time() - t0:.0f} s",
            flush=True)

    # ---- Means, variances, standard errors (in place on the (T, G) float64 sums).
    ctrl_cpm = c1 / n_ctrl
    var_c = np.maximum((c2 - n_ctrl * ctrl_cpm ** 2) / (n_ctrl - 1), 0.0)
    x = S1
    x /= n_t[:, None]                                          # S1 now holds the target means
    v = S2
    v -= n_t[:, None] * x ** 2
    v /= (n_t - 1)[:, None]
    np.maximum(v, 0.0, out=v)                                  # rounding can leave tiny negatives
    v /= n_t[:, None] * (x + eps) ** 2                         # v now holds the target's variance term
    v += (var_c / (n_ctrl * (ctrl_cpm + eps) ** 2))[None, :]
    np.sqrt(v, out=v)
    v /= _LN2
    v[:, ~measured] = np.inf
    x[:, ~measured] = 0.0
    ctrl_cpm[~measured] = 0.0

    gi = pd.Index(genes).get_indexer(targets)                  # each target's own gene on the axis (-1 if absent)
    own = np.where(gi >= 0, x[np.arange(T), np.maximum(gi, 0)], np.nan)
    base = np.where(gi >= 0, ctrl_cpm[np.maximum(gi, 0)], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        kd = np.where((gi >= 0) & measured[np.maximum(gi, 0)] & (base > 0), own / base, np.nan)
    meta = pd.DataFrame({"n_cells": n_t.astype(np.int64), "kd_remaining": kd, "phenotype_p": np.nan},
                        index=pd.Index(targets, name="target"))
    log(f"  {name}: done in {time.time() - t_start:.0f} s; {int(measured.sum()):,} of {G:,} axis genes measured",
        flush=True)
    return ResponseSet(name=name, genes=genes, measured=measured, targets=targets, target_cpm=x, ctrl_cpm=ctrl_cpm,
                       se_lfc=v, meta=meta)
