"""Tiny fake versions of the public datasets, with the real files' structure, for laptop tests.

Each builder writes a file whose layout matches the real one (column names, index formats, control labels,
raw-count matrices) but with a few hundred cells/rows, and returns the ground truth it planted, so tests can
check that loaders and models recover it.
"""

import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

CONTROL = "non-targeting"


def vcc_genes(controls_dir=None, n=None):
    """The real VCC gene axis if the controls bundle is present, else synthetic symbols."""
    if controls_dir is not None and (Path(controls_dir) / "gene_names.csv").exists():
        g = pd.read_csv(Path(controls_dir) / "gene_names.csv")["gene_name"].astype(str).to_numpy()
    else:
        g = np.array([f"GENE{i}" for i in range(n or 2000)])
    return g if n is None else g[:n]


def planted_effects(targets, genes, rng, n_affected=40, sd=1.0, null_frac=0.3):
    """log2FC matrix (T, G): each non-null target changes n_affected random genes; its own gene is knocked down."""
    lfc = np.zeros((len(targets), len(genes)))
    gidx = {g: i for i, g in enumerate(genes)}
    null = rng.random(len(targets)) < null_frac
    for i, t in enumerate(targets):
        if not null[i]:
            idx = rng.choice(len(genes), n_affected, replace=False)
            lfc[i, idx] = rng.normal(0, sd, n_affected)
        if t in gidx:
            lfc[i, gidx[t]] = np.log2(0.15)
    return lfc, null


def make_fake_replogle_bulk(path, genes, targets, rng, n_ntc=40, cells=(60, 300), base_umi=12000):
    """Replogle 2022 *_raw_bulk_01.h5ad lookalike: X = mean raw counts per cell per guide row.

    obs index "<id>_<GENE>_<transcript>_<ENSG>", controls "<id>_non-targeting_non-targeting_non-targeting";
    obs columns num_cells_filtered (some control rows NaN, as in the real file), fold_expr, energy_test_p_value;
    var index = fake Ensembl ids, var['gene_name'] = symbols (one duplicate, as in the real file).
    Returns dict(lfc=(T,G) planted log2FC, null=(T,) bool, base_cpm=(G,)).
    """
    G = len(genes)
    base = rng.gamma(0.4, 1.0, G) + 1e-3
    base_cpm = base / base.sum() * 1e6
    lfc, null = planted_effects(targets, genes, rng)
    rows, idx, meta = [], [], []
    for j in range(n_ntc):
        n = int(rng.integers(*cells))
        counts = rng.poisson(base_cpm * base_umi / 1e6 * n) / n          # mean counts per cell over n cells
        rows.append(counts)
        idx.append(f"{10000 + j}_{CONTROL}_{CONTROL}_{CONTROL}")
        meta.append((np.nan if j == 0 else float(n), 1.0, rng.uniform(0.05, 1.0)))
    for i, t in enumerate(targets):
        n = int(rng.integers(*cells))
        mu = base_cpm * np.exp2(lfc[i])
        counts = rng.poisson(mu / mu.sum() * base_umi * n) / n
        rows.append(counts)
        idx.append(f"{i}_{t}_P1P2_ENSG{i:011d}")
        meta.append((float(n), 0.15, 1e-6 if not null[i] else rng.uniform(0.2, 1.0)))
    obs = pd.DataFrame(meta, columns=["num_cells_filtered", "fold_expr", "energy_test_p_value"], index=idx)
    gene_name = list(genes)
    gene_name[-1] = gene_name[0]                                         # one duplicated symbol
    var = pd.DataFrame({"gene_name": gene_name}, index=[f"ENSG9{i:010d}" for i in range(G)])
    ad.AnnData(X=np.asarray(rows, dtype=np.float32), obs=obs, var=var).write_h5ad(path)
    return {"lfc": lfc, "null": null, "base_cpm": base_cpm}


def make_fake_h1(paths, genes, targets, rng, cells_per_target=420, n_ctrl=600, umi=54000, lfc=None):
    """Arc VCC-2025 single-cell lookalikes split over several files (targets spread across files; the SAME
    control cells in every file). obs: target_gene, guide_id, batch; var index = symbols; X = CSR raw counts.
    Returns dict(lfc=(T,G) planted log2FC, base_cpm=(G,))."""
    G = len(genes)
    base = rng.gamma(0.4, 1.0, G) + 1e-3
    base_cpm = base / base.sum() * 1e6
    if lfc is None:
        lfc, _ = planted_effects(targets, genes, rng)

    def cells(cpm, n):
        lib = rng.lognormal(np.log(umi), 0.3, n)
        return sp.csr_matrix(rng.poisson(cpm[None, :] / 1e6 * lib[:, None]).astype(np.float32))

    ctrl = cells(base_cpm, n_ctrl)
    groups = np.array_split(np.arange(len(targets)), len(paths))
    for path, grp in zip(paths, groups):
        blocks, labels = [ctrl], [CONTROL] * n_ctrl
        for i in grp:
            mu = base_cpm * np.exp2(lfc[i])
            blocks.append(cells(mu / mu.sum() * 1e6, cells_per_target))
            labels += [targets[i]] * cells_per_target
        n = len(labels)
        obs = pd.DataFrame({"target_gene": labels, "guide_id": "g", "batch": "Flex_1_01"},
                           index=[f"bc{k}" for k in range(n)])
        ad.AnnData(X=sp.vstack(blocks).tocsr(), obs=obs, var=pd.DataFrame(index=list(genes))).write_h5ad(path)
    return {"lfc": lfc, "base_cpm": base_cpm}


def make_fake_gtf(path, genes, rng, chrom_len=5_000_000):
    """GENCODE-like gzipped GTF with one 'gene' record per symbol (gene_type protein_coding), random positions
    on 3 chromosomes; a few deliberate close pairs. Returns DataFrame gene_name, chrom, tss, strand."""
    rows = []
    for i, g in enumerate(genes):
        chrom = f"chr{1 + i % 3}"
        start = int(rng.integers(1, chrom_len))
        strand = "+" if rng.random() < 0.5 else "-"
        rows.append((g, chrom, start, start + 5000, strand))
    for k in range(0, min(40, len(rows) - 1), 2):                       # bidirectional pairs: TSSs 300 bp apart,
        g, chrom, start, end, strand = rows[k]                          # opposite strands, pointing away
        name = rows[k + 1][0]
        if strand == "+":                                               # TSS_k = start → partner '-' ends at start-300
            rows[k + 1] = (name, chrom, start - 5300, start - 300, "-")
        else:                                                           # TSS_k = end → partner '+' starts at end+300
            rows[k + 1] = (name, chrom, end + 300, end + 5300, "+")
    with gzip.open(path, "wt") as fh:
        fh.write("##description: fake GENCODE\n")
        for g, chrom, start, end, strand in rows:
            attrs = f'gene_id "ENSG_{g}"; gene_type "protein_coding"; gene_name "{g}"; level 2;'
            fh.write(f"{chrom}\tHAVANA\tgene\t{start}\t{end}\t.\t{strand}\t.\t{attrs}\n")
            fh.write(f"{chrom}\tHAVANA\texon\t{start}\t{start + 100}\t.\t{strand}\t.\t{attrs}\n")
    tss = [(g, c, s if st == "+" else e, st) for g, c, s, e, st in rows]
    return pd.DataFrame(tss, columns=["gene_name", "chrom", "tss", "strand"])
