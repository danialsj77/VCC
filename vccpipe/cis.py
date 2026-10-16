"""Genome neighbourhood of each target: the CRISPRi "cis" effect.

Why this module exists
----------------------
The VCC screens use CRISPR interference (dCas9-KRAB). The guide RNA brings the KRAB repressor to the target's
promoter, where it spreads repressive chromatin over a stretch of DNA, typically a few kb. Any *other* gene whose
promoter lies in that stretch can be silenced as well. The classic case is a bidirectional promoter: two genes whose
transcription start sites (TSSs) are a few hundred bp apart, pointing away from each other.

The scorer removes only the target's own gene from each perturbation's metrics, so a silenced neighbour is
scored like any other response. Neighbour knockdowns are large, consistent across cell lines (they come from the
CRISPRi mechanism, not from the cell's biology), and predictable from genome coordinates alone. That makes them
one of the few responses we can predict for a target that no public screen measured.

The model
---------
A step function of the distance between the two TSSs:

    cis(t, g) = effect[b]       where  bins_kb[b] <= |TSS_g - TSS_t| / 1000 < bins_kb[b + 1]

and 0 when the genes are on different chromosomes, further apart than the last bin edge (50 kb by default), when
either TSS is unknown, and for the target's own gene (the knockdown itself is modelled elsewhere).

`effect[b]` is fitted on a source screen (e.g. the K562 genome-wide screen) as the **median** log2 fold change over
all (target, neighbour gene) pairs in bin b. Why the median:

* It is robust to trans effects. A strong knockdown (e.g. of a ribosomal gene) changes thousands of genes,
  neighbours included, by chance; a mean would pick that up as a "cis" signal.
* It predicts a change only where *most* neighbours at that distance change. Calling a gene that does not respond
  costs jac (over-calling) and possibly fid (wrong sign), so a typical-case prior is the safer default.

A bin with fewer than `min_pairs` (20) pairs gets effect 0: too little evidence to predict anything.

Typical use::

    tss = load_tss("data/annotation/gencode.v32.basic.annotation.gtf.gz")
    cis = CisModel().fit(k562, tss)                 # k562: a vccpipe.sources.ResponseSet
    Path("data/processed/cis.json").write_text(cis.to_json())
    prior = cis.predict(panel_targets, vcc_genes, tss)   # (T, G) float32 log2FC prior

Which annotation
----------------
Genes are matched to TSSs **by symbol**, so the GTF must use the same symbols as the VCC gene axis. The axis, the
H1 files and the Replogle screens all use the 10x Genomics GRCh38-2020-A reference, which is built on GENCODE v32,
so GENCODE **v32 basic** is the default (configs/v1.yaml `data.gencode`). A newer release silently loses genes that
were renamed since: with v46 (the previous default), 506 of the 18,533 axis genes get no TSS, ~476 of them only
because their symbol changed (matched by Ensembl id: KARS -> KARS1, MARCH2 -> MARCHF2, TAZ -> TAFAZZIN,
C1orf112 -> FIRRM, ...). Those genes then never get a cis prior, neither as targets nor as neighbours, and their
pairs are missing from the fit. A few axis genes have no TSS in any release we accept (immunoglobulin / T-cell
receptor segments and pseudogenes, filtered out by gene_type): that is expected. If you change the annotation,
delete data/processed/tss.csv and cis.json so they are rebuilt.

Coordinates are GTF coordinates (1-based); only differences between them matter here.
"""

import gzip
import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BINS_KB = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
DEFAULT_GENE_TYPES = ("protein_coding", "lncRNA")
MIN_PAIRS = 20                       # a bin needs at least this many (target, gene) pairs to get a non-zero effect

# GTF column 9 looks like:  gene_id "ENSG00000223972.5"; gene_type "lncRNA"; gene_name "DDX11L2"; level 2; ...
# The "(?:^|;)\s*" anchor makes sure we match the key itself, not a key that merely ends with it
# (e.g. "havana_gene_type" in some annotations).
_ATTR_RE = {key: re.compile(r'(?:^|;)\s*' + key + r' "([^"]*)"')
            for key in ("gene_name", "gene_type", "gene_biotype")}


# --------------------------------------------------------------------------------------------------------------
# TSS table from a GENCODE GTF
# --------------------------------------------------------------------------------------------------------------

def _open_text(path):
    """Open a GTF as text, gunzipping on the fly if it is gzipped (detected from the magic bytes, not the name)."""
    path = Path(path)
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if gz else open(path, "rt")


def _attr(attrs, key):
    m = _ATTR_RE[key].search(attrs)
    return m.group(1) if m else None


def load_tss(gtf_path, gene_types=DEFAULT_GENE_TYPES):
    """One TSS per gene symbol, from a GENCODE GTF (gzipped or plain).

    Returns a DataFrame with columns gene_name (str), chrom (str, as written in the GTF, e.g. "chr1"),
    tss (int64, 1-based) and strand ("+" / "-"), one row per gene_name, in file order.

    Use GENCODE v32 basic for the VCC axis (its symbols match the 10x GRCh38-2020-A reference the axis comes from;
    newer releases rename a few hundred axis genes, which then get no TSS: see "Which annotation" above).

    Rules, and why:
    * Only `gene` records. A gene record spans all its transcripts, so its 5' end is the gene's outermost TSS.
      (Transcript-level TSSs would be more precise for genes with alternative promoters; the gene record is a
      simple, reproducible choice and good enough at the kb scale of the bins.)
    * TSS = `start` on the + strand, `end` on the − strand (transcription runs from the 5' end).
    * Only `gene_types` (protein_coding and lncRNA by default): the VCC gene axis consists of such genes, and the
      many small RNA / pseudogene records would otherwise steal gene symbols. Older GENCODE releases (before ~v31)
      call lncRNAs "lincRNA", "antisense", ...; pass those types explicitly if you use such a file. Ensembl GTFs
      name the attribute gene_biotype, which we accept as a fallback.
    * The first record per gene_name wins. Duplicated symbols are rare and mostly the pseudo-autosomal genes that
      GENCODE lists on chrX and again on chrY; chrX comes first in the file, which is what we want.

    The file (~1.5 GB uncompressed for GENCODE comprehensive) is streamed line by line, so memory stays small.
    """
    gene_types = set(gene_types)
    seen = set()
    names, chroms, tss, strands = [], [], [], []
    with _open_text(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            # Cheap check first: split off only the first 3 columns. ~97% of lines are transcript/exon/CDS/UTR
            # records, and skipping them before the full split keeps the parse fast (a few seconds per GTF).
            head = line.split("\t", 3)
            if len(head) < 4 or head[2] != "gene":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            attrs = f[8]
            name = _attr(attrs, "gene_name")
            gtype = _attr(attrs, "gene_type") or _attr(attrs, "gene_biotype")
            if not name or gtype not in gene_types or name in seen:
                continue
            strand = f[6]
            if strand not in ("+", "-"):
                continue                                   # unstranded record: no defined start site
            seen.add(name)
            names.append(name)
            chroms.append(f[0])
            tss.append(int(f[3]) if strand == "+" else int(f[4]))
            strands.append(strand)
    return pd.DataFrame({"gene_name": pd.Series(names, dtype=object).astype(str),
                         "chrom": pd.Series(chroms, dtype=object).astype(str),
                         "tss": np.asarray(tss, dtype=np.int64),
                         "strand": pd.Series(strands, dtype=object).astype(str)})


# --------------------------------------------------------------------------------------------------------------
# Vectorized neighbour search
# --------------------------------------------------------------------------------------------------------------

def _locate(names, tss):
    """Chromosome code and TSS position for each name.

    Returns (chrom_code, pos): int arrays aligned with `names`. Chromosomes are integer-coded (factorized) so the
    neighbour search compares integers, not strings; chrom_code == -1 means "no TSS known for this name".
    """
    # Rows with a missing field (e.g. after a CSV round trip) carry no usable position: drop them first.
    t = tss.dropna(subset=["gene_name", "chrom", "tss"])
    t = t.drop_duplicates("gene_name", keep="first")       # the first record per symbol wins, as in load_tss
    code, _ = pd.factorize(t["chrom"].astype(str).to_numpy())
    index = pd.Index(t["gene_name"].astype(str).to_numpy())
    loc = index.get_indexer(pd.Index(np.asarray(names).astype(str)))   # -1 where the name has no TSS
    known = loc >= 0
    chrom = np.full(loc.size, -1, dtype=np.int64)
    pos = np.zeros(loc.size, dtype=np.int64)
    chrom[known] = code[loc[known]]
    pos[known] = t["tss"].to_numpy(dtype=np.int64)[loc[known]]
    return chrom, pos


def _pairs_within(t_chrom, t_pos, g_chrom, g_pos, window_bp):
    """All (target row, gene column) pairs on the same chromosome with |TSS_g - TSS_t| < window_bp.

    Entries with chrom code -1 are ignored (unknown TSS, or masked out by the caller).
    Returns (t_idx, g_idx, dist_bp) as int64 arrays.

    How: per chromosome, sort the genes by TSS once; then each target's neighbours form one contiguous run
    [lo, hi) of the sorted array, found by two binary searches (np.searchsorted, vectorized over all targets).
    The runs are expanded into explicit pairs without a Python loop over targets. Cost: O((T + G) log G + pairs),
    i.e. well under a second for ~10k targets on the 18.5k-gene axis.
    """
    out_t, out_g, out_d = [], [], []
    for c in np.unique(t_chrom[t_chrom >= 0]):
        gi = np.flatnonzero(g_chrom == c)
        if gi.size == 0:
            continue
        gi = gi[np.argsort(g_pos[gi], kind="stable")]          # gene columns of this chromosome, by TSS
        gp = g_pos[gi]
        ti = np.flatnonzero(t_chrom == c)
        tp = t_pos[ti]
        lo = np.searchsorted(gp, tp - window_bp, side="right")  # first gene with TSS > tp - window
        hi = np.searchsorted(gp, tp + window_bp, side="left")   # first gene with TSS >= tp + window
        n = hi - lo                                             # neighbours per target
        total = int(n.sum())
        if total == 0:
            continue
        # Expand the runs: pair k belongs to target rep[k]; its rank inside that target's run is
        # k - (start of the run in the flat pair list); its position in gp is lo[rep[k]] + rank.
        rep = np.repeat(np.arange(ti.size), n)
        run_start = np.cumsum(n) - n
        k = np.arange(total) - run_start[rep] + lo[rep]
        out_t.append(ti[rep])
        out_g.append(gi[k])
        out_d.append(np.abs(gp[k] - tp[rep]))
    if not out_t:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    return (np.concatenate(out_t).astype(np.int64), np.concatenate(out_g).astype(np.int64),
            np.concatenate(out_d).astype(np.int64))


def _neighbour_pairs(targets, genes, tss, max_kb, target_ok=None, gene_ok=None):
    """(t_idx, g_idx, distance_kb) for every target/gene pair within max_kb, excluding each target's own gene.

    target_ok / gene_ok are optional boolean masks restricting which targets / gene columns may pair.
    """
    targets = np.asarray(targets).astype(str)
    genes = np.asarray(genes).astype(str)
    t_chrom, t_pos = _locate(targets, tss)
    g_chrom, g_pos = _locate(genes, tss)
    if target_ok is not None:
        t_chrom = np.where(target_ok, t_chrom, -1)
    if gene_ok is not None:
        g_chrom = np.where(gene_ok, g_chrom, -1)
    ti, gi, d_bp = _pairs_within(t_chrom, t_pos, g_chrom, g_pos, max_kb * 1000.0)
    # The target's own gene is excluded by *name*, so this also works if the axis held a symbol twice.
    # (Another gene sharing the target's exact TSS is kept: it sits at distance 0 and is a genuine neighbour.)
    not_self = genes[gi] != targets[ti]
    return ti[not_self], gi[not_self], d_bp[not_self] / 1000.0


def cis_pairs(rs, tss, max_kb=DEFAULT_BINS_KB[-1], min_cells=50, min_ctrl_cpm=None):
    """Every (target, measured neighbour gene) pair of a source screen, with its distance and log2FC.

    The training table of `CisModel.fit`, exposed for diagnostics (e.g. plot lfc against distance_kb).
    Returns a DataFrame: target, gene, distance_kb, lfc (one row per pair).

    rs: a vccpipe.sources.ResponseSet (uses rs.genes, rs.measured, rs.targets, rs.lfc(), rs.meta['n_cells'];
        rs.ctrl_cpm only if min_ctrl_cpm is given).
    min_cells: only targets with at least this many cells (noisy small groups would blur the medians).
    min_ctrl_cpm: optional extension (not in DESIGN_v1, default off). If set, use only neighbour genes whose
        control mean CPM in the source is at least this value. Why you might want it: for a barely expressed
        gene the log2FC is pulled towards 0 by the pseudocount, which dilutes the medians, and the scorer only
        tests genes above 5 CPM anyway.
    """
    genes = np.asarray(rs.genes).astype(str)
    targets = np.asarray(list(rs.targets)).astype(str)
    if "n_cells" not in rs.meta.columns:
        raise ValueError("rs.meta needs an 'n_cells' column (cells per target) to apply min_cells")
    n_cells = rs.meta["n_cells"].reindex(targets).to_numpy(dtype=np.float64, na_value=np.nan)
    target_ok = n_cells >= min_cells                         # NaN (target missing from meta) → False
    gene_ok = np.asarray(rs.measured, dtype=bool)           # only genes the source actually measured
    if min_ctrl_cpm is not None:
        gene_ok = gene_ok & (np.asarray(rs.ctrl_cpm, dtype=np.float64) >= min_ctrl_cpm)

    ti, gi, d_kb = _neighbour_pairs(targets, genes, tss, max_kb, target_ok, gene_ok)

    # rs.lfc() materializes the full (T, G) matrix once (~0.7-1.5 GB for the K562 genome-wide screen). We need
    # only a few tens of thousands of entries, but going through lfc() keeps one definition of the fold change
    # (pseudocount, "0 where not measured") for the whole pipeline. Free it right after the gather.
    lfc = rs.lfc()
    vals = np.asarray(lfc[ti, gi], dtype=np.float64)
    del lfc
    ok = np.isfinite(vals)
    return pd.DataFrame({"target": targets[ti[ok]], "gene": genes[gi[ok]],
                         "distance_kb": d_kb[ok], "lfc": vals[ok]})


# --------------------------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------------------------

@dataclass
class CisModel:
    """Binned prior for neighbour silencing: log2FC of gene g when target t is knocked down, from TSS distance.

    bins_kb: bin edges for |TSS_g - TSS_t| in kb; bin b is [bins_kb[b], bins_kb[b+1]). The last edge is the
             neighbourhood window: genes further away get no cis effect.
    effect:  fitted median log2FC per bin (len(bins_kb) - 1 values), None before fitting.
    n_pairs: number of (target, gene) pairs per bin in the fit (kept so a reader can judge each bin's support).
    """
    bins_kb: list = field(default_factory=lambda: list(DEFAULT_BINS_KB))
    effect: list | None = None
    n_pairs: list | None = None

    def __post_init__(self):
        self.bins_kb = [float(b) for b in self.bins_kb]
        e = np.asarray(self.bins_kb)
        if e.size < 2 or not np.all(np.diff(e) > 0) or e[0] < 0:
            raise ValueError(f"bins_kb must be >= 0 and strictly increasing with >= 2 edges, got {self.bins_kb}")
        nb = e.size - 1
        if self.effect is not None:
            self.effect = [float(x) for x in self.effect]
            if len(self.effect) != nb:
                raise ValueError(f"effect has {len(self.effect)} values for {nb} bins")
        if self.n_pairs is not None:
            self.n_pairs = [int(x) for x in self.n_pairs]
            if len(self.n_pairs) != nb:
                raise ValueError(f"n_pairs has {len(self.n_pairs)} values for {nb} bins")

    # -- helpers ------------------------------------------------------------------------------------------------
    def _bin(self, distance_kb):
        """Bin index of each distance; -1 for distances outside [bins_kb[0], bins_kb[-1])."""
        edges = np.asarray(self.bins_kb)
        b = np.searchsorted(edges, distance_kb, side="right") - 1
        return np.where((b >= 0) & (b < edges.size - 1), b, -1)

    # -- fitting ------------------------------------------------------------------------------------------------
    def fit(self, rs, tss, min_cells=50, min_pairs=MIN_PAIRS, min_ctrl_cpm=None, min_abs_effect=0.0):
        """Fit the per-bin median log2FC on a source screen. Returns self.

        Uses every target of `rs` that has a TSS and at least `min_cells` cells, paired with every gene of the
        VCC axis that `rs` measured, that has a TSS on the same chromosome closer than bins_kb[-1], and that is
        not the target itself. A bin with fewer than `min_pairs` pairs gets effect 0.0, and so does a bin whose
        median |log2FC| is below `min_abs_effect` (far bins hover around 0; a tiny non-zero median there would
        nudge every gene within the window for no real reason).
        (`min_pairs` and `min_ctrl_cpm` are optional extensions of the DESIGN_v1 signature; see `cis_pairs`.)
        """
        pairs = cis_pairs(rs, tss, max_kb=self.bins_kb[-1], min_cells=min_cells, min_ctrl_cpm=min_ctrl_cpm)
        b = self._bin(pairs["distance_kb"].to_numpy())
        v = pairs["lfc"].to_numpy()
        nb = len(self.bins_kb) - 1
        n_pairs = np.bincount(b[b >= 0], minlength=nb)
        # One median per bin: nb is ~6, so a short loop over bins is clearer than any vectorized trick.
        effect = [float(np.median(v[b == k])) if n_pairs[k] >= min_pairs else 0.0 for k in range(nb)]
        effect = [e if abs(e) >= min_abs_effect else 0.0 for e in effect]
        self.effect = effect
        self.n_pairs = [int(n) for n in n_pairs]
        return self

    # -- prediction ---------------------------------------------------------------------------------------------
    def predict(self, targets, genes, tss):
        """Dense (T, G) float32 log2FC prior.

        Entry (t, g) is effect[bin] for every gene g of `genes` whose TSS lies on the target's chromosome within
        bins_kb[-1] of the target's TSS, whether or not any source measured g (genome distance is all we need).
        It is 0 elsewhere: different chromosome, too far, unknown TSS (target or gene), and the target's own column.
        """
        if self.effect is None:
            raise ValueError("CisModel is not fitted: call fit() or load fitted parameters with from_json()")
        genes = np.asarray(genes).astype(str)
        out = np.zeros((len(targets), genes.size), dtype=np.float32)
        ti, gi, d_kb = _neighbour_pairs(list(targets), genes, tss, self.bins_kb[-1])
        b = self._bin(d_kb)
        ok = b >= 0
        out[ti[ok], gi[ok]] = np.asarray(self.effect, dtype=np.float32)[b[ok]]
        return out

    # -- reporting and persistence ------------------------------------------------------------------------------
    def summary(self):
        """One row per bin: lo_kb, hi_kb, n_pairs, effect (for printing in logs and notebooks)."""
        nb = len(self.bins_kb) - 1
        return pd.DataFrame({"lo_kb": self.bins_kb[:-1], "hi_kb": self.bins_kb[1:],
                             "n_pairs": self.n_pairs if self.n_pairs is not None else [0] * nb,
                             "effect": self.effect if self.effect is not None else [np.nan] * nb})

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        known = {f.name for f in fields(cls)}               # ignore keys a later version might add
        return cls(**{k: v for k, v in d.items() if k in known})
