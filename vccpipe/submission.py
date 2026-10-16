"""Build, check and package a VCC submission with bounded memory.

The full file has 360,000 cells × 18,533 genes (~2.1e9 stored values). Building it in one piece needs ~50 GB of
RAM, so we write it in chunks (one h5ad per group of targets), then concatenate the chunks on disk.

`vcc prep` itself loads the whole file (~1.3× the matrix + 2 GB, i.e. ~35 GB at full size): run it on a big
machine (TSUBAME node_q) or try it on a 24 GB Mac with memory compression. When packaging, it also writes an
uncompressed copy of the prediction plus its .zst (~1.2× our h5ad) into a temporary folder ($TMPDIR, else /tmp);
`vcc_prep` warns when that folder looks too small.

Disk: the chunks are written lzf-compressed (~30% smaller, fast to read back); the concatenated file is uncompressed,
as `vcc prep` expects. Peak use while concatenating is the compressed chunks + the full h5ad.

Random numbers: every (context, target) gets its own generator, np.random.default_rng([seed, context_index,
target_index]) (indices into the bundle's context and target lists). A target's cells therefore depend only on
its own prediction, not on the other targets or on how they are split into chunks.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.experimental import concat_on_disk

from .common import CELLS_PER_PERT, CONTROL, load_bundle_info, load_controls

MAX_STORED = 4_750_000_000
MAX_COUNTS_PER_CELL = 1_000_000


def build_submission(predict_cpm, make_generator, controls_dir, out_h5ad, workdir, cells_per_pert=CELLS_PER_PERT,
                     chunk_targets=50, seed=0):
    """Write the submission h5ad.

    predict_cpm(ctx, targets, ctrl_X) -> array (len(targets), n_genes) of predicted mean CPM
    make_generator(ctrl_X) -> callable(target_cpm, n_cells, rng) -> CSR counts (see vccpipe.generator)
    Contexts come from the bundle's manifest and are handled by name, so labels can't be swapped.
    seed: base seed; target j of context k uses np.random.default_rng([seed, k, j]) (see the module docstring).
    """
    controls_dir, out_h5ad, workdir = Path(controls_dir), Path(out_h5ad), Path(workdir)
    genes, targets, contexts = load_bundle_info(controls_dir)
    chunk_dir = workdir / "submission_chunks"
    shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(parents=True)
    chunks, nnz, max_total = [], 0, 0.0
    for k, ctx in enumerate(contexts):
        ctrl = load_controls(controls_dir, ctx, genes)
        pred = np.asarray(predict_cpm(ctx, targets, ctrl))
        if pred.shape != (len(targets), genes.size):
            raise ValueError(f"prediction for {ctx} has shape {pred.shape}")
        gen = make_generator(ctrl)
        for start in range(0, len(targets), chunk_targets):
            sel = targets[start:start + chunk_targets]
            # One seeded stream per (context, target): j = start + i is the target's index in the full panel, so
            # the cells do not change with chunk_targets or with what the other targets predict.
            blocks = [gen(pred[start + i], cells_per_pert, np.random.default_rng([seed, k, start + i]))
                      for i in range(len(sel))]
            X = sp.vstack(blocks).tocsr()
            X.eliminate_zeros()
            if X.data.size and (X.data.min() < 0 or np.modf(X.data)[0].any()):
                raise ValueError("counts must be non-negative integers")
            max_total = max(max_total, float(np.asarray(X.sum(axis=1)).max()))
            nnz += X.nnz
            obs = pd.DataFrame({"target_gene": np.repeat(sel, cells_per_pert), "context": ctx})
            obs.index = [f"{ctx}_{start * cells_per_pert + j}" for j in range(len(obs))]   # unique per context
            path = chunk_dir / f"{ctx}_{start:04d}.h5ad"
            # lzf: cheap compression (~30% smaller chunks on real-like cells, little extra time); concat_on_disk
            # reads it back transparently and writes the final file uncompressed.
            ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(path, compression="lzf")
            chunks.append(path)
        print(f"  context {ctx}: {len(targets)} targets × {cells_per_pert} cells", flush=True)
    if max_total > MAX_COUNTS_PER_CELL:
        raise ValueError(f"a cell has {max_total:.0f} counts (> 1e6)")
    if nnz > MAX_STORED:
        raise ValueError(f"{nnz:,} stored values exceed the cap")
    out_h5ad.unlink(missing_ok=True)
    concat_on_disk([str(p) for p in chunks], str(out_h5ad), axis=0, join="inner")
    shutil.rmtree(chunk_dir, ignore_errors=True)
    n_cells = len(contexts) * len(targets) * cells_per_pert
    print(f"wrote {out_h5ad.name}: {n_cells:,} cells, {nnz:,} stored values ({nnz / n_cells:,.0f}/cell)")
    return out_h5ad, contexts


def vcc_exe():
    return shutil.which("vcc") or str(Path.home() / ".local" / "bin" / "vcc")


def vcc_prep(sub_h5ad, controls_dir, contexts, out_vcc=None, dry_run=True, cells_per_pert=None):
    """Validate (dry run) or package a submission exactly as the platform will. Contexts are always explicit:
    vcc 0.2.2 assumes A,B,C otherwise."""
    controls_dir = Path(controls_dir)
    cmd = [vcc_exe(), "prep", os.path.relpath(sub_h5ad), "-g", controls_dir / "gene_names.csv",
           "--perts", controls_dir / "pert_counts.csv", "--contexts", ",".join(contexts)]
    if cells_per_pert:
        cmd += ["--cells-per-pert", str(cells_per_pert)]
    cmd += ["--dry-run"] if dry_run else ["-o", out_vcc, "-f"]
    if not dry_run:
        _warn_if_tmp_small(sub_h5ad)
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    print(proc.stdout[-3000:], proc.stderr[-3000:])
    if proc.returncode != 0:
        raise RuntimeError("vcc prep failed")
    return proc


def _warn_if_tmp_small(sub_h5ad):
    """Packaging writes an uncompressed copy of the prediction plus its .zst into a temporary folder (vcc prep uses
    Python's TemporaryDirectory: $TMPDIR, else /tmp). On a compute node /tmp can be small; warn before a long run
    dies with 'No space left on device'. The need is estimated as 1.25 × our h5ad (the copy + ~20% for the .zst)."""
    try:
        tmp = Path(tempfile.gettempdir())
        need = 1.25 * Path(sub_h5ad).stat().st_size
        free = shutil.disk_usage(tmp).free
    except OSError as e:                     # a check, never a reason to stop packaging
        print(f"  (could not check the temporary folder's free space: {e})", flush=True)
        return
    print(f"  vcc prep temporary folder: {tmp} ({free / 1e9:,.1f} GB free, ~{need / 1e9:,.1f} GB needed)", flush=True)
    if free < need:
        print(f"  WARNING: {tmp} may be too small for packaging; point TMPDIR at a larger disk "
              "(on TSUBAME: export TMPDIR=$T4TMPDIR)", flush=True)
