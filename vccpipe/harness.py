"""Score predictions with Arc's official scorer (cell-eval2, vcc2026 profile) on a proxy with known answers.

Typical use::

    h = Harness(real_h5ad, workdir, gpu=False)
    h.build_bundle()                                   # once per proxy: the 0 (baseline) and 1 (replicate) ends
    scaled, raw = h.score("my_model", pred_h5ad)       # any number of predictions

Predictions for the local scorer must contain the proxy's control cells too (the scorer checks that the
perturbation sets match); under the competition rule those control rows are ignored.

`write_prediction` seeds the cell generator per target, so variants are compared on the same generated cells
wherever their predictions agree (a paired comparison; see its docstring).
"""

import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import CELLS_PER_PERT, CONTROL

SHORT = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse", "de_wilcoxon_lfc_nmae": "nmae",
         "de_wilcoxon_direction_fidelity_yield_raw": "fid", "de_wilcoxon_direction_reach_raw": "reach",
         "de_wilcoxon_sig_jaccard": "jac", "avg_score": "overall"}
ORDER = ["overall", "pds", "mse", "nmae", "fid", "reach", "jac"]


def cell_eval2_exe():
    return shutil.which("cell-eval2") or str(Path(sys.executable).with_name("cell-eval2"))


def run_cmd(cmd, log_path=None, quiet=False):
    """Run a command; save its output to a log; raise with the tail of the output on failure."""
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if log_path:
        Path(log_path).write_text(f"$ {' '.join(map(str, cmd))}\n\n{proc.stdout}\n{proc.stderr}")
    if not quiet:
        print(f"  {Path(str(cmd[0])).name} {cmd[1]} … exit={proc.returncode} ({time.time() - t0:.0f} s)", flush=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-4000:])
        raise RuntimeError(f"command failed: {cmd[:2]}")
    return proc


class Harness:
    def __init__(self, real_h5ad, workdir, gpu=False):
        self.real = Path(real_h5ad)
        self.work = Path(workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        engine = [] if gpu else ["--set", "device=cpu", "--set", "de.backend=pdex"]
        self.common = ["--preset", "vcc2026", "--pert-col", "target_gene", *engine]
        self.bundle = self.work / "bundle"
        self.exe = cell_eval2_exe()

    def build_bundle(self, force=False):
        """Baseline arm (context mean, tiled) + 5 split-half replicates → the scale's 0 and 1 ends."""
        if (self.bundle / "manifest.json").exists() and not force:
            print("  bundle exists, reusing it")
            return
        arm = self.work / "baseline" / "baseline_pred.h5ad"
        run_cmd([self.exe, "baseline", "-ar", self.real, "-o", self.work / "baseline", "--emit", "tile",
                 "--save-pred", arm, *self.common], self.work / "log_baseline.txt")
        run_cmd([self.exe, "prep-real-bundle", "--real", self.real, "--baseline", arm, "-o", self.bundle,
                 "--id", self.work.name, "--force", *self.common], self.work / "log_bundle.txt")
        arm.unlink(missing_ok=True)                    # the dense arm is large and no longer needed

    def score(self, name, pred_h5ad, keep_pred=False):
        """Score one prediction file. Returns (scaled scores, raw metric means) as pandas Series."""
        out = self.work / f"run_{name}"
        run_cmd([self.exe, "run", "-ap", pred_h5ad, "-ar", self.real, "-o", out, *self.common],
                self.work / f"run_{name}.log")
        run_cmd([self.exe, "score", "--user-agg", out / "agg_results.csv", "--real-bundle", self.bundle,
                 "-o", out / "scored.csv"], quiet=True)
        if not keep_pred:
            Path(pred_h5ad).unlink(missing_ok=True)
        scaled = pd.read_csv(out / "scored.csv").set_index("metric")["from_replicate"].rename(index=SHORT)
        raw = pd.read_csv(out / "agg_results.csv").set_index("statistic").loc["mean"].rename(index=SHORT)
        return scaled[ORDER], raw[[c for c in ORDER if c != "overall"]]

    def per_perturbation(self, name):
        """Per-perturbation metric values of a scored prediction (long format: perturbation, metric, value)."""
        return pd.read_csv(self.work / f"run_{name}" / "results.csv")


def write_prediction(path, targets, pred_cpm, generator, ctrl_X, genes, seed=0, cells_per_pert=CELLS_PER_PERT):
    """Generate cells for a local-scoring prediction file: one block per target + the proxy's control cells.

    Random numbers are seeded per target: target i (position in `targets`) uses default_rng([seed, i]).
    Why: numpy's binomial / Poisson / gamma / beta draws consume a varying amount of the random stream depending on
    their parameters. With one shared stream, two variants that differ only in effect sizes would, from the second
    target on, imitate *different* control cells, so each variant's score would carry its own cell-sampling luck and
    picking the best of ~20 close variants would be biased upward. With one stream per target, the first draw of
    each generator (the control cells to imitate) is identical across variants for every target, and a target's
    cells do not depend on what any other target predicts.
    """
    blocks, labels = [], []
    for i, t in enumerate(targets):
        rng = np.random.default_rng([seed, i])          # independent, reproducible stream for target i
        blocks.append(generator(pred_cpm[i], cells_per_pert, rng))
        labels += [t] * cells_per_pert
    blocks.append(sp.csr_matrix(ctrl_X, dtype=np.float32))
    labels += [CONTROL] * ctrl_X.shape[0]
    a = ad.AnnData(X=sp.vstack(blocks).tocsr(),
                   obs=pd.DataFrame({"target_gene": labels}, index=[f"p{i}" for i in range(len(labels))]),
                   var=pd.DataFrame(index=genes))
    a.write_h5ad(path)
    return Path(path)


def log_result(log_csv, **row):
    """Append one row (e.g. proxy, model, scores) to a CSV experiment log."""
    row = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
    log_csv = Path(log_csv)
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(log_csv, mode="a", header=not log_csv.exists(), index=False)
