"""Step 1 of model v1: turn the downloaded public data into model inputs (all on the VCC gene axis).

    python scripts/01_build_sources.py --config configs/v1.yaml [--smoke] [--skip-proxy] [--reuse]

Writes (paths from the config):
    <processed>/k562_gw.npz     ResponseSet of the Replogle K562 genome-wide screen     (vccpipe.sources)
    <processed>/h1_2025.npz     ResponseSet of the Arc 2025 H1 data (pseudobulk)         (vccpipe.sources)
    <processed>/tss.csv         one TSS per gene (GENCODE)                               (vccpipe.cis)
    <processed>/cis.json        CisModel fitted on K562 ONLY (H1 is the held-out line)  (vccpipe.cis)
    <work>/h1_proxy/real.h5ad   H1 formatted like a VCC context, for the official scorer (vccpipe.proxy)
    <work>/h1_proxy/targets.txt the proxy's targets, one per line (+ real.kd.csv: knockdown statistics)
    <work>/h1_proxy/strength.csv, strength_filter.json
                                the proxy's strength filter: per H1 target, how many genes respond strongly in K562,
                                and which targets were left out of the proxy for that reason (see below)

and prints a short summary: targets per source, how much of the VCC panel (controls/pert_counts.csv) each source
covers, and the fitted cis effects.

The proxy's strength filter (config `proxy.max_strong_genes`, `proxy.strong_z`)
---------------------------------------------------------------------------
The H1 panel responds much more strongly than the VCC panel. Counting, per target, the genes whose K562 response has
|z| > 4.5 among genes the scorer can test (K562 control CPM > 5): VCC panel targets have a median of 3 and a maximum
of 42; H1 targets have a median of 4 but ~16% have 50 or more (up to 1,250). Left in, those few strong targets
dominate everything validation measures on the proxy (the least-squares gains, the MSE, the choice of z0), so the
model would be tuned for a regime the VCC panel does not have. So an H1 target with MORE than `max_strong_genes` such
genes in K562 is excluded from the proxy (and therefore from the gain fit). H1 targets K562 did not measure have no
such count and are kept. `max_strong_genes: null` switches the filter off.

At full size (TSUBAME node_q) the two H1 passes read ~34 GB each; expect ~1-2 h in total. --reuse keeps outputs that
already exist (e.g. after a later step failed), --skip-proxy skips the proxy (02_validate.py needs it).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]           # the project folder (this file lives in scripts/)
sys.path.insert(0, str(ROOT))

from vccpipe.cis import CisModel, load_tss            # noqa: E402
from vccpipe.common import DE_GATE_CPM, load_bundle_info  # noqa: E402
from vccpipe.proxy import build_h1_proxy               # noqa: E402
from vccpipe.sources import ResponseSet, h1_pseudobulk, load_replogle_bulk  # noqa: E402


# ---- config (the same small loader is in each script, so every script stands alone) -------------------------
def _merge(base, over):
    """Deep-merge `over` into `base`: dicts merge key by key, everything else is replaced."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path, smoke=False):
    """Read the YAML config; with smoke=True apply its `smoke:` section; make every path absolute (from ROOT)."""
    cfg = yaml.safe_load(Path(path).read_text())
    if smoke:
        cfg = _merge(cfg, cfg.get("smoke") or {})
    cfg["paths"] = {k: (ROOT / v).resolve() for k, v in cfg["paths"].items()}
    return cfg


def input_files(cfg):
    """Local paths of the downloaded inputs, as 00_download_data.sh lays them out."""
    d, data = cfg["paths"]["data_dir"], cfg["data"]
    h1 = data["h1"]
    return {"k562_gw": d / data["replogle"]["k562_gw"]["file"],
            "gtf": d / data["gencode"]["file"],
            "h1": [d / h1["dir"] / Path(remote).name for remote in h1["files"]]}


# ---- the proxy's strength filter -------------------------------------------------------------------------------
def strong_gene_counts(k562, targets, strong_z, min_ctrl_cpm=DE_GATE_CPM):
    """For each of `targets` that K562 measured: how many genes respond strongly to its knockdown in K562.

    "Strongly" = |z| > strong_z, counted only over genes the scorer can test (K562 control mean CPM > min_ctrl_cpm,
    the scorer's DE gate) and never the target's own gene (its knockdown is excluded by every metric). z is the
    source's lfc / se (ResponseSet.z), the same z the effect model gates on.

    Returns a pandas Series (index = the targets K562 measured, in the order given; value = the count). Memory: one
    (n targets x G) float32 array (~22 MB for 300 targets), because only those rows of K562 are taken.
    """
    sub = k562.subset(targets)                          # rows of the targets K562 has (others skipped silently)
    if not sub.targets:
        return pd.Series([], dtype=np.int64, name="n_strong")
    strong = np.abs(sub.z()) > strong_z                 # (T, G) bool; z is 0 where K562 did not measure the gene
    testable = np.asarray(k562.measured, dtype=bool) & (np.asarray(k562.ctrl_cpm, dtype=np.float64) > min_ctrl_cpm)
    strong[:, ~testable] = False
    gidx = {g: j for j, g in enumerate(np.asarray(k562.genes).astype(str))}
    for i, t in enumerate(sub.targets):                 # the knockdown itself is not a "response"
        j = gidx.get(str(t))
        if j is not None:
            strong[i, j] = False
    return pd.Series(strong.sum(axis=1).astype(np.int64), index=list(sub.targets), name="n_strong")


def proxy_strength_filter(k562, h1_targets, max_strong_genes, strong_z, min_ctrl_cpm=DE_GATE_CPM):
    """Decide which H1 targets may enter the proxy (see the module docstring for why).

    Returns (table, allowed):
      table    DataFrame indexed by every H1 target: in_k562 (bool), n_strong (count, NaN if not in K562),
               excluded (bool: in K562 and n_strong > max_strong_genes)
      allowed  list of the H1 targets that are not excluded (H1's order), or None when the filter is off
               (max_strong_genes is None), meaning "no restriction"
    """
    h1_targets = [str(t) for t in h1_targets]
    counts = strong_gene_counts(k562, h1_targets, strong_z, min_ctrl_cpm)
    table = pd.DataFrame(index=pd.Index(h1_targets, name="target"))
    table["in_k562"] = table.index.isin(counts.index)
    table["n_strong"] = counts.reindex(table.index)               # NaN for targets K562 did not measure
    if max_strong_genes is None:
        table["excluded"] = False
        return table, None
    table["excluded"] = table["in_k562"] & (table["n_strong"] > float(max_strong_genes))
    allowed = [t for t in h1_targets if not table.at[t, "excluded"]]
    return table, allowed


# ---- steps ---------------------------------------------------------------------------------------------------
def step(title):
    print(f"\n=== {title} ===", flush=True)
    return time.time()


def done(t0):
    print(f"    ({time.time() - t0:,.0f} s)", flush=True)


def coverage_table(sources, panel):
    """How much of the VCC panel each source measured (the effect model can transfer only those targets)."""
    panel_set = set(panel)
    rows = []
    for rs in sources:
        t = set(map(str, rs.targets))
        n_cells = pd.to_numeric(rs.meta.get("n_cells"), errors="coerce") if "n_cells" in rs.meta else None
        rows.append({"source": rs.name, "targets": len(t), "genes measured": int(rs.measured.sum()),
                     "panel targets covered": f"{len(t & panel_set)}/{len(panel_set)}",
                     "median cells/target": float(n_cells.median()) if n_cells is not None else float("nan")})
    union = set().union(*[set(map(str, rs.targets)) for rs in sources]) if sources else set()
    rows.append({"source": "any source", "targets": len(union), "genes measured": None,
                 "panel targets covered": f"{len(union & panel_set)}/{len(panel_set)}",
                 "median cells/target": None})
    return pd.DataFrame(rows).set_index("source")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "v1.yaml"))
    ap.add_argument("--smoke", action="store_true", help="apply the config's smoke section (tiny laptop test)")
    ap.add_argument("--skip-proxy", action="store_true", help="do not build the H1 proxy")
    ap.add_argument("--reuse", action="store_true", help="keep outputs that already exist instead of rebuilding")
    args = ap.parse_args()

    cfg = load_config(args.config, args.smoke)
    P = cfg["paths"]
    files = input_files(cfg)
    missing = [str(p) for p in [files["k562_gw"], files["gtf"], *files["h1"]] if not p.exists()]
    if missing:
        sys.exit("missing input files (run scripts/00_download_data.sh first):\n  " + "\n  ".join(missing))
    genes, panel, contexts = load_bundle_info(P["controls_dir"])
    proc = P["processed_dir"]
    proc.mkdir(parents=True, exist_ok=True)
    print(f"config {args.config}{' (smoke)' if args.smoke else ''}\n  VCC axis: {genes.size:,} genes, panel "
          f"{len(panel)} targets, contexts {','.join(contexts)}\n  outputs: {proc} and {P['work_dir']}")

    def reuse(path):
        if args.reuse and Path(path).exists():
            print(f"    reusing {path}")
            return True
        return False

    # 1) K562 genome-wide screen (Replogle 2022 bulk): the main transfer source.
    t0 = step("K562 genome-wide (Replogle 2022, pseudobulk per target)")
    out = proc / "k562_gw.npz"
    if reuse(out):
        k562 = ResponseSet.load(out)
    else:
        k562 = load_replogle_bulk(files["k562_gw"], genes, "k562_gw")
        k562.save(out)
    print(f"    {len(k562.targets):,} targets, {int(k562.measured.sum()):,} of {genes.size:,} VCC genes measured")
    done(t0)

    # 2) H1 2025 (single-cell → pseudobulk). In validation it is only the TRUTH for fitting gains (never a source);
    #    the final submission model also uses it as a source.
    t0 = step("H1 hESC 2025 (Arc, single-cell → pseudobulk)")
    out = proc / "h1_2025.npz"
    if reuse(out):
        h1 = ResponseSet.load(out)
    else:
        h = cfg["sources"]["h1"]
        h1 = h1_pseudobulk(files["h1"], genes, name="h1_2025", min_cells=h.get("min_cells", 50),
                           max_cells_per_target=h.get("max_cells_per_target"), seed=h.get("seed", 0))
        h1.save(out)
    print(f"    {len(h1.targets):,} targets, {int(h1.measured.sum()):,} VCC genes measured")
    done(t0)

    # 3) Gene TSS table and the cis model (fitted on K562 only, so H1 stays a clean held-out line).
    t0 = step("TSS table (GENCODE) and cis model (fitted on K562)")
    out = proc / "tss.csv"
    if reuse(out):
        tss = pd.read_csv(out)
    else:
        tss = load_tss(files["gtf"])
        tss.to_csv(out, index=False)
    located = pd.Index(genes).isin(tss["gene_name"])
    on_axis = int(located.sum())
    print(f"    {len(tss):,} genes with a TSS; {on_axis:,} of {genes.size:,} VCC genes located")
    if on_axis < 0.99 * genes.size:
        # Usually an annotation release whose symbols differ from the VCC axis (10x GRCh38-2020-A = GENCODE v32):
        # renamed genes get no TSS, so they can never take part in the cis prior (as targets or as neighbours).
        lost = genes[~located]
        print(f"    WARNING: only {on_axis / genes.size:.1%} of the VCC genes have a TSS (< 99%); "
              f"{lost.size:,} are missing, e.g. {', '.join(map(str, lost[:12]))}. Is data.gencode the release "
              "that matches the VCC gene symbols (GENCODE v32)? If you change it, delete tss.csv and cis.json in "
              f"{proc} and rerun with --reuse.", flush=True)
    out = proc / "cis.json"
    if reuse(out):
        cis = CisModel.from_json(out.read_text())
    else:
        c = cfg.get("cis", {})
        cis = CisModel(bins_kb=c.get("bins_kb", CisModel().bins_kb)).fit(
            k562, tss, min_cells=c.get("min_cells", 50), min_ctrl_cpm=c.get("min_ctrl_cpm"),
            min_abs_effect=c.get("min_abs_effect", 0.0))
        out.write_text(cis.to_json())
    print("    cis effect per TSS-distance bin (median K562 log2FC of neighbouring genes):")
    print("      " + cis.summary().to_string(index=False, float_format=lambda x: f"{x:.3f}").replace("\n", "\n      "))
    done(t0)

    # 4) The H1 proxy context for the official scorer.
    kept, strength = None, None
    if not args.skip_proxy:
        t0 = step("H1 proxy context (VCC format: 400 cells/target, 18.4k controls, ~20k UMI)")
        p = cfg["proxy"]
        pdir = P["work_dir"] / "h1_proxy"
        out, tfile = pdir / "real.h5ad", pdir / "targets.txt"

        # 4a) Strength filter: leave out H1 targets that respond far more strongly than any VCC panel target.
        #     Computed from K562 only (cheap: the H1 targets' rows of K562), before the expensive proxy build.
        max_strong, strong_z = p.get("max_strong_genes"), float(p.get("strong_z", 4.5))
        strength, allowed = proxy_strength_filter(k562, h1.targets, max_strong, strong_z)
        n_excl = int(strength["excluded"].sum())
        n_in = int(strength["in_k562"].sum())
        if max_strong is None:
            print("    strength filter: off (proxy.max_strong_genes is null)")
        else:
            nk = strength.loc[strength["in_k562"], "n_strong"]
            print(f"    strength filter: {n_excl} of the {len(strength)} H1 targets excluded: their K562 response "
                  f"has > {max_strong} genes with |z| > {strong_z:g} (genes above {DE_GATE_CPM:g} CPM in K562 "
                  f"controls; the VCC panel's maximum is ~42).\n    Per H1 target measured in K562 ({n_in}): median "
                  f"{nk.median() if len(nk) else float('nan'):.0f}, max {nk.max() if len(nk) else float('nan'):.0f} "
                  f"such genes. The {len(strength) - n_in} H1 targets K562 did not measure are kept.")
            if n_excl:
                worst = strength[strength["excluded"]].sort_values("n_strong", ascending=False)
                print("    excluded: " + ", ".join(f"{t} ({int(n)})" for t, n in worst["n_strong"].items()))

        if reuse(out) and tfile.exists():
            kept = tfile.read_text().split()
            # A proxy from an earlier run may predate the filter or use other thresholds: say so loudly, because
            # validation would then be tuned on the strong targets the filter is meant to remove.
            bad = [t for t in kept if t in strength.index and bool(strength.at[t, "excluded"])]
            if bad:
                print(f"    WARNING: the reused proxy contains {len(bad)} target(s) the strength filter excludes "
                      f"({', '.join(bad[:10])}{' …' if len(bad) > 10 else ''}). Delete {out} and rerun (with "
                      "--reuse for the other outputs) to rebuild it.", flush=True)
        else:
            kept = build_h1_proxy(files["h1"], genes, out, n_ctrl=p["n_ctrl"], kd_min=p["kd_min"],
                                  median_umi=p["median_umi"], seed=p.get("seed", 0), targets=allowed,
                                  max_targets=p.get("max_targets"))
            tfile.write_text("\n".join(kept) + "\n")
            # Record the filter next to the proxy it shaped (02_validate.py copies the summary into best_params).
            strength.assign(in_proxy=strength.index.isin(kept)).to_csv(pdir / "strength.csv")
            (pdir / "strength_filter.json").write_text(json.dumps({
                "max_strong_genes": max_strong, "strong_z": strong_z, "min_ctrl_cpm": DE_GATE_CPM,
                "n_h1_targets": len(strength), "n_in_k562": n_in, "n_excluded": n_excl,
                "excluded": strength.index[strength["excluded"]].tolist(),
            }, indent=2) + "\n")
        done(t0)

    # ---- summary ---------------------------------------------------------------------------------------------
    print("\n=== Summary ===")
    with pd.option_context("display.width", 140):
        print(coverage_table([k562, h1], panel).to_string())
    if kept is not None:
        k = set(map(str, k562.targets))
        print(f"\nH1 proxy: {len(kept)} targets; {sum(t in k for t in kept)} of them measured in K562 (validation "
              f"transfers only from K562), {sum(t in set(panel) for t in kept)} in the VCC panel; "
              f"{int(strength['excluded'].sum())} H1 targets left out by the strength filter")
    print("\nnext: python scripts/02_validate.py --config", args.config, "--smoke" if args.smoke else "[--gpu]")


if __name__ == "__main__":
    main()
